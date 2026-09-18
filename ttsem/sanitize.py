"""Run a Triton program without a GPU, every launch executed by the semantics.

    python sanitize.py program.py [-- args of the program]

``JITFunction.run`` is replaced for the run. Each launch is recorded from the CPU tensors it was
given, compiled to TTIR for the target, executed by the level-1 semantics on a copy of their
storage, and what it computed is written back into the tensors, so the program goes on with the
kernel's results. A launch that reads or writes outside the tensors it was given, or uses a
value the semantics leaves undefined, stops the program with a :class:`KernelFault`: the op, the
buffer, the distance, in words the author of the kernel (a person or an agent) can act on.

A GPU forgives most of these: the allocator rounds sizes up, so a store a few elements past the
end lands in padding and every test passes. ``TRITON_INTERPRET=1`` does not forgive them and
does not explain them either (``free(): invalid pointer``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path
from typing import Any

from ttsem import harness
import numpy as np
from ttsem import values
from ttsem.memory import MemoryFault
from triton.runtime.jit import JITFunction


@dataclasses.dataclass
class Fault:
    kind: str  # oob_write | oob_read | poison
    launch: int
    kernel: str
    op: str
    buffer: str | None = None  # the kernel parameter whose tensor is the nearest one
    elements_past_end: int | None = None
    elements_before_start: int | None = None
    detail: str = ""
    source_file: str = ""  # where the op comes from in the kernel's own source
    source_line: int = 0
    source_text: str = ""


@dataclasses.dataclass
class Report:
    verdict: str  # ok | oob_write | oob_read | poison | unsupported | error
    launches: int
    fault: Fault | None = None
    message_for_agent: str = ""
    unsupported: list[str] = dataclasses.field(default_factory=list)


class KernelFault(Exception):
    def __init__(self, fault: Fault) -> None:
        super().__init__(fault.detail)
        self.fault = fault


class _Stop(Exception):
    """The launch could not be judged (unsupported op, internal error): the run ends here."""

    def __init__(self, verdict: str, detail: str, unsupported: list[str]) -> None:
        super().__init__(detail)
        self.verdict, self.detail, self.unsupported = verdict, detail, unsupported


def _op_of(exc: BaseException) -> str:
    for note in getattr(exc, "__notes__", None) or []:
        if note.startswith("while evaluating "):
            return note[len("while evaluating ") :]
    return ""


_LOC = re.compile(r'loc\("([^"]+)":(\d+):\d+')


def _source_of(op_text: str) -> tuple[str, int, str]:
    """(file, line, text of the line) the op was compiled from, read off its `loc`."""
    m = _LOC.search(op_text)
    if m is None:
        return "", 0, ""
    path, line = m.group(1), int(m.group(2))
    try:
        text = Path(path).read_text().splitlines()[line - 1].strip()
    except (OSError, IndexError):
        text = ""
    return path, line, text


def _with_source(fault: Fault) -> Fault:
    fault.source_file, fault.source_line, fault.source_text = _source_of(fault.op)
    return fault


def _locate(record: harness.LaunchRecord, fault: MemoryFault) -> tuple[str | None, int, int]:
    """(nearest buffer, elements past its end, elements before its start) of a memory fault."""
    addrs = np.asarray(fault.addrs, dtype=np.int64)
    best: tuple[int, str, int, int] | None = None
    for name, base in record.bases.items():
        arr = record.pre[name]
        size = int(arr.size * arr.itemsize)
        item = int(arr.itemsize)
        past = addrs[addrs >= base + size]
        before = addrs[addrs < base]
        gap = int(
            min(
                past.min() - (base + size) if past.size else 1 << 62,
                base - before.max() if before.size else 1 << 62,
            )
        )
        n_past = int(np.unique((past - base) // item).size) if past.size else 0
        n_before = int(np.unique((base - before + item - 1) // item).size) if before.size else 0
        # only the side that is nearest to this buffer counts as its overrun
        if past.size and before.size:
            if past.min() - (base + size) <= base - before.max():
                n_before = 0
            else:
                n_past = 0
        if best is None or gap < best[0]:
            best = (gap, name, n_past, n_before)
    if best is None:
        return None, 0, 0
    return best[1], best[2], best[3]


def _message(fault: Fault) -> str:
    where = f"kernel `{fault.kernel}`, launch {fault.launch}"
    op = fault.op.split(" loc(")[0][:160]
    if fault.source_line:
        name = Path(fault.source_file).name
        op = f"{name}:{fault.source_line}: {fault.source_text}" if fault.source_text else op
    if fault.kind in ("oob_write", "oob_read"):
        verb = "writes" if fault.kind == "oob_write" else "reads"
        side = (
            f"{fault.elements_past_end} element(s) past the end"
            if fault.elements_past_end
            else f"{fault.elements_before_start} element(s) before the start"
        )
        return (
            f"{where}: `{op}` {verb} {side} of the tensor passed as `{fault.buffer}`. "
            "A GPU run usually hides this (allocations are rounded up). Check the mask of this "
            "access against the tensor's real size, for every program id and for sizes that are "
            "not a multiple of the block."
        )
    return (
        f"{where}: `{fault.op[:160]}` uses a value the semantics leaves undefined: {fault.detail}"
    )


def _write_back(record: harness.LaunchRecord, execution: harness.Execution) -> None:
    import torch

    leaves = [
        (leaf, v)
        for name, value in record.bound.items()
        for leaf, v in harness._flat_args(name, value)
    ]
    done: set[int] = set()
    for leaf, value in leaves:
        if leaf not in record.ptrs or not harness._is_tensor_like(value):
            continue
        base = record.bases.get(leaf, record.ptrs[leaf])
        got = execution.buffer(record, leaf)
        if got is None or base in done:
            continue
        done.add(base)
        tensor = harness._unwrap(value)
        flat = torch.empty(0, dtype=torch.uint8, device=tensor.device)
        flat.set_(tensor.untyped_storage())
        raw = np.ascontiguousarray(got).view(np.uint8).reshape(-1)
        flat[: raw.size].copy_(torch.from_numpy(raw.copy()))


def run_program(
    program: Path, argv: list[str] | tuple[str, ...] = (), cc: int = harness.DEFAULT_CC
) -> Report:
    """Run ``program`` (a script with ``main()``) on the CPU under the semantics."""
    harness._install_fake_driver(cc)
    target = harness.target_for("cpu", cc)
    launches = [0]
    orig_run = JITFunction.run

    def run(self: JITFunction, *args: Any, grid: Any, warmup: bool, **kwargs: Any) -> Any:
        if warmup:
            return None
        launch = launches[0]
        launches[0] += 1
        recorded = harness.record_launch(
            self, args, kwargs, grid, lambda *a, **k: None, str(program), launch
        )
        record = recorded.record
        ir_text = harness.ir_for_launch(record, "ttir", target)
        try:
            execution = harness.execute(record, ir_text, raise_faults=True)
        except MemoryFault as exc:
            buffer, past, before = _locate(record, exc)
            kind = "oob_write" if exc.kind == "write" else "oob_read"
            fault = Fault(kind, launch, record.fn_name, _op_of(exc), buffer, past, before, str(exc))
            raise KernelFault(_with_source(fault)) from exc
        except values.Poison as exc:
            raise KernelFault(
                _with_source(Fault("poison", launch, record.fn_name, _op_of(exc), detail=str(exc)))
            ) from exc
        if execution.failure is not None:
            failure = execution.failure
            raise _Stop(failure.verdict, failure.message, list(failure.unsupported or []))
        _write_back(record, execution)
        return None

    JITFunction.run = run
    try:
        harness._run_program(program, "cpu", tuple(argv))
    except KernelFault as stop:
        fault = stop.fault
        return Report(fault.kind, launches[0], fault, _message(fault))
    except _Stop as stop:
        return Report(stop.verdict, launches[0], None, stop.detail, stop.unsupported)
    finally:
        JITFunction.run = orig_run
    return Report("ok", launches[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("program", type=Path)
    ap.add_argument("--cc", type=int, default=harness.DEFAULT_CC)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="arguments of the program, after --")
    args = ap.parse_args()
    rest = [a for a in args.rest if a != "--"]
    report = run_program(args.program, rest, args.cc)
    print(f"{report.verdict}: {report.launches} launch(es)")
    if report.message_for_agent:
        print(report.message_for_agent)
    if args.json:
        args.json.write_text(json.dumps(dataclasses.asdict(report), indent=1))
    return 0 if report.verdict == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
