"""Run a Triton program without a GPU, every launch executed by the semantics.

    python -m ttsem.sanitize program.py [-- args of the program]
    python -m ttsem.sanitize pytest tests/ -q         # a test suite written for a GPU, as it is

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
import contextlib
import dataclasses
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ttsem import harness
import numpy as np
from ttsem import pidraces
from ttsem import values
from ttsem.memory import MemoryFault
from triton.runtime.jit import JITFunction


@dataclasses.dataclass
class Fault:
    kind: str  # oob_write | oob_read | poison | race
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
    verdict: str  # ok | oob_write | oob_read | poison | race | unsupported | error
    launches: int
    fault: Fault | None = None
    message_for_agent: str = ""
    unsupported: list[str] = dataclasses.field(default_factory=list)
    races_unchecked: int = 0


class KernelFault(Exception):
    """Raised out of the launching call. Its text is the message for the author of the kernel,
    so a test runner that only prints the exception already says what to fix."""

    def __init__(self, fault: Fault) -> None:
        super().__init__(fault.detail)
        self.fault = fault

    def __str__(self) -> str:
        return message(self.fault)


class Unjudged(Exception):
    """The launch could not be judged (unsupported op, internal error): the run ends here."""

    def __init__(self, verdict: str, detail: str, unsupported: list[str]) -> None:
        super().__init__(detail)
        self.verdict, self.detail, self.unsupported = verdict, detail, unsupported


def _op_of(exc: BaseException) -> str:
    for note in getattr(exc, "__notes__", None) or []:
        if note.startswith("while evaluating "):
            return note[len("while evaluating ") :]
    return ""


_LOC = re.compile(r'loc\((?:"[^"]*"\()?"([^"]+)":(\d+):\d+')  # plain, or named: loc("x"("f":1:2))
_ALIAS_USE = re.compile(r"loc\((#loc\d*)\)\s*$")
_ALIAS_DEF = re.compile(r"^(#loc\d*) = loc\((.*)\)\s*$", re.M)


def _source_of(op_text: str, ir_text: str = "") -> tuple[str, int, str]:
    """(file, line, text of the line) the op was compiled from, read off its `loc`. The printer
    writes most locations as aliases (`loc(#loc7)`, defined at the end of the module, possibly
    through a name: `#loc7 = loc("x"(#loc3))`), so the module's text resolves them."""
    m = _LOC.search(op_text)
    use = _ALIAS_USE.search(op_text)
    if m is None and use is not None and ir_text:
        table = dict(_ALIAS_DEF.findall(ir_text))
        alias, body = use.group(1), ""
        for _ in range(8):  # name -> name -> file:line, never deep
            body = table.get(alias, "")
            m = _LOC.search(f"loc({body}")
            inner = re.search(r"\((#loc\d*)\)", body)
            if m is not None or inner is None:
                break
            alias = inner.group(1)
    if m is None:
        return "", 0, ""
    path, line = m.group(1), int(m.group(2))
    try:
        text = Path(path).read_text().splitlines()[line - 1].strip()
    except (OSError, IndexError):
        text = ""
    return path, line, text


def _with_source(fault: Fault, ir_text: str = "") -> Fault:
    fault.source_file, fault.source_line, fault.source_text = _source_of(fault.op, ir_text)
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


def _race_fault(
    record: harness.LaunchRecord, race: pidraces.PidRace, launch: int, ir_text: str = ""
) -> Fault:
    """The race as a fault on the storing side, with the buffer and the element it is about."""
    buffer, element = None, None
    for name, base in record.bases.items():
        arr = record.pre[name]
        if base <= race.address < base + arr.size * arr.itemsize:
            buffer, element = name, int((race.address - base) // arr.itemsize)
    sides = sorted((race.first, race.second), key=lambda side: side[1] == "r")  # a writer first
    (pid_a, kind_a, op_a), (pid_b, kind_b, op_b) = sides
    words = {"r": "loads", "w": "stores to", "a": "atomically updates"}
    other = _source_of(op_b, ir_text)
    other_where = f"{Path(other[0]).name}:{other[1]}: {other[2]}" if other[1] else op_b[:120]
    detail = (
        f"program instances {pid_a} and {pid_b} both touch element {element} of `{buffer}`: "
        f"{pid_a} {words[kind_a]} it, {pid_b} {words[kind_b]} it (`{other_where}`). "
        f"{race.shared_bytes} byte(s) are shared this way ({race.kind})."
    )
    return Fault("race", launch, record.fn_name, op_a, buffer, detail=detail)


def message(fault: Fault) -> str:
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
    if fault.kind == "race":
        return (
            f"{where}: `{op}`: {fault.detail} Program instances run in any order and at the same "
            "time, so the result depends on the schedule; run one after the other it looks "
            "right, and an output comparison cannot see it. Make the instances write disjoint "
            "elements, or use an atomic update (`tl.atomic_add` and the like) for the shared ones."
        )
    return f"{where}: `{op}` uses a value the semantics leaves undefined: {fault.detail}"


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


def _is_cuda(device: Any) -> bool:
    return str(device).startswith("cuda") if device is not None else False


def _cuda_is_cpu() -> Any:
    """A torch function mode that sends every `device="cuda"` to the CPU, so that a program or
    a test suite written for a GPU runs unmodified on a machine without one, and that poisons
    the memory `torch.empty` and its kin return. Nothing else changes: dtypes, shapes, strides
    and values are torch's own."""
    import torch
    from torch.overrides import TorchFunctionMode

    cpu = torch.device("cpu")
    uninitialised = {"empty", "empty_like", "empty_strided", "new_empty", "new_empty_strided"}

    def poison(tensor: Any) -> Any:
        """Memory nobody wrote holds NaN (a loud pattern for integers): `torch.empty` returns
        zero pages often enough, on a GPU too, that a kernel reading its output before writing
        it passes its tests."""
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return tensor
        if tensor.dtype.is_floating_point or tensor.dtype.is_complex:
            tensor.fill_(float("nan"))
        elif tensor.dtype == torch.bool:
            tensor.fill_(True)
        else:
            tensor.fill_(torch.iinfo(tensor.dtype).max - 0x5A)
        return tensor

    class CudaIsCpu(TorchFunctionMode):
        def __torch_function__(
            self, func: Any, types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            kwargs = dict(kwargs or {})
            if _is_cuda(kwargs.get("device")):
                kwargs["device"] = cpu
            name = getattr(func, "__name__", "")
            if name in uninitialised:
                return poison(func(*args, **kwargs))
            if name == "cuda" and args and isinstance(args[0], torch.Tensor):
                return args[0]
            if name == "to" and args and isinstance(args[0], torch.Tensor):
                args = tuple(
                    cpu if _is_cuda(a) and not isinstance(a, torch.Tensor) else a for a in args
                )
            return func(*args, **kwargs)

    return CudaIsCpu()


def _pretend_cuda() -> Any:
    """What a GPU program asks of torch before it launches anything and the fake driver's stub of
    `torch.cuda` does not answer: `assert x.is_cuda`, a device guard, `set_device`. While the
    session is open every tensor is on the one pretend device. Returns the undo."""
    import torch

    class NoDevice(contextlib.nullcontext):  # type: ignore[type-arg]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()

    answers: dict[str, Any] = {
        "device": NoDevice,
        "_DeviceGuard": NoDevice,
        "set_device": lambda device=None: None,
    }
    saved = {name: getattr(torch.cuda, name, None) for name in answers}
    for name, answer in answers.items():
        setattr(torch.cuda, name, answer)
    torch.Tensor.is_cuda = property(lambda self: True)  # type: ignore[assignment,misc]
    had_stream = hasattr(torch._C, "_cuda_getCurrentRawStream")
    if not had_stream:  # imported by name by kernels that Inductor generated
        torch._C._cuda_getCurrentRawStream = lambda device=0: 0  # type: ignore[attr-defined]

    def undo() -> None:
        del torch.Tensor.is_cuda
        if not had_stream:
            del torch._C._cuda_getCurrentRawStream
        for name, original in saved.items():
            if original is not None:
                setattr(torch.cuda, name, original)

    return undo


def _no_benchmarks() -> Any:
    """Timing means nothing under an interpreter. A benchmark call runs its function once (so
    every configuration an autotuner tries is still executed, and judged) and reports a
    constant; a `perf_report` is skipped with a line saying so. Returns the undo."""
    import triton.testing as testing
    from triton.runtime import driver

    def bench_once(fn: Any, *args: Any, quantiles: Any = None, **kwargs: Any) -> Any:
        fn()
        return [1.0] * len(quantiles) if quantiles else 1.0

    def skip_report(self: Any, *args: Any, **kwargs: Any) -> None:
        print("ttsem: benchmarks are not run without a GPU (perf_report skipped)")

    saved = (testing.do_bench, getattr(testing, "do_bench_cudagraph", None), testing.Mark.run)
    testing.do_bench = bench_once
    if saved[1] is not None:
        testing.do_bench_cudagraph = bench_once
    testing.Mark.run = skip_report
    active = driver.active
    had = getattr(active, "get_benchmarker", None)
    active.get_benchmarker = lambda: bench_once
    if not hasattr(active, "utils"):  # programs size their launches from the device's properties
        import types

        def properties(device: Any = 0) -> dict[str, Any]:
            # the keys of the NVIDIA driver; the register count is Blackwell's, the warp is 32
            return {"max_num_regs": 1 << 16, "warpSize": 32, **active.get_device_properties(device)}

        active.utils = types.SimpleNamespace(get_device_properties=properties)

    def undo() -> None:
        testing.do_bench = saved[0]
        if saved[1] is not None:
            testing.do_bench_cudagraph = saved[1]
        testing.Mark.run = saved[2]
        if had is not None:
            active.get_benchmarker = had

    return undo


@dataclasses.dataclass
class Session:
    launches: int = 0
    races_unchecked: int = 0  # launches too large for the access log (`Memory.access_budget`)


@contextlib.contextmanager
def session(
    cc: int = harness.DEFAULT_CC, source: str = "<program>", cuda_is_cpu: bool = True
) -> Iterator[Session]:
    """While open, every Triton launch of this process is executed by the semantics on the CPU
    tensors it was given and written back into them. A launch that faults raises
    :class:`KernelFault` out of the launching call. With ``cuda_is_cpu`` every request for a
    CUDA device gets the CPU, so code written for a GPU runs as it is."""
    harness._install_fake_driver(cc)
    restore_bench = _no_benchmarks()
    target = harness.target_for("cpu", cc)
    state = Session()
    orig_run = JITFunction.run

    def run(self: JITFunction, *args: Any, grid: Any, warmup: bool, **kwargs: Any) -> Any:
        __tracebackhide__ = True  # pytest: the failure is the caller's line, not ours
        if warmup:
            return None
        launch = state.launches
        state.launches += 1
        recorded = harness.record_launch(
            self, args, kwargs, grid, lambda *a, **k: None, source, launch
        )
        record = recorded.record
        ir_text = harness.ir_for_launch(record, "ttir", target)
        try:
            many = int(np.prod([int(g) for g in record.grid])) > 1
            execution = harness.execute(record, ir_text, raise_faults=True, trace_access=many)
        except MemoryFault as exc:
            buffer, past, before = _locate(record, exc)
            kind = "oob_write" if exc.kind == "write" else "oob_read"
            fault = Fault(kind, launch, record.fn_name, _op_of(exc), buffer, past, before, str(exc))
            raise KernelFault(_with_source(fault, ir_text)) from None
        except values.Poison as exc:
            raise KernelFault(
                _with_source(
                    Fault("poison", launch, record.fn_name, _op_of(exc), detail=str(exc)), ir_text
                )
            ) from None
        if execution.failure is not None:
            failure = execution.failure
            raise Unjudged(failure.verdict, failure.message, list(failure.unsupported or []))
        _write_back(record, execution)
        if execution.memory.access_overflow:
            state.races_unchecked += 1
        race = pidraces.find_race(execution.memory.access_log)
        if race is not None:
            raise KernelFault(_with_source(_race_fault(record, race, launch, ir_text), ir_text))
        return None

    JITFunction.run = run
    redirect = _cuda_is_cpu() if cuda_is_cpu else contextlib.nullcontext()
    restore_cuda = _pretend_cuda() if cuda_is_cpu else (lambda: None)
    try:
        with redirect:
            yield state
    finally:
        JITFunction.run = orig_run
        restore_cuda()
        restore_bench()


def report_of(fault: Fault, launches: int) -> Report:
    return Report(fault.kind, launches, fault, message(fault))


def run_program(
    program: Path, argv: list[str] | tuple[str, ...] = (), cc: int = harness.DEFAULT_CC
) -> Report:
    """Run ``program`` (a script with ``main()``) on the CPU under the semantics."""
    with session(cc, str(program)) as state:
        try:
            harness._run_program(program, "cpu", tuple(argv))
        except KernelFault as stop:
            return report_of(stop.fault, state.launches)
        except Unjudged as stop:
            return Report(stop.verdict, state.launches, None, stop.detail, stop.unsupported)
    return Report("ok", state.launches, races_unchecked=state.races_unchecked)


def run_script(
    script: Path, argv: list[str] | tuple[str, ...] = (), cc: int = harness.DEFAULT_CC
) -> Report:
    """Run any Python script as ``python script.py argv...`` would, under the semantics."""
    import runpy

    old_argv = sys.argv
    sys.argv = [str(script), *argv]
    try:
        with session(cc, str(script)) as state:
            try:
                runpy.run_path(str(script), run_name="__main__")
            except KernelFault as stop:
                return report_of(stop.fault, state.launches)
            except Unjudged as stop:
                return Report(stop.verdict, state.launches, None, stop.detail, stop.unsupported)
    finally:
        sys.argv = old_argv
    return Report("ok", state.launches, races_unchecked=state.races_unchecked)


def run_pytest(args: list[str], cc: int = harness.DEFAULT_CC) -> int:
    """``pytest args...`` in this process with every launch under the semantics: a kernel fault
    is the failure of the test that launched it, with the message for the author as its text."""
    import pytest

    with session(cc, "<pytest>"):
        return int(pytest.main(list(args)))


def main() -> int:
    """`sanitize.py script.py [args]`, or `sanitize.py pytest [pytest args]`."""
    argv = sys.argv[1:]
    if argv and argv[0] == "pytest":
        return run_pytest(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("program", type=Path)
    ap.add_argument("--cc", type=int, default=harness.DEFAULT_CC)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="arguments of the program, after --")
    args = ap.parse_args()
    rest = [a for a in args.rest if a != "--"]
    has_main = "def main(" in args.program.read_text()
    runner = run_program if has_main else run_script
    report = runner(args.program, rest, args.cc)
    print(f"{report.verdict}: {report.launches} launch(es)")
    if report.races_unchecked:
        print(f"{report.races_unchecked} launch(es) too large to be checked for races")
    if report.message_for_agent:
        print(report.message_for_agent)
    if args.json:
        args.json.write_text(json.dumps(dataclasses.asdict(report), indent=1))
    return 0 if report.verdict == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
