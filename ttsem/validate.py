"""Find the first pass that breaks a launch's semantics.

``MLIR_ENABLE_DUMP=1`` prints the whole module before every pass in the pipeline runs
(``split_dump`` parses that stream); each dump is run, independently, through this package's
own :class:`ttsem.interp.Interp` on the same inputs a real launch used
(:func:`ttsem.harness.capture_launches`), and compared with the recorded device/interpreter
output (:func:`ttsem.harness.run_launch`). The first pass whose output stops matching is the
culprit: this is the whole point of having an executable semantics at all.

Capturing the dump needs a real compile to happen, and on a machine without a GPU that compile
still needs the fake driver (:func:`ttsem.harness._install_fake_driver`) and must never try to
actually launch the compiled kernel; see ``_compile_runner.py``, run as a subprocess so
``MLIR_ENABLE_DUMP``'s stderr stream is captured in isolation.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ttsem import harness

try:
    from ttsem import ops  # type: ignore[import-not-found]
except ImportError:
    ops = None  # type: ignore[assignment]

_COMPILE_RUNNER = "ttsem._compile_runner"

_HEADER_PREFIX = "// -----// IR Dump "
_HEADER_SUFFIX = "//----- //"
_OP_RE = re.compile(r"^'([^']+)' operation(?:: *(\S+))?$")


@dataclasses.dataclass
class PassResult:
    """The verdict of running one pass's dump through :func:`harness.run_launch`."""

    pass_name: str
    verdict: str  # "match" | "mismatch" | "unsupported" | "error"
    n_diff: int
    unsupported: list[str]
    message: str = ""
    extra_reads: int = 0  # bytes this stage loads that the first decided stage did not


# The continuation below the first `llvm.func`: it receives the launch and the remaining
# `(pass_name, module_text)` stages, LLVM dialect first, and returns their verdicts in the same
# shape, so that `culprit_of` names one pass over the whole chain down to PTX.
BelowLLVM = Callable[["harness.LaunchRecord", list[tuple[str, str]]], list["PassResult"]]


@dataclasses.dataclass
class Report:
    program: str
    passes: list[PassResult]
    first_bad_pass: str | None
    culprit: str | None = None  # the pass after which the IR agrees with a wrong device
    kernel: str = ""  # the tt.func the launch ran; joins with static tables
    changed_by: str | None = None  # the pass whose output first disagrees with the reference


def _parse_header(line: str) -> tuple[str, str, str | None] | None:
    """``(pass_descriptor, op_kind, symbol)`` for one dump header line, or ``None``.

    A header looks like::

        // -----// IR Dump Before CanonicalizerPass: canonicalize{...}
            ('builtin.module' operation) //----- //
        // -----// IR Dump Before SomeFuncPass ('tt.func' operation: @kernel) //----- //
    """
    line = line.strip()
    if not (line.startswith(_HEADER_PREFIX) and line.endswith(_HEADER_SUFFIX)):
        return None
    middle = line[len(_HEADER_PREFIX) : -len(_HEADER_SUFFIX)].strip()
    if middle.startswith("Before "):
        when, rest = "Before", middle[len("Before ") :]
    elif middle.startswith("After "):
        when, rest = "After", middle[len("After ") :]
    else:
        return None
    paren = rest.rfind(" (")
    if paren == -1 or not rest.endswith(")"):
        return None
    pass_descr, op_descr = rest[:paren].strip(), rest[paren + 2 : -1].strip()
    m = _OP_RE.match(op_descr)
    if not m:
        return None
    op_kind, symbol = m.group(1), m.group(2)
    return f"{when} {pass_descr}", op_kind, symbol


def _starts_module(line: str) -> bool:
    """A module's first line, in the pretty form (`module {`) or the generic one
    (`"builtin.module"() ({`)."""
    return line.startswith("module") or line.startswith('"builtin.module"')


def _split_modules(lines: list[str]) -> list[str]:
    """One text per ``module`` printed in a dump body.

    A pass that dumps its intermediate steps (``tritongpu-pipeline{dump-intermediate-steps}``)
    prints several modules under one header. Each starts at a line beginning with ``module``;
    the alias lines (``#blocked = ...``, ``#loc = ...``) printed just before it belong to it.
    """
    starts = [i for i, line in enumerate(lines) if _starts_module(line)]
    if not starts:
        return ["\n".join(lines).strip("\n")]
    heads: list[int] = []
    for k, start in enumerate(starts):
        floor = starts[k - 1] + 1 if k else 0
        head = start
        while head > floor and lines[head - 1].startswith("#"):
            head -= 1
        heads.append(head)
    bounds = heads[1:] + [len(lines)]
    chunks = []
    for h, e in zip(heads, bounds, strict=True):
        body = lines[h:e]
        # the step header the pass prints before its next module (`// ...`) and blank lines
        # belong to nobody: left in, two identical modules would differ by a comment
        while body and (not body[-1].strip() or body[-1].lstrip().startswith("//")):
            body.pop()
        chunks.append("\n".join(body).strip("\n"))
    return chunks


# A stage whose printed text the compiler itself cannot re-read is not the semantics' failure:
# the reason is named and the stage is `unsupported`, not `error`.
_UNREADABLE = (
    (
        "cannot name an operation with no results",
        "pretty printer: nvws.warp_group drops its result types (triton#11752)",
    ),
    (
        "region with at least 1 blocks",
        "transient IR of a nested pass pipeline (an empty warp_specialize.partitions)",
    ),
)


def _unreadable_stage(message: str) -> str | None:
    for signature, reason in _UNREADABLE:
        if signature in message:
            return reason
    return None


def split_dump(text: str) -> list[tuple[str, str]]:
    """``(pass_name, module_text)`` for every *whole-module* dump, in order.

    Per-function dumps (``('tt.func' operation: @kernel)``) are dropped: only a
    ``'builtin.module' operation`` dump is a full module ``mlir.parse`` can run.
    """
    modules: list[tuple[str, str]] = []
    pass_name: str | None = None
    body: list[str] = []
    keep = False

    def flush() -> None:
        if keep and pass_name is not None:
            for k, chunk in enumerate(_split_modules(body)):
                name = pass_name if k == 0 else f"{pass_name} (step {k})"
                modules.append((name, chunk))

    for line in text.splitlines():
        header = _parse_header(line)
        if header is not None:
            flush()
            pass_name, op_kind, _symbol = header
            body = []
            keep = op_kind == "builtin.module"
            continue
        if keep:
            body.append(line)
    flush()
    return modules


def capture_dump(
    program_path: Path,
    device: str,
    cc: int = harness.DEFAULT_CC,
    timeout: float = 300.0,
) -> str:
    """Run ``program_path`` with ``MLIR_ENABLE_DUMP=1`` in a subprocess; return its stderr.

    The subprocess switches the wheel's printer to the generic form first, so the dump parses
    without a ``triton-opt`` round trip on any machine."""
    if device not in ("cpu", "cuda"):
        raise ValueError(f"unsupported device {device!r}")
    env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
    env["MLIR_ENABLE_DUMP"] = "1"
    with tempfile.TemporaryDirectory(prefix="ttsem-dump-") as tmp:
        env["TRITON_CACHE_DIR"] = tmp
        cmd = [sys.executable, "-m", _COMPILE_RUNNER, str(program_path), "--cc", str(cc)]
        cmd += ["--device", device, "--generic"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    return result.stderr


def _all_ops_unsupported(module: Any) -> bool:
    """True when no op anywhere in ``module`` is one :mod:`ops` implements.

    Independent of :class:`interp.Interp`'s own short-circuiting (it may stop at the first
    unsupported op, which says nothing about the ops after it): this walks the parsed module
    directly, so a pass dump that is entirely LLVM dialect (nothing left to model at level 1)
    can be recognised without running it at all.
    """
    if ops is None:
        return False

    def walk(op_list: list[Any]) -> bool:
        for op in op_list:
            if op.name in ops.OPS:
                return True
            for region in op.regions:
                for block in region.blocks:
                    if walk(block.ops):
                        return True
        return False

    for fn_op in module.funcs.values():
        for region in fn_op.regions:
            for block in region.blocks:
                if walk(block.ops):
                    return False
    return True


def validate(
    program_path: Path,
    record: harness.LaunchRecord,
    device: str = "cpu",
    triton_opt: str | None = None,
    cc: int = harness.DEFAULT_CC,
) -> Report:
    """Run every pass dump of ``record``'s launch through :func:`harness.run_launch`."""
    dump_text = capture_dump(program_path, device, cc)
    modules = split_dump(dump_text)
    if not modules:
        return Report(program_path.name, [], None)
    return validate_stages(record, modules, triton_opt, program_path.name)


def validate_stages(
    record: harness.LaunchRecord,
    stages: Iterable[tuple[str, str]],
    triton_opt: str | None = None,
    name: str = "",
    below_llvm: BelowLLVM | None = None,
) -> Report:
    """The data API: ``stages`` are ``(pass_name, module_text)`` pairs in pipeline order, from
    any source (an ``MLIR_ENABLE_DUMP`` trace, a pass-by-pass replay, a single module). Each
    module is run on ``record``'s pre-launch state and compared with its post-launch state;
    the report names the first mismatching stage and the culprit boundary.

    The tile semantics stops at the first module that contains an ``llvm.func``. With
    ``below_llvm`` given, that module and every later stage are handed to it instead, and its
    verdicts continue the same list; without it they are one ``unsupported`` entry."""
    results: list[PassResult] = []
    baseline_reads = None
    todo = list(stages)
    for i, (pass_name, module_text) in enumerate(todo):
        if harness.mlir is None:
            results.append(PassResult(pass_name, "error", -1, [], "mlir.py is not available yet"))
            continue
        if "llvm.func" in module_text:
            if below_llvm is None:
                results.append(
                    PassResult(pass_name, "unsupported", -1, ["<llvm dialect>"], "past lowering")
                )
            else:
                results.extend(below_llvm(record, todo[i:]))
            break
        try:
            generic = harness.mlir.to_generic(module_text, triton_opt)
            module = harness.mlir.parse(generic)
        except Exception as e:
            reason = _unreadable_stage(str(e))
            if reason is not None:
                results.append(
                    PassResult(pass_name, "unsupported", -1, [reason], f"parse failed: {e}")
                )
            else:
                results.append(PassResult(pass_name, "error", -1, [], f"parse failed: {e!r}"))
            continue
        if _all_ops_unsupported(module):
            results.append(PassResult(pass_name, "unsupported", -1, ["<all ops>"], "past lowering"))
            break
        cmp = harness.run_launch(record, generic, trace_reads=True)
        extra = 0
        if cmp.reads is not None:
            if baseline_reads is None:
                baseline_reads = cmp.reads
            else:
                extra = int(np.setdiff1d(cmp.reads, baseline_reads).size)
        results.append(
            PassResult(
                pass_name, cmp.verdict, cmp.n_diff, cmp.unsupported, cmp.message, extra_reads=extra
            )
        )

    first_bad = next((r.pass_name for r in results if r.verdict == "mismatch"), None)
    idx = changed_by_index(results)
    changed_by = results[idx].pass_name if idx is not None else None
    return Report(name, results, first_bad, culprit_of(results), record.fn_name, changed_by)


def changed_by_index(results: list[PassResult]) -> int | None:
    """The index of the last stage that agrees with the reference before the first one that
    disagrees, or ``None``.

    The mirror of :func:`culprit_of`. A dump named "Before P" is the IR that P receives; when
    it agrees with the reference and the next decided stage does not, P's output means something
    P's input did not. This is the reading when the reference computes what the source meant
    (the interpreter on the cpu path); :func:`culprit_of` is the reading when the reference is a
    device that agrees with the end of the pipeline.
    """
    decided = [
        (i, r) for i, r in enumerate(results) if r.verdict in ("match", "approx", "mismatch")
    ]
    for (i, r), (_, nxt) in zip(decided, decided[1:], strict=False):
        if r.verdict != "mismatch" and nxt.verdict == "mismatch":
            return i
    return None


def culprit_of(results: list[PassResult]) -> str | None:
    """The pass that made the IR agree with a device that disagrees with the source.

    A dump named "Before P" is the IR that P receives. When the stages read mismatch up to
    "Before P" and match from the next decided stage on, the device computes what P's output
    means and not what P's input meant: P changed the program. Stages that were not decided
    (unsupported, error, poison) are skipped when looking for the first agreeing one.
    """
    decided = [r for r in results if r.verdict in ("match", "approx", "mismatch")]
    for i, r in enumerate(decided):
        if r.verdict != "mismatch":
            continue
        later = decided[i + 1 :]
        if later and all(x.verdict != "mismatch" for x in later):
            return r.pass_name
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("program", type=Path)
    ap.add_argument("--triton-opt", default=None)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    ap.add_argument("--cc", type=int, default=harness.DEFAULT_CC)
    ap.add_argument("--json", type=Path, default=None, help="write the reports here")
    args = ap.parse_args()

    records = harness.capture_launches(args.program, args.device, args.cc)
    if not records:
        print("no launches captured")
        return 1

    exit_code = 0
    launches: list[dict[str, Any]] = []
    for record in records:
        report = validate(
            args.program, record, device=args.device, triton_opt=args.triton_opt, cc=args.cc
        )
        print(f"=== launch {record.launch_id} ({record.fn_name}) ===")
        for r in report.passes:
            extra = f" extra_reads={r.extra_reads}" if r.extra_reads else ""
            print(f"  {r.verdict:<12} n_diff={r.n_diff:<4} {r.pass_name}  {r.message}{extra}")
        print(f"  first bad pass: {report.first_bad_pass}")
        if report.changed_by:
            print(f"  changed by: {report.changed_by} (its output is the first to disagree)")
        if report.culprit is not None:
            print(f"  culprit: {report.culprit} (the device agrees with the IR only after it)")
        if report.first_bad_pass is not None:
            exit_code = 1
        launches.append(
            {
                "launch_id": record.launch_id,
                "fn_name": record.fn_name,
                "passes": [dataclasses.asdict(r) for r in report.passes],
                "first_bad_pass": report.first_bad_pass,
                "culprit": report.culprit,
                "kernel": report.kernel,
            }
        )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"program": args.program.name, "launches": launches}, indent=1)
        )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
