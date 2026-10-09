"""Subprocess entry point for ``ttsem.judge_tritonbench``: one TritonBench file, run here.

Not meant to be imported: ``ttsem.judge_tritonbench`` runs it, under ``bwrap`` and with a timeout,
as

    python -m ttsem._tritonbench_runner FILE.py OUT.json OUT.npz [--events FD]

A TritonBench file is an operator, a line of 146 `#`, and the task's test, which calls the
operator on its test cases and keeps what it returns in a module-level variable (`result_gold` in
TritonBench-G, `test_results` in TritonBench-T). The file is executed as `python FILE.py` would
(seed 0 first), inside `sanitize.session()`: every Triton launch is compiled by the real frontend
and executed by ttsem on CPU tensors, and a launch that reads or writes out of bounds, races with
another program instance or performs an operation with an undefined result stops the run there,
with the kernel's line (memory nobody wrote is filled with NaN instead, so it shows in the values).

What the test kept is written as plain arrays: its structure (dicts, lists, scalars, strings) in
OUT.json and its tensors in OUT.npz (bfloat16 and float16 widened to float32), which the judge
reads back with `allow_pickle=False`. TritonBench itself compares the stdout of the two runs, and
its tests print nothing.

`run` in OUT.json: ok | unsupported | error (the semantics cannot go on) | memory (the run's
memory cap) | tool_error | call_error (the file raised, or raised the tool's own exception
itself). A fault (oob_read, oob_write, race, poison), or a launch the semantics cannot judge, is
not written there: it goes to the judge at once on the events pipe (`--events FD`) and the process
ends (`ttsem.judge_io.Channel`), so the file cannot catch it and go on as if the kernel had run. A
run that ends otherwise writes OUT.json and OUT.npz, sends `done` and leaves the same way: no exit
handler of the file's runs after it.
"""

from __future__ import annotations

import argparse
import ast
import json
import runpy
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ttsem import _guard, sanitize
from ttsem._judge_runner import answer_frame, blame, cap_memory, first_line, raised_by_answer
from ttsem.judge_io import Channel


def plain(value: Any, arrays: list[np.ndarray]) -> Any:
    """`value` with every tensor or array replaced by `{"array": i}`, the array appended to
    `arrays`; containers kept (tuples as lists), NumPy scalars as Python ones, other objects by
    their type's name (which the judge does not compare).

    Everything it returns is of the exact built-in types (an array is a plain ndarray, a string a
    plain str): this is the last time the run reads the file's objects, and writing the result
    afterwards runs none of their methods."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.dtype.is_floating_point and tensor.dtype not in (torch.float32, torch.float64):
            tensor = tensor.float()  # bfloat16 and float16 compared in float32
        value = np.ascontiguousarray(tensor.cpu().numpy())
    if isinstance(value, np.generic):
        item = value.item()
        if not isinstance(item, np.generic):
            return plain(item, arrays)
        value = np.asarray(value)  # a longdouble has no Python scalar: an array of one
    if isinstance(value, np.ndarray) and not value.dtype.hasobject:
        arrays.append(np.array(value, copy=True, subok=False))
        return {"array": len(arrays) - 1}
    if isinstance(value, dict):
        return {"dict": [[exact_str(k), plain(v, arrays)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"list": [plain(v, arrays) for v in value]}
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int.__int__(value)
    if isinstance(value, float):
        return float.__float__(value)
    if isinstance(value, str):
        return str.__str__(value)
    return {"object": exact_str(type(value).__name__)}


def exact_str(value: Any) -> str:
    """`str(value)` as a plain str (not a subclass with methods of its own)."""
    return str.__str__(str(value))


def results_name(script: Path) -> str:
    """The module-level name the file keeps its test's results in: `result_gold` in G,
    `test_results` in T, read off the last `name = test_...()` of the file."""
    name = "result_gold"
    try:
        tree = ast.parse(script.read_text(encoding="utf-8"))
    except (SyntaxError, ValueError, OSError):
        return name
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id.startswith("test")
        ):
            name = node.targets[0].id
    return name


def fault_event(fault: sanitize.Fault) -> dict[str, Any]:
    """A fault as the judge's run report gives it."""
    return {
        "kind": "fault",
        "run": fault.kind,
        "kernel": fault.kernel,
        "buffer": fault.buffer,
        "fault_file": Path(fault.source_file).name,
        "line": fault.source_line,
        "source": fault.source_text,
        "message": sanitize.message(fault)[:600],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("script", type=Path)
    ap.add_argument("out_json", type=Path)
    ap.add_argument("out_npz", type=Path)
    ap.add_argument("--events", type=int, help="the pipe to the judge (`judge_io.Channel`)")
    args = ap.parse_args(argv)
    channel = Channel(args.events)  # before anything of the file's is imported
    # its closures, held here
    stop, launch, finish = channel.stop, channel.launch, channel.finish
    unchecked = channel.unchecked
    # a SIGINT in the middle of a launch would raise KeyboardInterrupt into the file, which could
    # catch it and go on without the launch's verdict; nothing interrupts a judged run
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cap_memory()
    script: Path = args.script
    report: dict[str, Any] = {"file": script.name, "run": "ok", "launches": 0}
    own = {script.resolve()}
    guards: dict[str, Callable[[], None]] = {}  # the integrity checks, once armed
    start = time.time()
    torch.manual_seed(0)
    sys.argv = [str(script)]
    results: Any = None

    def on_launch(n: int) -> None:
        """The count over the channel, out of a forged report's reach."""
        launch(n)

    def on_check() -> None:
        """Around the semantics' run of each launch: the checker's own code is the code it was
        before the file ran, or the judge hears `tampered` and the run ends there."""
        enforce = guards.get("enforce")
        if enforce is not None:
            enforce()

    def on_stop(fault: sanitize.KernelFault | sanitize.Unjudged) -> None:
        """A launch faulted, or cannot be judged: the judge hears it now, and the file never
        gets control back. The tool's own exception raised by the file's code is its error."""
        if isinstance(fault, sanitize.Unjudged) and raised_by_answer(fault, own):
            return
        if isinstance(fault, sanitize.KernelFault):
            event = fault_event(fault.fault)
        else:
            event = {"kind": "unjudged", "run": fault.verdict, "message": fault.detail[:300]}
            event["ops"] = fault.unsupported
        stop(event)  # with the counts the channel kept: the launches, those not race-checked

    if "_inductor" in script.read_text(encoding="utf-8", errors="replace"):
        sanitize.inductor_names()  # code in Inductor's style: `grid`, `_empty_strided_cuda`
    # the session stays open to the end: the process leaves by the channel, and the snapshot
    # below includes what the session installed
    session = sanitize.session(
        source=str(script),
        on_stop=on_stop,
        on_launch=on_launch,
        on_check=on_check,
        on_unchecked=unchecked,  # a launch too large for the race check, counted by the channel
        inductor_lookback=False,  # an answer is untrusted: every buffer it touches is race-checked
    )
    state = session.__enter__()
    from triton.runtime.jit import JITFunction

    guard = _guard.checker_integrity(
        roots=[Path(_guard.__file__).resolve().parent],
        hooks={"JITFunction.run": JITFunction.run},
    )
    guards["enforce"] = _guard.enforcer(
        guard.arm(), stop, {"kind": "unjudged", "run": "tampered"}
    )  # the session hook is in place; the snapshot is taken before the file runs
    seal = _guard.sealer(guards["enforce"])
    try:
        scope = runpy.run_path(str(script), run_name="__main__")
        results = scope.get(results_name(script))
    except sanitize.KernelFault as fault:  # raised outside a launch: the file's own doing
        stop({**fault_event(fault.fault), "launches": state.launches})
    except MemoryError:
        report.update(run="memory", message="more memory than this run is allowed")
    except BaseException as exc:  # noqa: BLE001  the file under test may raise anything
        if isinstance(exc, sanitize.Unjudged) and not raised_by_answer(exc, own):
            # the semantics' own, out of a launch the session does not watch
            report.update(run=exc.verdict, message=exc.detail[:300], ops=exc.unsupported)
        else:
            inside, frame = blame(exc, own)
            where = answer_frame(exc, own) or frame  # a compile error: the file's own launch
            lines = [ln.strip() for ln in str(exc).strip().splitlines() if ln.strip()]
            # a compilation error says what went wrong last
            report.update(
                run="tool_error" if inside else "call_error",
                message=first_line(exc),
                detail=" | ".join(lines[-2:])[:300],
                where=f"{Path(where.filename).name}:{where.lineno}",
            )
    report["launches"] = state.launches
    report["races_unchecked"] = state.races_unchecked
    arrays: list[np.ndarray] = []
    if report["run"] == "ok":
        report["results"] = plain(results, arrays)  # the file's objects, read for the last time
    report["seconds"] = round(time.time() - start, 1)
    # the last code of the file's has run (its objects' finalizers too): the checker must still be
    # the one snapshotted, or the judge hears `tampered` and no report is written
    seal()
    if report["run"] == "ok":
        np.savez(args.out_npz, *arrays)
    args.out_json.write_text(json.dumps(report, indent=1))
    finish(0)


if __name__ == "__main__":
    sys.exit(main())
