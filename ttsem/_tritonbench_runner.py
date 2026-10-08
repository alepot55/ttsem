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
import os
import runpy
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ttsem import _guard, harness, pidraces, sanitize
from ttsem import memory as memory_mod
from ttsem._judge_runner import answer_frame, blame, cap_memory, first_line, raised_by_answer
from ttsem.judge_io import Channel


def plain(value: Any, arrays: list[np.ndarray]) -> Any:
    """`value` with every tensor or array replaced by `{"array": i}`, the array appended to
    `arrays`; containers kept (tuples as lists), NumPy scalars as Python ones, other objects by
    their type's name (which the judge does not compare)."""
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
        arrays.append(value)
        return {"array": len(arrays) - 1}
    if isinstance(value, dict):
        return {"dict": [[str(k), plain(v, arrays)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"list": [plain(v, arrays) for v in value]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"object": type(value).__name__}


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
    cap_memory()
    script: Path = args.script
    report: dict[str, Any] = {"file": script.name, "run": "ok", "launches": 0}
    own = {script.resolve()}
    live: dict[str, Any] = {}
    start = time.time()
    torch.manual_seed(0)
    sys.argv = [str(script)]
    results: Any = None
    from triton.runtime.jit import JITFunction

    guard = _guard.checker_integrity(
        memory=memory_mod, harness=harness, pidraces=pidraces, jitfunction=JITFunction, os_module=os
    )

    def on_launch(n: int) -> None:
        """The count over the channel (out of a forged report's reach), and a check the file did
        not swap the checker's own code before the semantics ran this launch."""
        bad = guard.breach(skip=_guard.HOOK)
        if bad is not None:
            raise sanitize.Unjudged(
                "tampered", f"the file changed the judge's own code ({bad})", []
            )
        channel.launch(n)

    def on_stop(stop: sanitize.KernelFault | sanitize.Unjudged) -> None:
        """A launch faulted, or cannot be judged: the judge hears it now, and the file never
        gets control back. The tool's own exception raised by the file's code is its error."""
        if isinstance(stop, sanitize.Unjudged) and raised_by_answer(stop, own):
            return
        if isinstance(stop, sanitize.KernelFault):
            event = fault_event(stop.fault)
        else:
            event = {"kind": "unjudged", "run": stop.verdict, "message": stop.detail[:300]}
            event["ops"] = stop.unsupported
        state = live.get("state")
        if state is not None:
            event |= {"launches": state.launches, "races_unchecked": state.races_unchecked}
        channel.stop(event)

    if "_inductor" in script.read_text(encoding="utf-8", errors="replace"):
        sanitize.inductor_names()  # code in Inductor's style: `grid`, `_empty_strided_cuda`
    with sanitize.session(source=str(script), on_stop=on_stop, on_launch=on_launch) as state:
        live["state"] = state
        guard.arm()  # the session hook is in place; snapshot before the file runs
        try:
            scope = runpy.run_path(str(script), run_name="__main__")
            results = scope.get(results_name(script))
        except sanitize.KernelFault as stop:  # raised outside a launch: the file's own doing
            channel.stop({**fault_event(stop.fault), "launches": state.launches})
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
        # before the account is trusted: the launch hook included now, a last check the file did
        # not swap the checker's own code without a watched launch seeing it
        bad = guard.breach()
        if bad is not None:
            report.update(run="tampered", message=f"the file changed the judge's own code ({bad})")
    if report["run"] == "ok":
        arrays: list[np.ndarray] = []
        report["results"] = plain(results, arrays)
        np.savez(args.out_npz, *arrays)
    report["seconds"] = round(time.time() - start, 1)
    args.out_json.write_text(json.dumps(report, indent=1))
    channel.finish(0)


if __name__ == "__main__":
    sys.exit(main())
