"""One TritonBench file on a GPU, as the benchmark runs it: the audit's control.

    python -P audits/tritonbench/device_runner.py FILE.py OUT.json OUT.npz

Run by `device.py`, inside the judge's `bwrap` sandbox with the GPU's device nodes let in, never
outside it. Nothing of ttsem runs here: the file is executed as `python FILE.py` would (seed 0
first) on the real driver, and what its test kept (`result_gold` in TritonBench-G, `test_results`
in TritonBench-T) is written in the format of ``ttsem._tritonbench_runner`` (structure in OUT.json,
tensors in OUT.npz, bfloat16 and float16 widened to float32), so that
``ttsem.judge_tritonbench.decide`` reads both sides the same way. Kernel launches are counted as the
semantics counts them (`JITFunction.run`, warm-ups left out).

`run` in OUT.json: ok | call_error (the file raised).
"""

from __future__ import annotations

import ast
import json
import runpy
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def plain(value: Any, arrays: list[np.ndarray]) -> Any:
    """`value` with every tensor or array replaced by `{"array": i}`, the array appended to
    `arrays`; containers kept (tuples as lists), NumPy scalars as Python ones, other objects by
    their type's name. The same as ``ttsem._tritonbench_runner.plain``, which imports the
    semantics; the control must not."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.dtype.is_floating_point and tensor.dtype not in (torch.float32, torch.float64):
            tensor = tensor.float()
        value = np.ascontiguousarray(tensor.cpu().numpy())
    if isinstance(value, np.generic):
        item = value.item()
        if not isinstance(item, np.generic):
            return plain(item, arrays)
        value = np.asarray(value)
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
    """The module-level name the file keeps its test's results in, read off the last
    `name = test_...()` of the file (`result_gold` when there is none)."""
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


def first_line(exc: BaseException) -> str:
    """An exception in one line, worded as the semantics' runner words it: the judge compares an
    answer's message with the reference's word for word."""
    text = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {text[0] if text else ''}"[:300]


def count_launches() -> list[int]:
    """Count every Triton launch from now on (a warm-up compiles and launches nothing)."""
    from triton.runtime.jit import JITFunction

    count = [0]
    original = JITFunction.run

    def run(self: Any, *args: Any, grid: Any, warmup: bool, **kwargs: Any) -> Any:
        if not warmup:
            count[0] += 1
        return original(self, *args, grid=grid, warmup=warmup, **kwargs)

    JITFunction.run = run  # type: ignore[method-assign]
    return count


def main() -> int:
    script, out_json, out_npz = (Path(a) for a in sys.argv[1:4])
    report: dict[str, Any] = {"file": script.name, "run": "ok", "launches": 0}
    launches = count_launches()
    start = time.time()
    torch.manual_seed(0)
    sys.argv = [str(script)]
    results: Any = None
    try:
        scope = runpy.run_path(str(script), run_name="__main__")
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # an asynchronous fault surfaces here, as the file's
        results = scope.get(results_name(script))
    except BaseException as exc:  # noqa: BLE001  the file under test may raise anything
        lines = [ln.strip() for ln in str(exc).strip().splitlines() if ln.strip()]
        report.update(
            run="call_error", message=first_line(exc), detail=" | ".join(lines[-2:])[:300]
        )
    report["launches"] = launches[0]
    if report["run"] == "ok":
        arrays: list[np.ndarray] = []
        report["results"] = plain(results, arrays)
        np.savez(out_npz, *arrays)
    report["seconds"] = round(time.time() - start, 1)
    out_json.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
