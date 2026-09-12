"""Validate every kernel launch of a pytest run against the level-1 semantics, as it happens.

    TTSEM_OUT=/tmp/ttsem TTSEM_STAGE=ttgir TTSEM_TRITON_OPT=/path/triton-opt \\
    pytest -p ttsem.pytest_ttsem python/test/unit/language/test_core.py --device cuda

For each ``kernel[grid](...)`` that is not a warm-up: copy the storages of the tensor arguments,
let the launch run on the device, copy them again, compile the launch like the oracle does, get
the IR of ``TTSEM_STAGE`` (``ttir`` or ``ttgir``), run it on the CPU from the pre-launch state and
compare with the post-launch state bitwise. One JSON line per launch in ``TTSEM_OUT/<worker>.jsonl``
with the test id, the kernel, the verdict and the first differences; a summary at session end.

Nothing is stored beyond that line, so the whole upstream suite fits in a run. Launches whose IR
cannot be obtained (a compile error, a program that has no unique ``tt.func``) are reported as
``error`` and never fail the test they belong to: the suite's own verdicts stay untouched.
"""

from __future__ import annotations

import collections
import hashlib
import importlib
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

from ttsem import harness

_state: dict[str, Any] = {}


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--ttsem-stage", default=os.environ.get("TTSEM_STAGE", "ttgir"))
    parser.addoption("--ttsem-out", default=os.environ.get("TTSEM_OUT", "ttsem_out"))
    parser.addoption("--ttsem-triton-opt", default=os.environ.get("TTSEM_TRITON_OPT"))
    parser.addoption(
        "--ttsem-below-llvm",
        default=os.environ.get("TTSEM_BELOW_LLVM"),
        help="`module:Class` of a validate.BelowLLVM continuation; every launch is then also "
        "compiled with the dump on and its stages from the first llvm.func are handed to it",
    )
    parser.addoption(
        "--ttsem-dump",
        default=os.environ.get("TTSEM_DUMP"),
        help="write each distinct module of the stage here, named by its content hash, for "
        "races_corpus.py and any other reader of final IR",
    )
    parser.addoption(
        "--ttsem-max-launches",
        type=int,
        default=int(os.environ.get("TTSEM_MAX_LAUNCHES", "64")),
        help="launches validated per test; the rest run untouched (a test that launches "
        "thousands of times would otherwise take hours on the CPU)",
    )


def _target() -> Any:
    import torch

    major, minor = torch.cuda.get_device_capability()
    return harness.GPUTarget("cuda", major * 10 + minor, 32)


def pytest_configure(config: Any) -> None:
    out = Path(config.getoption("--ttsem-out"))
    out.mkdir(parents=True, exist_ok=True)
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    _state.update(
        stage=config.getoption("--ttsem-stage"),
        triton_opt=config.getoption("--ttsem-triton-opt"),
        dump=config.getoption("--ttsem-dump"),
        below=_load_below(config.getoption("--ttsem-below-llvm")),
        max_launches=config.getoption("--ttsem-max-launches"),
        log=open(out / f"{worker}.jsonl", "a"),
        counts=collections.Counter(),
        nodeid="?",
        launch_idx=0,
    )
    if not _state["triton_opt"]:
        # No `triton-opt` (a wheel): the dumps must come out generic, and the printer switch
        # only takes before the first compile of the process, so it happens here and not in
        # `dump_for_launch`, which runs after the test's own compile.
        try:
            harness.mlir.enable_generic_printing()
        except RuntimeError as e:
            warnings.warn(f"ttsem: cannot switch the printer to generic form: {e}", stacklevel=1)
    _install_hook()


def pytest_runtest_setup(item: Any) -> None:
    _state["nodeid"] = item.nodeid
    _state["launch_idx"] = 0


def _install_hook() -> None:
    from triton.runtime.jit import JITFunction

    orig_run = JITFunction.run

    def run(self: Any, *args: Any, grid: Any = None, warmup: bool = False, **kwargs: Any) -> Any:
        if warmup or _state.get("log") is None:
            return orig_run(self, *args, grid=grid, warmup=warmup, **kwargs)
        if _state["launch_idx"] >= _state["max_launches"]:
            _state["launch_idx"] += 1
            _state["counts"]["skipped"] += 1
            return orig_run(self, *args, grid=grid, warmup=warmup, **kwargs)
        record = harness.record_launch(self, args, kwargs, grid, orig_run, _state["nodeid"])
        result = record.result
        _validate(record.record)
        return result

    JITFunction.run = run


def _validate(record: Any) -> None:
    t0 = time.time()
    line: dict[str, Any] = {
        "nodeid": _state["nodeid"],
        "launch": _state["launch_idx"],
        "fn": record.fn_name,
        "stage": _state["stage"],
    }
    _state["launch_idx"] += 1
    try:
        text = harness.ir_for_launch(
            record, _state["stage"], _target(), triton_opt=_state["triton_opt"]
        )
        dump_module(_state.get("dump"), text, _state["stage"])
        cmp = harness.run_launch(record, text)
        line.update(
            verdict=cmp.verdict,
            n_diff=cmp.n_diff,
            unsupported=cmp.unsupported,
            message=cmp.message[:300],
            diffs=[d for d in cmp.diffs][:2],
        )
    except Exception as e:  # never let the checker break the suite
        line.update(verdict="error", message=harness._describe(e)[:400])
    if _state.get("below") is not None:
        try:
            line.update(below_llvm_line(record, _state["below"]))
        except Exception as e:
            line.update(below_verdict="error", below_message=harness._describe(e)[:300])
    line["seconds"] = round(time.time() - t0, 3)
    _state["counts"][line["verdict"]] += 1
    _state["log"].write(json.dumps(line, default=str) + "\n")
    _state["log"].flush()


def _load_below(spec: str | None) -> Any:
    if not spec:
        return None
    module_name, _, attr = spec.partition(":")
    obj = getattr(importlib.import_module(module_name), attr or "BelowLlvmAdapter")
    return obj() if isinstance(obj, type) else obj


def below_llvm_line(record: Any, below: Any) -> dict[str, Any]:
    """Compile the launch with the dump on, hand the stages from the first `llvm.func` to the
    continuation, and summarise its verdicts: the field set the JSONL line carries."""
    from ttsem import minimize, validate

    stages = validate.split_dump(harness.dump_for_launch(record, _target()))
    # which passes changed the module on this launch: the pass-stress column, tile and LLVM
    boundaries = [(n, text) for n, text in stages if "(step " not in n]
    bare = [minimize.strip_locs(text) for _, text in boundaries]
    fired = [boundaries[i][0][:80] for i in range(len(boundaries) - 1) if bare[i] != bare[i + 1]]
    start = next((i for i, (_, text) in enumerate(stages) if "llvm.func" in text), None)
    if start is None:
        return {"below_verdict": "no-llvm-stage", "below": [], "fired": fired}
    # the float policy the tile levels used, for the continuation: which inexact classes the
    # program contains (reassociated reductions, approximate divisions) is a fact of the TTGIR,
    # invisible once everything is llvm.fadd; the adapter reads `record.inexact`
    inexact_note = ""
    try:
        tile = harness.mlir.parse(
            harness.mlir.to_generic(stages[start - 1][1], _state.get("triton_opt"))
        )
        record.inexact = harness.scan_inexact(tile)
        inexact_note = ",".join(sorted(record.inexact.classes))
    except Exception as e:
        record.inexact = None
        inexact_note = f"none: {type(e).__name__}"
    handed = []
    for name, text in stages[start:]:
        try:  # generic already when the wheel's printer could be switched, else via triton-opt
            handed.append((name, harness.mlir.to_generic(text, _state.get("triton_opt"))))
        except Exception as e:
            return {"below_verdict": "error", "below": [], "below_message": f"{name}: {e!r}"[:300]}
    results = below(record, handed)
    rows = [
        {
            "pass": r.pass_name[:80],
            "verdict": r.verdict,
            "n_diff": r.n_diff,
            "unsupported": list(r.unsupported)[:8],
            "message": (r.message or "")[:160],
        }
        for r in results
    ]
    verdicts = [r.verdict for r in results]
    return {
        "below_verdict": harness._aggregate(verdicts) if verdicts else "empty",
        "below": rows,
        "fired": fired,
        "inexact": inexact_note,
    }


def dump_module(directory: str | None, text: str, stage: str) -> Path | None:
    """Write ``text`` under ``directory`` as ``<sha1>.<stage>.generic``; identical modules
    (the same kernel launched many times) collapse into one file."""
    if not directory:
        return None
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{hashlib.sha1(text.encode()).hexdigest()[:16]}.{stage}.generic"
    if not path.exists():
        path.write_text(text)
    return path


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    log = _state.get("log")
    if log is None:
        return
    log.close()
    _state["log"] = None
    sys.stderr.write(f"\nttsem launches: {dict(_state['counts'])}\n")
