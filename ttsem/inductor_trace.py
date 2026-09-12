"""Validate every Triton kernel `torch.compile` launches while a script runs.

Inductor does not launch its kernels through `JITFunction.run`: its `CachingAutotuner.run`
picks a compiled launcher and calls it directly, so the pytest plugin's hook never sees
them. This runner routes the autotuner through the `JITFunction` instead (the same path
Inductor itself takes under `TRITON_INTERPRET=1`, with the first config's block sizes and
`num_warps`/`num_stages` passed explicitly), installs the plugin's hook, and runs the script.
Every kernel launch is then recorded, compiled once more for its stage's IR, run on the
semantics, and compared with the device, one JSONL line per launch as under pytest.

    python -m ttsem.inductor_trace --out DIR [--stage ttgir] [--triton-opt PATH] [--dump DIR] \\
        [--below-llvm module:Class] script.py [args...]
"""

from __future__ import annotations

import argparse
import collections
import os
import runpy
import sys
from pathlib import Path
from typing import Any


def route_inductor_through_jit() -> None:
    """Make `CachingAutotuner.run` launch through `self.fn[grid](...)`."""
    from torch._inductor.runtime import triton_heuristics as th

    from ttsem import pytest_ttsem

    def run(
        self: Any, *args: Any, stream: Any = None, benchmark_run: bool = False, **kwargs: Any
    ) -> Any:
        if not self.launchers:  # `configs` is dropped after precompile; the launcher keeps it
            self.precompile()
        cfg = self.launchers[0].config
        new_args, grid = self._interpret_args_grid(args, cfg)
        pytest_ttsem._state["nodeid"] = f"inductor::{self.fn.__name__}"
        return self.fn[grid](
            *new_args,
            **kwargs,
            **cfg.kwargs,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )

    th.CachingAutotuner.run = run  # type: ignore[method-assign]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stage", default="ttgir")
    ap.add_argument("--triton-opt", default=os.environ.get("TTSEM_TRITON_OPT"))
    ap.add_argument("--dump", default=None)
    ap.add_argument("--below-llvm", default=None)
    ap.add_argument("--max-launches", type=int, default=100_000)
    ap.add_argument("script", type=Path)
    ap.add_argument("script_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    from ttsem import pytest_ttsem

    args.out.mkdir(parents=True, exist_ok=True)
    pytest_ttsem._state.update(
        stage=args.stage,
        triton_opt=args.triton_opt,
        max_launches=args.max_launches,
        dump=args.dump,
        below=pytest_ttsem._load_below(args.below_llvm),
        log=open(args.out / "inductor.jsonl", "a"),
        counts=collections.Counter(),
        nodeid="inductor",
        launch_idx=0,
    )
    pytest_ttsem._install_hook()
    route_inductor_through_jit()
    sys.argv = [str(args.script), *args.script_args]
    try:
        runpy.run_path(str(args.script), run_name="__main__")
    finally:
        pytest_ttsem.pytest_sessionfinish(None, 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
