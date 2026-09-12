"""Subprocess entry point for ``validate.py``'s ``MLIR_ENABLE_DUMP`` capture without a GPU.

Installs the fake CUDA driver (:func:`ttsem.harness._install_fake_driver`) and
replaces ``JITFunction.run`` so every launch is compiled for that fake target -- which is what
actually produces the ``MLIR_ENABLE_DUMP`` pass-by-pass text on stderr, the whole point of
running this -- and never executed: there is no GPU here to launch a real kernel on.

Not meant to be imported: run as ``python -m ttsem._compile_runner PROGRAM.py --cc CC``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from triton.backends.compiler import GPUTarget
from triton.runtime.jit import JITFunction

from ttsem import harness


def _install(target: GPUTarget) -> None:
    def run(self: JITFunction, *args: Any, grid: Any, warmup: bool, **kwargs: Any) -> Any:
        if not warmup:
            try:
                harness._compile_asm(self, args, kwargs, target)
            except Exception as e:  # the dump on stderr is what matters, not this return value
                print(f"[_compile_runner] compile failed: {e!r}", file=sys.stderr)
        return None

    JITFunction.run = run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("program", type=Path)
    ap.add_argument("--cc", type=int, default=harness.DEFAULT_CC)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument(
        "--generic",
        action="store_true",
        help="switch the wheel's printer to the generic form, so the dump needs no triton-opt",
    )
    args = ap.parse_args()

    if args.generic:
        from ttsem import mlir

        try:
            mlir.enable_generic_printing()
        except RuntimeError as exc:  # a source build hides the symbol: the dump comes pretty
            print(f"ttsem: {exc}", file=sys.stderr)
    if args.device == "cuda":
        harness._run_program(args.program, "cuda")
        return 0
    harness._install_fake_driver(args.cc)
    _install(harness.target_for("cpu", args.cc))
    harness._run_program(args.program, "cpu")
    return 0


if __name__ == "__main__":
    sys.exit(main())
