"""Subprocess entry point for ``harness.py``'s CPU reference pass.

``triton.language``'s own ``@triton.jit``-decorated builtins (``tl.zeros``, ``tl.sort``,
``tl.associative_scan``, ...) are wrapped as plain, uninterpretable ``JITFunction`` the first
time ``triton.language`` is imported into a process, and stay that way regardless of what
``TRITON_INTERPRET`` is set to afterwards. ``harness.py`` itself imports ``triton`` at module
load (needed for its compile-side helpers), so its own process can never get correct
interpreted semantics for a kernel that calls one of those builtins. This script exists only to
get a fresh process where ``TRITON_INTERPRET`` is set *before* triton is ever imported; see
``harness._capture_interpreted`` for the parent-side half of this.

Not meant to be imported: run as ``python -m ttsem._interp_runner PROGRAM.py OUT.pkl``.
"""

from __future__ import annotations

import os

os.environ["TRITON_INTERPRET"] = "1"

import pickle
import sys
from pathlib import Path
from typing import Any

from triton.runtime.interpreter import InterpretedFunction

from ttsem import harness


def _capture(program_path: Path) -> list[dict[str, Any]]:
    launches: list[dict[str, Any]] = []
    orig_run = InterpretedFunction.run

    def run(self: InterpretedFunction, *args: Any, grid: Any, warmup: bool, **kwargs: Any) -> Any:
        if warmup:
            return orig_run(self, *args, grid=grid, warmup=warmup, **kwargs)
        bound = harness._bind_names(self.arg_names, args, kwargs)
        # the same leaves `harness.record_launch` keeps: tensors, tuple members, and the base
        # tensor behind a host descriptor, each as its whole storage (a descriptor over a slice
        # addresses memory outside the slice)
        leaves = [(leaf, v) for name, v in bound.items() for leaf, v in harness._flat_args(name, v)]
        tensors = {leaf: v for leaf, v in leaves if harness._is_tensor_like(v)}
        tensors.update({leaf: v.base for leaf, v in leaves if harness._is_descriptor(v)})
        copies = {name: harness.storage_copy(v) for name, v in tensors.items()}
        pre = {name: arr for name, (_, arr) in copies.items()}
        bases = {name: base for name, (base, _) in copies.items()}
        ptrs = {name: int(v.data_ptr()) for name, v in tensors.items()}
        result = orig_run(self, *args, grid=grid, warmup=warmup, **kwargs)
        post = {name: harness.storage_copy(v)[1] for name, v in tensors.items()}
        launches.append(
            {
                "fn_name": self.fn.__name__,
                "grid": grid if not callable(grid) else ("callable",),
                # Only the non-tensor parameters: tensors are carried as pre/post/ptrs instead,
                # and torch tensors would make this dict unnecessarily heavy to pickle.
                "bound": {k: v for k, v in bound.items() if not harness._is_tensor_like(v)},
                "pre": pre,
                "post": post,
                "elems": {name: str(harness._unwrap(v).dtype) for name, v in tensors.items()},
                "ptrs": ptrs,
                "bases": bases,
            }
        )
        return result

    InterpretedFunction.run = run
    try:
        harness._run_program(program_path, "cpu")
    finally:
        InterpretedFunction.run = orig_run
    return launches


def main() -> int:
    program_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    launches = _capture(program_path)
    with out_path.open("wb") as f:
        pickle.dump(launches, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
