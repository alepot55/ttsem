"""Triton 3.9.0's TTIR, as the wheel prints it, read and run by the semantics without Triton.

Triton 3.9.0 changed how TTIR is spelled: a pointer carries a named address space
(`!tt.ptr<f32, "constant">` for a `tl.const` argument, where 3.8.0 wrote `!tt.ptr<f32>`), and the
cache modifier and the eviction policy of a load or a store are one `cachePolicy` attribute
(`#tt.cache_policy<cache_modifier = ca, eviction_policy = evict_normal>`, where 3.8.0 wrote the
integers `cache` and `evict`). The semantics reads neither, so nothing in the reader changed; this
file is what says so. The fixture is the TTIR of `fixtures/triton39_cache_policy.py` from the
3.9.0 wheel, and the test needs neither Triton nor a GPU, so it runs on every Python the package
supports.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ttsem import mlir
from ttsem.interp import Interp
from ttsem.ir_types import Module, Op
from ttsem.memory import Memory

FIXTURES = Path(__file__).resolve().parent / "fixtures"
N = 100  # the elements the launch works on; two programs of 64 lanes cover 128


def _ops(module: Module) -> Iterator[Op]:
    stack = list(reversed(module.ops))
    while stack:
        op = stack.pop()
        yield op
        for region in reversed(op.regions):
            for block in reversed(region.blocks):
                stack.extend(reversed(block.ops))


def _module() -> Module:
    return mlir.parse((FIXTURES / "triton39_cache_policy.ttir.generic").read_text())


def test_a_named_address_space_is_a_pointer_and_a_cache_policy_is_an_attribute() -> None:
    module = _module()
    params = module.funcs["kernel"].regions[0].blocks[0].args
    pointers = [ty for _, ty in params if ty.kind == "ptr"]
    assert [ty.encoding for ty in pointers] == ['"constant"', None, None]
    assert all(ty.elem is not None and ty.elem.name == "f32" for ty in pointers)
    policies = [
        str(op.attrs["cachePolicy"]) for op in _ops(module) if op.name in ("tt.load", "tt.store")
    ]
    assert policies == [
        "#tt.cache_policy<cache_modifier = ca, eviction_policy = evict_normal>",
        "#tt.cache_policy<cache_modifier = none, eviction_policy = evict_last>",
        "#tt.cache_policy<cache_modifier = wt, eviction_policy = evict_normal>",
    ]


def test_a_module_with_named_pointers_and_cache_policies_computes_the_same_values() -> None:
    """Masked loads through a `"constant"` pointer, a masked store: the lanes past `n` neither
    read nor write, and the others hold `x + y`."""
    x = np.arange(N, dtype=np.float32)
    y = np.ones(N, dtype=np.float32)
    out = np.full(128, -7.0, dtype=np.float32)
    memory = Memory()
    bases = {"x": 0x10000, "y": 0x20000, "out": 0x30000}
    for name, array in (("x", x), ("y", y), ("out", out)):
        memory.register(bases[name], array)
    args = [np.array(bases[name], dtype=np.int64) for name in ("x", "y", "out")]
    args.append(np.array(N, dtype=np.int32))
    Interp(_module(), memory, (2, 1, 1)).run_grid("kernel", args)
    assert np.array_equal(out[:N], x + y)
    assert np.all(out[N:] == -7.0)
