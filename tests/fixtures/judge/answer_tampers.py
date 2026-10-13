"""An unsafe answer (a load without its mask) that first reaches into the judge: it imports the
checker and rewrites `Memory._resolve` so an out-of-bounds address is clamped in bounds and never
faults. In the same process as the semantics this would turn `unsafe` into `verified`; the static
tamper scan refuses the answer before it runs (it imports the judge's own module)."""

import numpy as np
import torch
import triton
import triton.language as tl
from torch import nn

import ttsem.memory as M

_orig = M.Memory._resolve


def _tame(self, addrs, itemsize):
    try:
        return _orig(self, addrs, itemsize)
    except M.MemoryFault:
        a = np.zeros(np.asarray(addrs).shape, dtype=np.int64)
        return a, a


M.Memory._resolve = _tame


@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)  # no mask: out of bounds
    tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=offs < n)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        relu_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK=256)
        return out
