"""A correct answer whose autotuner's first config does not compile (its block is over the
kernel's own `static_assert`): Triton's autotuner skips it, so it is never the one a GPU runs."""

import torch
import triton
import triton.language as tl
from torch import nn


@triton.autotune(configs=[triton.Config({"BLOCK": 2048}), triton.Config({"BLOCK": 256})], key=["n"])
@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    tl.static_assert(BLOCK <= 1024)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        relu_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](x, out, n)
        return out
