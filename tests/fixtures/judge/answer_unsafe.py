"""An unsafe answer: the load has no mask, so the last block reads 96 elements past the end of
the input (4000 elements, 16 blocks of 256). The values are right: on a GPU the read usually
lands in the allocator's padding and every output comparison passes."""

import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        relu_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK=256)
        return out
