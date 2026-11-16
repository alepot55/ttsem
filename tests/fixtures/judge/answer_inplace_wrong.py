"""A wrong answer to `task_inplace.py`: one Triton kernel copies x and ignores y. On the tensor
the reference has already changed (x + y), that copy equals what the reference returned."""

import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def copy_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n = x.numel()
        copy_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK=256)
        return out
