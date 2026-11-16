"""A correct answer to `task_inplace.py` that leaves its inputs alone: one Triton kernel writes
x + y into a new tensor."""

import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n = x.numel()
        add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)
        return out
