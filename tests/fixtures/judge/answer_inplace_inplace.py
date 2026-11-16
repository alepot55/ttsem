"""A correct answer to `task_inplace.py` that does what the reference does: one Triton kernel adds
y into x's own storage, and the answer returns x."""

import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def add_into_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(x_ptr + offs, x + y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = x.numel()
        add_into_kernel[(triton.cdiv(n, 256),)](x, y, n, BLOCK=256)
        return x
