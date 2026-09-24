"""A wrong answer: a leaky ReLU where the task asks for ReLU (negative inputs keep a tenth)."""

import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.where(x > 0, x, 0.1 * x), mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        relu_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK=256)
        return out
