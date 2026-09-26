"""An answer whose grid's first entry is a tuple (`lambda meta: ((n,),)`): Triton's launcher
parses each entry as an int, so on a GPU the launch raises a TypeError."""

import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
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
        grid = lambda meta: ((triton.cdiv(n, meta["BLOCK"]),),)  # noqa: E731
        relu_kernel[grid](x, out, n, BLOCK=256)
        return out
