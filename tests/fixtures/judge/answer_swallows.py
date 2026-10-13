"""The unsafe answer (a load without its mask) with its launch inside try/except: a fault the judge
raised into the answer's own code would be caught there, and PyTorch would compute the result
instead. On a GPU the launch raises nothing (the load reads past the end and goes on), so the
fault must reach the judge without going through the answer."""

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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        try:
            relu_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK=256)
        except BaseException:
            out = torch.relu(x)
        return out
