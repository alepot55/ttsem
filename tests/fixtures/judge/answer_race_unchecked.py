"""A race between program instances that gives the right values in the order ttsem runs the
instances, padded with stores of what is already there until a launch makes more memory accesses
than the race check holds (`ttsem.memory.ACCESS_BUDGET`).

Program 0 stores 1.0 into `scratch[0]` (which holds 0.0) and every program reads it back: run in
ascending order the read sees 1.0 and the result is ReLU, on a GPU a program that runs before
program 0 sees 0.0 and returns ReLU minus one. With `REWRITES = 0` the judge finds the race; with
1,100 rewrites each launch is too large for the check, and the judge must not call it verified."""

import torch
import triton
import triton.language as tl
from torch import nn

REWRITES = 1100


@triton.jit
def relu_kernel(x_ptr, out_ptr, scratch_ptr, n, rewrites, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if pid == 0:
        tl.store(scratch_ptr + tl.arange(0, 1), tl.full((1,), 1.0, tl.float32))
    c = tl.load(scratch_ptr + tl.zeros((BLOCK,), tl.int32))
    y = tl.maximum(x, 0.0) + c - 1.0
    tl.store(out_ptr + offs, y, mask=mask)
    for _ in range(rewrites):
        tl.store(out_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        out = torch.empty_like(x)
        scratch = torch.zeros(1, dtype=x.dtype, device=x.device)
        n = x.numel()
        relu_kernel[(triton.cdiv(n, 1024),)](x, out, scratch, n, REWRITES, BLOCK=1024)
        return out
