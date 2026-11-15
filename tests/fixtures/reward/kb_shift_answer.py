"""Positive control (hand-written): the shift done in place, `y[1:] = y[:-1]`, by one kernel whose
block (1,024) covers the task's 1,000 elements, so uncapped one program instance loads every value
before it stores any. With the block capped, instance k loads y[16k - 1], which instance k - 1
stores: a race between instances that the judge must report."""

import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def shift_kernel(y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    keep = (offs >= 1) & (offs < n)
    src = tl.load(y_ptr + offs - 1, mask=keep)
    tl.store(y_ptr + offs, src, mask=keep)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = x.contiguous().clone()
        n = y.numel()
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        shift_kernel[grid](y, n, BLOCK=BLOCK)
        return y
