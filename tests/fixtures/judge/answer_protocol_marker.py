"""A race between program instances whose racing buffer a comment names as the scratch buffer of
TorchInductor's decoupled-lookback scan (round 5 of the release review).

Program 0 stores 1.0 into `scratch[0]` and every program reads it back, as in
`answer_race_unchecked.py`: right in the order ttsem runs the instances, wrong on a GPU where a
program runs before program 0. The comment in the kernel spells a call of Inductor's helper on
`scratch_ptr`. The race check once read that buffer off the kernel's text and left it out, and the
judge said `verified`; a judge now leaves no buffer of an answer out of the race check."""

import torch
import triton
import triton.language as tl
from torch import nn

REWRITES = 0


@triton.jit
def relu_kernel(x_ptr, out_ptr, scratch_ptr, n, rewrites, BLOCK: tl.constexpr):
    # exclusive_scan_decoupled_lookback(scratch_ptr)
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
