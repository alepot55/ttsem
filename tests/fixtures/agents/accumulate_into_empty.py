"""A kernel that accumulates into its output, and a caller that allocates the output with
`torch.empty`. Fresh pages are zero more often than not, on a GPU too, so the sum comes out right
until the allocator hands back memory that was used before."""

import torch
import triton
import triton.language as tl


@triton.jit
def accumulate(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    acc = tl.load(out_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc + tl.load(x_ptr + offs, mask=mask, other=0.0), mask=mask)


n = 1000
x = torch.ones(n, device="cuda")
out = torch.empty(n, device="cuda")  # should have been torch.zeros
accumulate[(triton.cdiv(n, 256),)](x, out, n, BLOCK=256)
print("ALLNAN", bool(torch.isnan(out).all()))
assert torch.equal(out, x), "the output depends on what the memory held before"
