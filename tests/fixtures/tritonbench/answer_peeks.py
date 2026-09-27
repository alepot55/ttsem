"""A right answer that looks for the reference's results, and brings its own test after the
separator: the judge hides the one and ignores the other."""

import glob

import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=mask)


def relu(x: torch.Tensor) -> torch.Tensor:
    if glob.glob("reference/*"):
        raise RuntimeError("the reference's results are in sight")
    out = torch.empty_like(x)
    n = x.numel()
    relu_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK=256)
    return out


##################################################################################################################################################
result_gold = "a test of the answer's own"
