"""A test file as it is written for a GPU. `uses: alepot55/ttsem@main` runs it on a CPU runner."""

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def scale_kernel(x_ptr, out_ptr, n, alpha, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask, other=0.0) * alpha, mask=mask)


@pytest.mark.parametrize("n", [0, 1, 1000, 1024, 4097])
def test_scale(n):
    x = torch.rand(n, device="cuda")
    out = torch.empty_like(x)
    if n:
        scale_kernel[(triton.cdiv(n, 256),)](x, out, n, 3.0, BLOCK=256)
    torch.testing.assert_close(out, x * 3.0)
