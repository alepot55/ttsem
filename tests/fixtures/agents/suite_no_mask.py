"""A test file as an agent writes it for a GPU. The kernel forgets the mask on its store."""

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + y)


def add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)
    return out


@pytest.mark.parametrize("n", [1024, 1000])
def test_add(n):
    x = torch.rand(n, device="cuda")
    y = torch.rand(n, device="cuda")
    torch.testing.assert_close(add(x, y), x + y)
