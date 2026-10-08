"""A correct kernel launched over so many elements that the race check runs out of its access
budget (over 4,000,000 accesses): the launch is run and its values checked, but no race between
program instances is looked for in it. The test passes; the point is that the session reports the
unchecked launch rather than dropping it silently."""

import torch
import triton
import triton.language as tl


@triton.jit
def add_one_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1.0, mask=mask)


def test_add_one_over_many_elements() -> None:
    n = 2_200_000  # a load and a store each log n addresses: over the 4,000,000 budget
    x = torch.randn(n, device="cuda").contiguous()
    out = torch.empty_like(x)
    add_one_kernel[(triton.cdiv(n, 1024),)](x, out, n, BLOCK=1024)
    assert torch.allclose(out, x + 1.0)
