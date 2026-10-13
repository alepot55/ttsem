"""A correct kernel launched over so many elements that the race check's access log runs out
(over 4,000,000 accesses). Past the log, only the range of addresses of each access is kept, and
each program instance loads and stores its own block, so no two instances' ranges meet and the
ranges alone rule a race out: the launch counts as checked (`suite_big_tiled.py` is one they
cannot settle)."""

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
