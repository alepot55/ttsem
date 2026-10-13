"""A correct copy in 64 x 64 tiles of a 2048 x 1024 matrix: a load and a store of 2,097,152
elements each, more than the race check's access log holds (4,000,000). Past the log, only the
range of addresses of each access is kept, and the range of a tile's store spans 64 rows, so it
meets the ranges of the tiles beside it in the same rows: the ranges cannot rule a race out. The
test passes; the point is that the session reports the launch it could not check."""

import torch
import triton
import triton.language as tl


@triton.jit
def copy_kernel(x_ptr, out_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)[:, None]
    cols = tl.program_id(1) * BN + tl.arange(0, BN)[None, :]
    mask = (rows < M) & (cols < N)
    x = tl.load(x_ptr + rows * N + cols, mask=mask)
    tl.store(out_ptr + rows * N + cols, x, mask=mask)


def test_copy_in_tiles() -> None:
    m, n = 2048, 1024
    x = torch.randn(m, n, device="cuda")
    out = torch.empty_like(x)
    copy_kernel[(triton.cdiv(m, 64), triton.cdiv(n, 64))](x, out, m, n, BM=64, BN=64)
    assert torch.equal(out, x)
