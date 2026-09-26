"""An answer whose autotuner has no config that fits a GPU's shared memory: 128 x 128 tiles with
a K step of 64 and four or five pipeline stages need 262,144 B or more on an H100 by Triton
3.8's count, over its 232,448 B. On the GPU every config is skipped, and the launch with the
first one fails."""

import torch
import triton
import triton.language as tl
from torch import nn

CONFIGS = [
    triton.Config({"BM": 128, "BN": 128, "BK": 64}, num_warps=8, num_stages=stages)
    for stages in (4, 5)
]


@triton.autotune(configs=CONFIGS, key=["M", "N", "K"])
@triton.jit
def matmul_kernel(a, b, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        ka = k + rk
        x = tl.load(a + rm[:, None] * K + ka[None, :], mask=(rm[:, None] < M) & (ka[None, :] < K))
        y = tl.load(b + ka[:, None] * N + rn[None, :], mask=(ka[:, None] < K) & (rn[None, :] < N))
        acc += tl.dot(x, y)
    tl.store(c + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        M, K = a.shape
        N = b.shape[1]
        c = torch.empty((M, N), dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))  # noqa: E731
        matmul_kernel[grid](a.contiguous(), b.contiguous(), c, M, N, K)
        return c
