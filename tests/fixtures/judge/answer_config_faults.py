"""An answer right under its autotuner's first config and out of bounds under the second, where
the autotuner's own benchmarking cannot see it: the grid is sized for a block of 256 whatever the
config, so with a block of 128 the index kernel writes only half of `idx`, and the gather reads
the other half, memory nobody wrote, as offsets. Benchmarking runs every config on the same
`idx`, which the first config has already written whole; only the sweep, which runs the forward
under the second config, reads past the end of `x`."""

import torch
import triton
import triton.language as tl
from torch import nn


@triton.autotune(configs=[triton.Config({"BLOCK": 256}), triton.Config({"BLOCK": 128})], key=["n"])
@triton.jit
def index_kernel(idx_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(idx_ptr + offs, offs, mask=offs < n)


@triton.jit
def gather_relu_kernel(x_ptr, idx_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = tl.load(idx_ptr + offs, mask=mask, other=0)
    x = tl.load(x_ptr + idx, mask=mask)
    tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        n = x.numel()
        idx = torch.empty(n, dtype=torch.int32)
        out = torch.empty_like(x)
        grid = (triton.cdiv(n, 256),)  # the first config's block, whatever the autotuner picks
        index_kernel[grid](idx, n)
        gather_relu_kernel[grid](x, idx, out, n, BLOCK=256)
        return out
