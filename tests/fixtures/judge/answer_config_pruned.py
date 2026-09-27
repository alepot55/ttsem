"""A correct answer whose autotuner prunes, before benchmarking, the configs it cannot use: one
program per row, so a block shorter than the row leaves its tail at zero. `early_config_prune`
drops the blocks shorter than the row (the third config), and `perf_model` with `top_k=1` keeps
the smallest block left (the first), so the autotuner never runs the other two."""

import torch
import triton
import triton.language as tl
from torch import nn


def long_enough(configs, named_args, **kwargs):
    return [c for c in configs if c.kwargs["BLOCK"] >= named_args["dim"]]


def block_size(**meta):
    return meta["BLOCK"]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 1024}),
        triton.Config({"BLOCK": 2048}),
        triton.Config({"BLOCK": 512}),
    ],
    key=["dim"],
    prune_configs_by={"early_config_prune": long_enough, "perf_model": block_size, "top_k": 1},
)
@triton.jit
def row_relu_kernel(x_ptr, out_ptr, dim, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < dim
    x = tl.load(x_ptr + row * dim + offs, mask=mask)
    tl.store(out_ptr + row * dim + offs, tl.maximum(x, 0.0), mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        out = torch.zeros_like(x)
        row_relu_kernel[(x.shape[0],)](x, out, x.shape[1])
        return out
