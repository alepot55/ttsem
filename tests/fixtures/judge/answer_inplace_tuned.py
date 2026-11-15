"""A correct answer to `task_inplace.py` whose kernel is autotuned and writes into x's own
storage (`restore_value` puts x back after each benchmark run, as a GPU's autotuner needs). The
judge forces every config in turn; each run must start from the inputs the task made, not from
what the previous config left in x."""

import torch
import triton
import triton.language as tl
from torch import nn


@triton.autotune(
    configs=[triton.Config({"BLOCK": b}) for b in (128, 256, 512)],
    key=["n"],
    restore_value=["x_ptr"],
)
@triton.jit
def add_into_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(x_ptr + offs, x + y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
        add_into_kernel[grid](x, y, n)
        return x
