"""A task whose reference changes its first input in place (`x += y`) and returns it, as 10 of the
GFT tasks do (`x += y`, `x *= y`, `x.mul_(...)`). An answer that is judged on the same tensors the
reference already changed applies the operation twice."""

import torch
from torch import nn


class Model(nn.Module):
    """The reference: adds `y` to `x` in place."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x += y
        return x


batch_size = 4
dim = 1000


def get_inputs():
    return [torch.randn(batch_size, dim), torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
