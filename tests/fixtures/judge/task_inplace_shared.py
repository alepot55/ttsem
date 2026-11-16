"""`task_inplace.py` with the tensors kept at module level: `get_inputs` returns the same two
tensors every time. A reference that changes `x` in place then changes it for every trial after
it, in the reference's own process and in the answer's, unless each call gets a copy."""

import torch
from torch import nn

batch_size = 4
dim = 1000

X = torch.randn(batch_size, dim)
Y = torch.randn(batch_size, dim)


class Model(nn.Module):
    """The reference: adds `y` to `x` in place."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x += y
        return x


def get_inputs():
    return [X, Y]


def get_init_inputs():
    return []
