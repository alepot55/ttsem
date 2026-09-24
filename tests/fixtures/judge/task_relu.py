"""A task in KernelBench's own format: ReLU over a batch of vectors whose length (1000) is not a
multiple of any power-of-two block, so an access without its mask runs past the end."""

import torch
from torch import nn


class Model(nn.Module):
    """The reference: applies ReLU to its input."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


batch_size = 4
dim = 1000


def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]


def get_init_inputs():
    return []
