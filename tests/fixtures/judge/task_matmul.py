"""A task in KernelBench's own format: a small matrix product."""

import torch
from torch import nn


class Model(nn.Module):
    """The reference: multiplies its two inputs."""

    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a @ b


M = N = K = 64


def get_inputs():
    return [torch.randn(M, K), torch.randn(K, N)]


def get_init_inputs():
    return []
