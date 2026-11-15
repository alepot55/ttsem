"""Positive control of the reward's cap, KernelBench-style: a task, a shift right by one, sized so
the judge does not scale it (1,000 elements)."""

import torch
import torch.nn as nn


class Model(nn.Module):
    """out[0] = x[0], out[i] = x[i - 1]."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cat([x[:1], x[:-1]])


n = 1000


def get_inputs():
    return [torch.randn(n)]


def get_init_inputs():
    return []
