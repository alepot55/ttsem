"""A Gluon kernel: a blocked layout spelled out, a load and a store. Only compiled in the
tests (Gluon has no interpreter to capture a reference from on the CPU)."""

import torch
import triton.experimental.gluon.language as ttgl
from triton.experimental.gluon import jit as gluon_jit


@gluon_jit
def copy_kernel(x_ptr, y_ptr, N: ttgl.constexpr):
    layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [4], [0])
    offs = ttgl.arange(0, N, layout=layout)
    ttgl.store(y_ptr + offs, ttgl.load(x_ptr + offs) * 2)


def launch_args() -> tuple:
    x = torch.arange(128, dtype=torch.float32)
    y = torch.empty_like(x)
    return copy_kernel, (x, y), {"N": 128, "num_warps": 4}
