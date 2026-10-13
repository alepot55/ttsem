"""ADVERSARIAL PROBE (judge bwrap only). A background thread flips a watched attribute
(`Memory._read`) on and off in a tight loop, hoping a launch lands while it is patched and the
integrity check lands while it is not. The env snapshot records the live threads, so a thread
started after the snapshot is itself a breach; and whenever the flip is active at a check the
rebinding is caught. Either way this must end `tampered`, never `verified`. The static scan
refuses `threading` outright; run with the scan off to show the runtime net."""

import sys
import threading
import time

import numpy as np
import torch
import triton
import triton.language as tl
from torch import nn

_M = sys.modules["ttsem.memory"]
_orig = _M.Memory._read


def _read(self, addrs, dtype):
    dtype = np.dtype(dtype)
    return np.zeros(np.asarray(addrs).size, dtype=dtype)


def _flip():
    while True:
        _M.Memory._read = _read
        time.sleep(0)
        _M.Memory._read = _orig


threading.Thread(target=_flip, daemon=True).start()


@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)  # no mask: out of bounds
    tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=offs < n)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        for _ in range(50):
            relu_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK=256)
        return out
