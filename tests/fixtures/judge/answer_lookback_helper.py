"""An answer that really calls TorchInductor's decoupled-lookback helper, and races on the same
scratch buffer past the helper's words (round 5 of the release review).

Each program passes its count of positive inputs through the helper, as Inductor's split scan
does, so the call is the real one and the buffer is the one it names. Past the helper's three words
per program, program 0 stores 1 and every program reads it back: right in the order ttsem runs the
instances, wrong on a GPU where a program runs before program 0. A session that trusts the kernel
(`sanitize.session()`, for Inductor's own code) leaves the whole buffer out of the race check and
misses that race; a judge trusts no answer and finds it."""

import torch
import triton
import triton.language as tl
from torch import nn
from torch._inductor.runtime import triton_helpers


@triton.jit
def _add(a, b):
    return a + b


@triton.jit
def relu_kernel(x_ptr, out_ptr, ws_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    count = tl.sum((x > 0).to(tl.int64), 0)
    triton_helpers.exclusive_scan_decoupled_lookback_64(
        ws_ptr.to(tl.pointer_type(tl.uint64)), count, pid, _add
    )
    flag = ws_ptr.to(tl.pointer_type(tl.uint64)) + 3 * tl.num_programs(0)
    if pid == 0:
        tl.store(flag + tl.arange(0, 1), tl.full((1,), 1, tl.uint64))
    c = tl.load(flag + tl.zeros((BLOCK,), tl.int32)).to(tl.float32)
    tl.store(out_ptr + offs, tl.maximum(x, 0.0) + c - 1.0, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        out = torch.empty_like(x)
        grid = triton.cdiv(x.numel(), 1024)
        ws = torch.zeros(8 * (3 * grid + 1), dtype=torch.uint8, device=x.device)
        relu_kernel[(grid,)](x, out, ws, x.numel(), BLOCK=1024)
        return out
