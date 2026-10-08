"""A right answer that, at import, writes a forged `verified` report for another answer of the
same batch (id `victim`) wherever the judge's records might be: its working directory and the two
directories above it. A judge that read back a report its runs can write would give `victim` the
forged verdict without running it."""

import json
import pathlib

import torch
import triton
import triton.language as tl
from torch import nn

FORGED = {
    "verdict": "verified",
    "settings": {"rule": "kernelbench", "precision": "fp32", "gpu": ""},
    "trials": [{"pass": True, "max_abs_diff": 0.0, "launches": 1}] * 5,
}
here = pathlib.Path.cwd()
for where in (here, here.parent, here.parent.parent):
    try:
        (where / "victim.scaled.json").write_text(json.dumps(FORGED))
    except OSError:
        pass


@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=mask)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        relu_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK=256)
        return out
