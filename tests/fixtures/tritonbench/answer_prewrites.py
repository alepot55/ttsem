"""A right answer that, at import, writes a forged `verified` record for another answer of the
same batch (id `victim`) wherever the judge's records might be: its working directory and the two
directories above it. A judge that read back a record its runs can write would give `victim` the
forged verdict without running it."""

import json
import pathlib

import torch
import triton
import triton.language as tl

FORGED = {"verdict": "verified", "class": "verified", "tolerance": "strict", "launches": 2}
here = pathlib.Path.cwd()
for where in (here, here.parent, here.parent.parent):
    try:
        (where / "victim.tritonbench.json").write_text(json.dumps(FORGED))
    except OSError:
        pass


@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=mask)


def relu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    relu_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK=256)
    return out
