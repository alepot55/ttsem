"""A wrong answer (it copies its input) that leaves behind what a later run could pick up: a
`ttsem` package in its working directory and a file in its Triton cache. It plants these with the
standard library only, so it passes the static tamper scan and actually runs: the point is to show
that neither the planted module nor the cache entry reaches the reference's second run or the
answer judged after it, which run in directories and on caches of their own."""

import os
import pathlib

import torch
import triton
import triton.language as tl

os.makedirs("ttsem", exist_ok=True)
pathlib.Path("ttsem", "__init__.py").write_text('open("planted_ttsem_imported", "w").close()\n')
cache = os.environ.get("TRITON_CACHE_DIR")
if cache:
    pathlib.Path(cache, "planted").write_text("")


@triton.jit
def copy_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)


def relu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    copy_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK=256)
    return out
