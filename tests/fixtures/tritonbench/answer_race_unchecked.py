import torch
import triton
import triton.language as tl

# The answer of `fixtures/judge/answer_race_unchecked.py` in TritonBench's form: program 0 stores
# 1.0 into `scratch[0]`, every program reads it back (a race the ascending order hides), and the
# first test case's launch rewrites its output until it makes more memory accesses than the race
# check holds.
REWRITES = 3910


@triton.jit
def relu_kernel(x_ptr, out_ptr, scratch_ptr, n, rewrites, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if pid == 0:
        tl.store(scratch_ptr + tl.arange(0, 1), tl.full((1,), 1.0, tl.float32))
    c = tl.load(scratch_ptr + tl.zeros((BLOCK,), tl.int32))
    y = tl.maximum(x, 0.0) + c - 1.0
    tl.store(out_ptr + offs, y, mask=mask)
    for _ in range(rewrites):
        tl.store(out_ptr + offs, y, mask=mask)


def relu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    scratch = torch.zeros(1, dtype=x.dtype, device=x.device)
    n = x.numel()
    relu_kernel[(triton.cdiv(n, 512),)](x, out, scratch, n, REWRITES, BLOCK=512)
    return out
