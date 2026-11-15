import torch  # noqa: F401  (kept: the answer format TritonBench answers follow)
import triton
import triton.language as tl


@triton.jit
def shift_kernel(x_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    vals = tl.load(x_ptr + offs, mask=offs < n - 1)
    tl.store(x_ptr + offs + 1, vals, mask=offs < n - 1)


def shift_right(x):
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    shift_kernel[grid](x, n, BLOCK_SIZE=BLOCK_SIZE)
    return x


##################################################################################################################################################
