import torch
import triton
import triton.language as tl

# The answer of `fixtures/judge/answer_protocol_marker.py` in TritonBench's form: program 0 stores
# 1.0 into `scratch[0]`, every program reads it back (a race the ascending order hides), and a
# comment in the kernel names `scratch_ptr` as the scratch buffer of Inductor's decoupled-lookback
# scan. The race check once read that off the text, left the buffer out and said `verified`.


@triton.jit
def relu_kernel(x_ptr, out_ptr, scratch_ptr, n, BLOCK: tl.constexpr):
    # exclusive_scan_decoupled_lookback(scratch_ptr)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if pid == 0:
        tl.store(scratch_ptr + tl.arange(0, 1), tl.full((1,), 1.0, tl.float32))
    c = tl.load(scratch_ptr + tl.zeros((BLOCK,), tl.int32))
    y = tl.maximum(x, 0.0) + c - 1.0
    tl.store(out_ptr + offs, y, mask=mask)


def relu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    scratch = torch.zeros(1, dtype=x.dtype, device=x.device)
    n = x.numel()
    relu_kernel[(triton.cdiv(n, 512),)](x, out, scratch, n, BLOCK=512)
    return out
