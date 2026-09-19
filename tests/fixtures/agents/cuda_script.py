"""A plain script as it is written for a GPU: `device="cuda"`, `.cuda()`, no `main()`."""

import torch
import triton
import triton.language as tl


@triton.jit
def scale_kernel(x_ptr, out_ptr, n, alpha, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask, other=0.0) * alpha, mask=mask)


n = 1000
x = torch.arange(n, dtype=torch.float32).cuda()
out = torch.empty(n, dtype=torch.float32, device="cuda")
scale_kernel[(triton.cdiv(n, 128),)](x, out, n, 2.0, BLOCK=128)
torch.cuda.synchronize()
expected = torch.arange(n, dtype=torch.float32, device=torch.device("cuda")) * 2
assert torch.equal(out, expected), "wrong result"
print("SCRIPT_OK", out.to("cuda").sum().item())
