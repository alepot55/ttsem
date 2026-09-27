"""A task in TritonBench's own format: the reference operator, the separator, the task's test.
The test's second case (33 rows of 7) is not a multiple of any power-of-two block, so an access
without its mask runs past the end."""

import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=mask)


def relu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    relu_kernel[(triton.cdiv(n, 128),)](x, out, n, BLOCK=128)
    return out


##################################################################################################################################################


def test_relu():
    results = {}
    x = torch.randn(1024, device="cuda")
    results["test_case_1"] = relu(x)
    x = torch.randn(33, 7, device="cuda")
    results["test_case_2"] = relu(x)
    return results


result_gold = test_relu()
