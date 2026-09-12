"""Does an e8m0 scale byte of 0xFF (NaN) give NaN on the device and in the interpreter?"""

import os
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def k(a_ptr, b_ptr, sa_ptr, sb_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
    sa = tl.load(sa_ptr + offs_m[:, None] * (K // 32) + tl.arange(0, K // 32)[None, :])
    sb = tl.load(sb_ptr + offs_n[:, None] * (K // 32) + tl.arange(0, K // 32)[None, :])
    c = tl.dot_scaled(a, sa, "bf16", b, sb, "bf16")
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], c)


def main() -> None:
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    M = N = 16
    K = 32
    a = torch.ones((M, K), dtype=torch.bfloat16, device=dev)
    b = torch.ones((K, N), dtype=torch.bfloat16, device=dev)
    sa = torch.full((M, K // 32), 127, dtype=torch.uint8, device=dev)  # scale 1.0
    sb = torch.full((N, K // 32), 127, dtype=torch.uint8, device=dev)
    sa[0, 0] = 0xFF  # NaN scale on row 0
    out = torch.zeros((M, N), dtype=torch.float32, device=dev)
    k[(1,)](a, b, sa, sb, out, M=M, N=N, K=K, num_warps=4)
    mode = os.environ.get("TRITON_INTERPRET", "0")
    print(f"{dev} (TRITON_INTERPRET={mode}): row0[:3]", out[0, :3].tolist())
    print(f"{dev} (TRITON_INTERPRET={mode}): row1[:3]", out[1, :3].tolist())


if __name__ == "__main__":
    main()
