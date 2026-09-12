"""Fuzzer program 10-5 (num_warps=4, num_stages=1).

rowsum body=rowsum range_step dt=float32 steps=3+tail mask=k_minus/k_minus grid=3x2x1 stages=1
elt=max
"""

import argparse

import numpy as np
import torch
import triton
import triton.language as tl

NUM_WARPS = 4
NUM_STAGES = 1
GRID = (3, 2, 1)
OUT_ELEMS = 98304
IN_DTYPE = torch.float32
FILL = 2147483648.0
SEED = 79195


@triton.jit
def kernel(
    out_ptr,
    a_ptr,
    b_ptr,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        m_a = offs_k[None, :] < K - k
        a = tl.load(a_ptrs, mask=m_a, other=0)
        a = tl.maximum(a, 1).to(a.dtype)
        m_b = offs_k[:, None] < K - k
        b = tl.load(b_ptrs, mask=m_b, other=0)
        rs = tl.sum(a.to(tl.float32), axis=1)
        cs = tl.sum(b.to(tl.float32), axis=0)
        acc += rs[:, None] * cs[None, :]
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    offs = offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(out_ptr + offs, acc)


def inputs() -> tuple[np.ndarray, np.ndarray]:
    rs = np.random.RandomState(SEED)
    a = rs.randint(-4, 4 + 1, (384, 192)).astype(np.float32)
    b = rs.randint(-4, 4 + 1, (192, 256)).astype(np.float32)
    return a, b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    a, b = (torch.from_numpy(x).to(IN_DTYPE).to(args.device) for x in inputs())
    out = torch.full((OUT_ELEMS,), FILL, dtype=torch.float32, device=args.device)
    kernel[GRID](
        out,
        a,
        b,
        176,
        192,
        1,
        256,
        1,
        256,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=64,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    np.save(args.out, out.cpu().numpy())


if __name__ == "__main__":
    main()
