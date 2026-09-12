"""Fuzzer program 10-3 (num_warps=4, num_stages=2).

desc_dot bfloat16 blocks=16x64x16 shape=80x128x91 grid=5x2 steps=6 loop=ws stages=2 pads=5/8/4
desc=host
"""

import argparse
import os

import numpy as np
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

NUM_WARPS = 4
NUM_STAGES = 2
GRID = (5, 2)
OUT_SHAPE = (80, 132)
IN_DTYPE = torch.bfloat16
FILL = 2147483648.0
SEED = 79193
INTERP = os.environ.get("TRITON_INTERPRET") == "1"


@triton.jit
def kernel(
    out_d,
    a_d,
    b_d,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INTERP: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_m = pid_m * BLOCK_M
    pid_n = tl.program_id(1)
    off_n = pid_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for ki in tl.range(0, tl.cdiv(K, BLOCK_K), warp_specialize=True, num_stages=2):
        k = ki * BLOCK_K
        a = a_d.load([off_m, k])
        if INTERP:
            a = a.to(tl.float32)
        b = b_d.load([k, off_n])
        if INTERP:
            b = b.to(tl.float32)
        acc = tl.dot(a, b, acc)
    out_d.store([off_m, off_n], acc)


def inputs() -> tuple[np.ndarray, ...]:
    rs = np.random.RandomState(SEED)
    a = rs.randint(-2, 2 + 1, (80, 96)).astype(np.float32)
    b = rs.randint(-2, 2 + 1, (96, 136)).astype(np.float32)
    return a, b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    a, b = (torch.from_numpy(x).to(IN_DTYPE).to(args.device) for x in inputs())
    out = torch.full(OUT_SHAPE, FILL, dtype=torch.float32, device=args.device)
    a_d = TensorDescriptor.from_tensor(a[:80, :91], [16, 16])
    b_d = TensorDescriptor.from_tensor(b[:91, :128], [16, 64])
    out_d = TensorDescriptor.from_tensor(out[:80, :128], [16, 64])
    kernel[GRID](
        out_d,
        a_d,
        b_d,
        91,
        BLOCK_M=16,
        BLOCK_N=64,
        BLOCK_K=16,
        INTERP=INTERP,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    np.save(args.out, out.cpu().numpy().reshape(-1))


if __name__ == "__main__":
    main()
