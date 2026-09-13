"""A warp-specialized descriptor matmul: `tl.range(..., warp_specialize=True, num_stages=3)`.

Compiled for sm_100 this goes through `nvws-insert-aref`, so the stages between it and
`nvws-lower-aref` carry `nvws.aref.*` and `nvws.descriptor_load` in one loop body.
"""

import argparse

import numpy as np
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

NUM_WARPS = 4
NUM_STAGES = 3
GRID = (2, 2)
OUT_SHAPE = (64, 64)


@triton.jit
def kernel(out_d, a_d, b_d, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    off_m = tl.program_id(0) * BLOCK_M
    off_n = tl.program_id(1) * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for ki in tl.range(0, tl.cdiv(K, BLOCK_K), warp_specialize=True, num_stages=3):
        k = ki * BLOCK_K
        a = a_d.load([off_m, k])
        b = b_d.load([k, off_n])
        acc = tl.dot(a, b, acc, input_precision="ieee")
    out_d.store([off_m, off_n], acc)


def inputs() -> tuple[np.ndarray, np.ndarray]:
    rs = np.random.RandomState(3)
    a = rs.randint(-3, 4, (64, 80)).astype(np.float32)
    b = rs.randint(-3, 4, (80, 64)).astype(np.float32)
    return a, b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    a, b = (torch.from_numpy(x).to(args.device) for x in inputs())
    out = torch.zeros(OUT_SHAPE, dtype=torch.float32, device=args.device)
    a_d = TensorDescriptor.from_tensor(a, [32, 16])
    b_d = TensorDescriptor.from_tensor(b, [16, 32])
    out_d = TensorDescriptor.from_tensor(out, [32, 32])
    kernel[GRID](
        out_d,
        a_d,
        b_d,
        80,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=16,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    np.save(args.out, out.cpu().numpy().reshape(-1))


if __name__ == "__main__":
    main()
