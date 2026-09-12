"""Fuzzer program 10-3: cast binop (num_warps=1)."""

import argparse

import numpy as np
import torch
import triton
import triton.language as tl

NUM_WARPS = 1
OUT_ELEMS = 1024
OUT_DTYPE = torch.uint8


@triton.jit
def _add(a, b):
    return a + b


@triton.jit
def _max(a, b):
    return tl.maximum(a, b)


@triton.jit
def kernel(out_ptr, x_ptr, D0: tl.constexpr):
    t0 = (tl.arange(0, D0) * 9 + 23) % 4096
    t1 = tl.arange(0, 1024) % 100
    t2 = tl.load(x_ptr + t0, mask=(t0 & 1) == 0, other=t1)
    t3 = t2 % 100
    t4 = t3.to(tl.uint8)
    t5 = tl.arange(0, 1024)
    t6 = t5 % 10
    t7 = t6.to(tl.uint8)
    t8 = t4 + t7
    tl.static_assert(t8.dtype == tl.uint8)
    offs = tl.arange(0, 1024)
    tl.store(out_ptr + offs, t8, mask=offs < 512)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    x_np = np.random.RandomState(0).randint(0, 1000, 4096).astype(np.int32)
    x = torch.from_numpy(x_np).to(args.device)
    fill = torch.iinfo(OUT_DTYPE).max
    out = torch.full((OUT_ELEMS,), fill, dtype=OUT_DTYPE, device=args.device)
    kernel[(1,)](out, x, D0=1024, num_warps=NUM_WARPS)
    np.save(args.out, out.cpu().numpy())


if __name__ == "__main__":
    main()
