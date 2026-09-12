"""Fuzzer program 10-7: ascan_add sort where ascan_max trans cast (num_warps=2)."""

import argparse

import numpy as np
import torch
import triton
import triton.language as tl

NUM_WARPS = 2
OUT_ELEMS = 512
OUT_DTYPE = torch.uint8


@triton.jit
def _add(a, b):
    return a + b


@triton.jit
def _max(a, b):
    return tl.maximum(a, b)


@triton.jit
def kernel(out_ptr, x_ptr, D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr):
    t0 = (tl.reshape(tl.arange(0, D0 * D1 * D2 * D3), (D0, D1, D2, D3)) * 11 + 14) % 4096
    t1 = tl.load(x_ptr + t0, mask=(t0 & 1) == 0, other=74)
    t2 = tl.associative_scan(t1, 1, _add, reverse=True)
    t3 = tl.sort(t2, descending=True)
    t4 = tl.expand_dims(tl.expand_dims(tl.expand_dims(tl.arange(0, 16) % 2 == 0, 0), 1), 2)
    t5 = tl.reshape(tl.arange(0, 512), (2, 4, 4, 16))
    t6 = tl.where(t4, t3, t5)
    t7 = tl.associative_scan(t6, 1, _max, reverse=True)
    t8 = tl.permute(t7, (2, 0, 1, 3))
    t9 = t8 % 100
    t10 = t9.to(tl.uint8)
    tl.static_assert(t10.dtype == tl.uint8)
    offs = tl.reshape(tl.arange(0, 512), (4, 2, 4, 16))
    tl.store(out_ptr + offs, t10, mask=offs % 2 == 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    x_np = np.random.RandomState(0).randint(0, 1000, 4096).astype(np.int32)
    x = torch.from_numpy(x_np).to(args.device)
    fill = torch.iinfo(OUT_DTYPE).max
    out = torch.full((OUT_ELEMS,), fill, dtype=OUT_DTYPE, device=args.device)
    kernel[(1,)](out, x, D0=2, D1=4, D2=4, D3=16, num_warps=NUM_WARPS)
    np.save(args.out, out.cpu().numpy())


if __name__ == "__main__":
    main()
