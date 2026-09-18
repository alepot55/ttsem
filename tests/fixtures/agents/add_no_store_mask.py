"""Vector add as an agent plausibly writes it: the mask is on the loads and not on the store."""

import argparse

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="out.npy")
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    x = torch.arange(a.n, dtype=torch.float32, device=a.device)
    y = torch.ones(a.n, dtype=torch.float32, device=a.device)
    out = torch.empty(a.n, dtype=torch.float32, device=a.device)
    add_kernel[(triton.cdiv(a.n, 256),)](x, y, out, a.n, BLOCK=256)
    np.save(a.out, out.cpu().numpy())


if __name__ == "__main__":
    main()
