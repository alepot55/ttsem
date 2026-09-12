"""triton#11407: the pipeliner moves the address arithmetic of a later iteration into the
current one and does not predicate it, so `(i * BLOCK) // (n - i)` divides by zero on a
trip the source never takes. The masked load never dereferences that address, so the device
output is right; the pipelined IR nevertheless executes a division the source program does
not. n=6, three stages."""

import argparse

import numpy as np
import torch
import triton
import triton.language as tl

BLOCK = 64


@triton.jit
def kernel(p, out, n, BLOCK: tl.constexpr, NS: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in tl.range(0, n, num_stages=NS):
        o = (i * BLOCK) // (n - i)
        acc += tl.load(p + o + offs)
    tl.store(out + offs, acc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = 6
    p_host = (np.arange(1024, dtype=np.float32) % 7.0 + 1.0).astype(np.float32)
    p = torch.from_numpy(p_host).to(args.device)
    out = torch.zeros((BLOCK,), dtype=torch.float32, device=args.device)
    kernel[(1,)](p, out, n, BLOCK=BLOCK, NS=3, num_warps=4)
    np.save(args.out, out.cpu().numpy())


if __name__ == "__main__":
    main()
