"""triton#11601: FuseNestedLoops hoists the scalar load that bounds the inner loop above an
outer loop that may not run. m=0, flatten=True: the source never reads q; the compiled program
reads it before the loop. The outputs are identical, only the read-set moves, so the minimiser
needs `--reads`. A variant of the original reproducer with q a real buffer.
"""

import argparse

import numpy as np
import torch
import triton
import triton.language as tl

BLOCK = 128


@triton.jit
def kload(p, out, q, m, BLOCK: tl.constexpr, FLAT: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for _i in tl.range(0, m, flatten=FLAT):
        bound = tl.load(q)  # q need only be dereferenceable when m > 0
        for j in range(0, bound):
            acc += tl.load(p + j * BLOCK + offs)
    tl.store(out + offs, acc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    m = 0
    p = torch.arange(4 * BLOCK, dtype=torch.float32, device=args.device)
    q = torch.full((1,), 2, dtype=torch.int32, device=args.device)
    out = torch.full((BLOCK,), -1.0, dtype=torch.float32, device=args.device)
    kload[(1,)](p, out, q, m, BLOCK=BLOCK, FLAT=1, num_warps=4)
    np.save(args.out, out.cpu().numpy())


if __name__ == "__main__":
    main()
