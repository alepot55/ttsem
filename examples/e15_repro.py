"""triton#11519: FuseNestedLoops trusts an llvm.assume on the branch the loop nest cannot reach.

m=4, n=0, flatten=True: the inner loop never runs, so out must stay 0 and sink untouched.
"""

import argparse

import numpy as np
import torch
import triton
import triton.language as tl

BLOCK = 64
SENTINEL = -7777.0


@triton.jit
def kernel(p, out, sink, m, n, BLOCK: tl.constexpr, FLAT: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    if m == 0:
        tl.assume(n > 0)
        acc += tl.load(p + offs)
    else:
        for i in tl.range(0, m, flatten=FLAT):
            for j in range(0, n):
                acc += tl.load(p + j * BLOCK + offs)
                tl.store(sink + (i * 8 + j) * BLOCK + offs, acc)
    tl.store(out + offs, acc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    m, n = 4, 0
    sink_len = max(m, 1) * 8 * BLOCK
    p_host = (np.arange(BLOCK * max(n, 1), dtype=np.float32) % 7.0 + 1.0).astype(np.float32)
    p = torch.from_numpy(p_host).to(args.device)
    out = torch.full((BLOCK,), SENTINEL, dtype=torch.float32, device=args.device)
    sink = torch.full((sink_len,), SENTINEL, dtype=torch.float32, device=args.device)
    kernel[(1,)](p, out, sink, m, n, BLOCK=BLOCK, FLAT=1, num_warps=4)
    np.save(args.out, np.concatenate([out.cpu().numpy(), sink.cpu().numpy()]))


if __name__ == "__main__":
    main()
