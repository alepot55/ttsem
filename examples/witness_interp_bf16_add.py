"""The shipped interpreter adds bf16 as bit patterns: `g + g` for g = bf16(-3..4) gives
0x8080, 0x8000, 0x7f00, 0, 0x7f00, 0x8000, 0x8080, 0x8100 (the 16-bit sums of the encodings)
where the device gives 0xc0c0, 0xc080, 0xc000, 0, 0x4000, 0x4080, 0x40c0, 0x4100. Run with
TRITON_INTERPRET=1 --device cpu and without it on a GPU; the harness's cpu path inherits the
interpreter's answer as its reference, so a bf16 mismatch on the cpu path says nothing about
the compiler (5 Sep 2026, Triton 3.7.1 and 3.8.0; the same family as #11584, closed as not
supported).
"""

import argparse

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def k(x_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    g = x.to(tl.bfloat16)
    tl.store(out_ptr + offs, g + g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    x = torch.arange(-3, 5, dtype=torch.float32, device=a.device)
    out = torch.zeros(8, dtype=torch.bfloat16, device=a.device)
    k[(1,)](x, out, N=8)
    np.save(a.out, out.view(torch.int16).cpu().numpy())


if __name__ == "__main__":
    main()
