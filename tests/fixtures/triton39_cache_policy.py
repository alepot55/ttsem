"""A launch whose TTIR Triton 3.9.0 prints in the forms 3.8.0 did not use.

`tl.const` makes `x_ptr` a `!tt.ptr<f32, "constant">` (3.8.0: `!tt.ptr<f32>`), and the cache
modifier and the eviction policy of a load or a store are one `cachePolicy` attribute,
`#tt.cache_policy<cache_modifier = ca, eviction_policy = evict_normal>` (3.8.0: two integers,
`cache` and `evict`). `triton39_cache_policy.ttir.generic` is the TTIR of this launch from the
3.9.0 wheel, no GPU: `python -m ttsem.validate tests/fixtures/triton39_cache_policy.py`.
"""

import argparse

import numpy as np
import torch
import triton
import triton.language as tl

N = 100
BLOCK = 64


@triton.jit
def kernel(x_ptr: tl.const, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(x_ptr + offs, mask=mask, other=-1.0, cache_modifier=".ca")
    b = tl.load(y_ptr + offs, mask=mask, other=-1.0, eviction_policy="evict_last")
    tl.store(out_ptr + offs, a + b, mask=mask, cache_modifier=".wt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    x = torch.arange(N, dtype=torch.float32, device=args.device)
    y = torch.ones(N, dtype=torch.float32, device=args.device)
    out = torch.full((N,), -7.0, dtype=torch.float32, device=args.device)
    kernel[(2,)](x, y, out, N, BLOCK=BLOCK)
    np.save(args.out, out.cpu().numpy())


if __name__ == "__main__":
    main()
