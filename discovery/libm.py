"""The errors of numpy's float32 ``exp`` and ``log`` on this machine, over every float32 input.

The rounding model (``lean/Discovery/Approx.lean``) takes the relative error of ``exp`` (``ue``),
its absolute error below the normal range (``ηe``) and the relative error of ``log`` (``ul``) as
parameters. A bound proved over the model holds for numpy's runs only if these parameters cover
numpy's functions, so they are measured on every float32 input of the range that matters, against
float64 (whose own error, about 1e-16, is 1e-9 of float32's u).

    python -m discovery.libm --out discovery/results/m2/libm.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

U = 2.0**-24
TINY = np.float32(2.0**-126)  # smallest normal float32


def _floats(lo_bits: int, hi_bits: int, chunk: int):
    """Every float32 whose bit pattern is in [lo_bits, hi_bits), in chunks."""
    for start in range(lo_bits, hi_bits, chunk):
        stop = min(start + chunk, hi_bits)
        yield np.arange(start, stop, dtype=np.uint32).view(np.float32)


def scan_exp(chunk: int = 1 << 24) -> dict:
    """exp on every float32 in [-104, 89] whose result is finite: the relative error of normal
    results, the absolute error of subnormal ones."""
    worst_rel, worst_abs, n, overflow = 0.0, 0.0, 0, 0
    arg_rel = arg_abs = 0.0
    hi_pos = int(np.float32(89.0).view(np.uint32)) + 1
    hi_neg = int(np.float32(-104.0).view(np.uint32)) + 1  # bit patterns of negatives grow with |x|
    for lo, hi in ((0, hi_pos), (0x80000000, hi_neg)):
        for x in _floats(lo, hi, chunk):
            with np.errstate(all="ignore"):
                y = np.exp(x).astype(np.float64)
                r = np.exp(x.astype(np.float64))
            finite = np.isfinite(y)  # overflow is outside the model: counted, not measured
            overflow += int((~finite).sum())
            x, y, r = x[finite], y[finite], r[finite]
            n += x.size
            normal = r >= float(TINY)
            if normal.any():
                rel = np.abs(y[normal] - r[normal]) / r[normal]
                i = int(np.argmax(rel))
                if rel[i] > worst_rel:
                    worst_rel, arg_rel = float(rel[i]), float(x[normal][i])
            if (~normal).any():
                ab = np.abs(y[~normal] - r[~normal])
                i = int(np.argmax(ab))
                if ab[i] > worst_abs:
                    worst_abs, arg_abs = float(ab[i]), float(x[~normal][i])
    return {
        "inputs": n,
        "overflowed_not_measured": overflow,
        "range": [-104.0, 89.0],
        "max_rel_error": worst_rel,
        "max_rel_error_in_u": worst_rel / U,
        "at": arg_rel,
        "max_abs_error_subnormal": worst_abs,
        "max_abs_error_subnormal_in_2^-149": worst_abs / 2.0**-149,
        "at_subnormal": arg_abs,
    }


def scan_log(chunk: int = 1 << 24) -> dict:
    """log on every positive finite float32 (subnormals included): the relative error."""
    worst, arg, n = 0.0, 0.0, 0
    hi = int(np.float32(np.finfo(np.float32).max).view(np.uint32)) + 1
    for x in _floats(1, hi, chunk):
        with np.errstate(all="ignore"):
            y = np.log(x).astype(np.float64)
            r = np.log(x.astype(np.float64))
        n += x.size
        nz = r != 0.0
        rel = np.abs(y[nz] - r[nz]) / np.abs(r[nz])
        i = int(np.argmax(rel))
        if rel[i] > worst:
            worst, arg = float(rel[i]), float(x[nz][i])
        exact_one = ~nz
        if exact_one.any() and np.any(y[exact_one] != 0.0):
            worst, arg = float("inf"), 1.0
    return {"inputs": n, "max_rel_error": worst, "max_rel_error_in_u": worst / U, "at": arg}


def scalar_vs_vector() -> dict:
    """The scans call numpy on arrays; the semantics also calls it on 0-d arrays and on arrays of
    length 1, which numpy may route to other code. The bits are compared on a sample."""
    rng = np.random.default_rng(0)
    xs = np.concatenate(
        [
            rng.uniform(-104, 88.7, 400_000),
            rng.uniform(-1, 1, 200_000),
            rng.uniform(-88, -86, 100_000),
        ]
    ).astype(np.float32)
    ys = np.concatenate(
        [
            rng.uniform(1e-30, 10, 300_000),
            rng.uniform(0.5, 2, 300_000),
            np.exp(rng.uniform(-80, 80, 100_000)),
        ]
    ).astype(np.float32)
    vec_e, vec_l = np.exp(xs), np.log(ys)
    dif_e = sum(
        np.exp(np.asarray(x)).view(np.uint32) != v.view(np.uint32)
        for x, v in zip(xs, vec_e, strict=True)
    )
    dif_l = sum(
        np.log(np.asarray(y)).view(np.uint32) != v.view(np.uint32)
        for y, v in zip(ys, vec_l, strict=True)
    )
    ones = np.array([np.exp(xs[i : i + 1])[0] for i in range(0, 700_000, 7)])
    dif_e1 = int(np.sum(ones.view(np.uint32) != vec_e[0:700_000:7].view(np.uint32)))
    return {
        "exp_sample": int(xs.size),
        "exp_differ": int(dif_e),
        "log_sample": int(ys.size),
        "log_differ": int(dif_l),
        "exp_len1_sample": int(ones.size),
        "exp_len1_differ": dif_e1,
    }


def scan_float16() -> dict:
    """exp and log on every finite float16 (u = 2^-11, subnormal spacing 2^-24): the errors the
    model's parameters must cover in that format. Overflow is counted where float64 is finite."""
    u16 = 2.0**-11
    x = np.arange(0, 1 << 16, dtype=np.uint32).astype(np.uint16).view(np.float16)
    x = x[np.isfinite(x)]
    with np.errstate(all="ignore"):
        y = np.exp(x).astype(np.float64)
        r = np.exp(x.astype(np.float64))
    fin = np.isfinite(y) & np.isfinite(r)
    tiny = 2.0**-14
    normal = fin & (r >= tiny)
    rel = np.abs(y[normal] - r[normal]) / r[normal]
    sub = fin & (r < tiny)
    ab = np.abs(y[sub] - r[sub])
    pos = x[(x > 0) & np.isfinite(x)]
    with np.errstate(all="ignore"):
        ly = np.log(pos).astype(np.float64)
        lr = np.log(pos.astype(np.float64))
    nz = lr != 0
    lrel = np.abs(ly[nz] - lr[nz]) / np.abs(lr[nz])
    return {
        "format": "float16",
        "inputs": int(x.size),
        "exp_max_rel_in_u": float(rel.max() / u16),
        "exp_overflowed": int((~np.isfinite(y) & np.isfinite(r)).sum()),
        "exp_max_abs_subnormal_in_2^-24": float(ab.max() / 2.0**-24) if ab.size else 0.0,
        "log_max_rel_in_u": float(lrel.max() / u16),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="discovery/results/m2/libm.json")
    ap.add_argument("--exp-only", action="store_true", help="keep the log scan already in --out")
    ap.add_argument(
        "--extra-only",
        action="store_true",
        help="keep both float32 scans in --out; redo the scalar check and the float16 scan",
    )
    args = ap.parse_args(argv)
    t0 = time.time()
    old = json.loads(Path(args.out).read_text()) if args.exp_only or args.extra_only else {}
    out = {
        "numpy": np.__version__,
        "machine": platform.processor() or platform.machine(),
        "exp": old["exp"] if args.extra_only else scan_exp(),
        "log": old["log"] if args.exp_only or args.extra_only else scan_log(),
    }
    out["seconds"] = old["seconds"] if args.extra_only else round(time.time() - t0, 1)
    out["scalar_vs_vector"] = scalar_vs_vector()
    out["float16"] = scan_float16()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
