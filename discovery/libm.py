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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="discovery/results/m2/libm.json")
    ap.add_argument("--exp-only", action="store_true", help="keep the log scan already in --out")
    args = ap.parse_args(argv)
    t0 = time.time()
    out = {
        "numpy": np.__version__,
        "machine": platform.processor() or platform.machine(),
        "exp": scan_exp(),
        "log": scan_log() if not args.exp_only else json.loads(Path(args.out).read_text())["log"],
    }
    out["seconds"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
