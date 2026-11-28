"""The floating-point tables of the library: every program's error in units of u, measured.

For each program: random rows of each family (10 seeds: min, median, max) and the worst row the
adversary found (every family, a few seeds, a climb each), all measured with the exact reference.
Writes CSV and JSON under ``discovery/results/`` and the worst inputs as ``.npz`` next to them.

    python -m discovery.fptable --out discovery/results/m1
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np

from discovery import adversary, fsem, ir, problems

LIBRARY = Path(__file__).parent / "lean" / "Discovery" / "Library" / "index.json"


def library() -> list[tuple[str, str, ir.Program]]:
    """(module, problem, program) for every entry of the proof library."""
    out = []
    for e in json.loads(LIBRARY.read_text(encoding="utf-8")):
        prob = problems.PROBLEMS[e["problem"]]
        out.append((e["file"].removesuffix(".lean"), e["problem"], prob.parse(e["program"])))
    return out


def random_rows(
    problem: str, prog: ir.Program, fmt: str, n: int, seeds: int, **runkw
) -> dict[str, list[float]]:
    """Error of each family's rows, exact reference, one value per seed."""
    out: dict[str, list[float]] = {}
    for fam in adversary.families(problem):
        vals = []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            el = adversary.generate(problem, fam, rng, n, runkw.get("block", 256))
            vals.append(
                fsem.measure(problem, prog, el, fmt, exact=True, **runkw)[fsem.METRIC[problem]]
            )
        out[fam] = vals
    return out


def _stats(vals: list[float]) -> tuple[float, float, float]:
    a = np.asarray(vals, dtype=np.float64)
    return float(a.min()), float(np.median(a)), float(a.max())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="discovery/results/m1")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--climb-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--formats", nargs="+", default=["float32", "float16", "bfloat16"])
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runkw = {"block": args.block, "inner": "pairwise", "order": "left"}
    settings = {
        "n": args.n,
        "block": args.block,
        "inner": "pairwise (adjacent pairs, depth ceil(log2 block))",
        "order": "left (the kernel's loop), depth = blocks - 1",
        "random_seeds": args.seeds,
        "climb_steps": args.steps,
        "climb_seeds": args.climb_seeds,
        "metric": fsem.METRIC,
        "unit": {f: fsem.FORMATS[f].u for f in args.formats},
        "numpy": np.__version__,
        "machine": platform.processor() or platform.machine(),
        "reference": "exact: mpmath at 160 bits for exp, rationals for the variance",
    }
    (out / "settings.json").write_text(json.dumps(settings, indent=1), encoding="utf-8")

    rows = []
    for module, problem, prog in library():
        for fmt in args.formats:
            rand = random_rows(problem, prog, fmt, args.n, args.seeds, **runkw)
            row = {"program": module, "problem": problem, "format": fmt}
            for fam, vals in rand.items():
                lo, med, hi = _stats(vals)
                row[f"{fam}_min"], row[f"{fam}_median"], row[f"{fam}_max"] = lo, med, hi
            if fmt == "float32":
                found = adversary.attack(
                    problem, prog, fmt, args.n, args.steps, tuple(args.climb_seeds), **runkw
                )
                worst = max(found, key=lambda f: f.error if np.isfinite(f.error) else -1)
                row["adversary_max"] = worst.error
                row["adversary_family"] = worst.family
                row["adversary_runs"] = sum(f.evaluations for f in found)
                adversary.save_input(out / "worst" / f"{module}-{fmt}.npz", worst)
            rows.append(row)
            print(module, fmt, {k: round(v, 2) for k, v in row.items() if isinstance(v, float)})

    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with (out / "library.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    (out / "library.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
