"""F1 and F2 for a proved bound: a long attack on numpy float32, and the bound's tightness.

F1 (the brief's falsification table): every proved bound against numpy float32 on random and
adversarial rows; one measured error above the bound means the rounding model is wrong. The attack
here is longer than the one that let the bound through (more steps, more seeds, the worst rows of
M1 as extra starts). F2: the ratio bound / worst error found; above 1,000 for every program the
bounds are empty.

    python -m discovery.fpcheck --program SoftmaxOnline --bound "2*(d+1)*R*u + ..."
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from discovery import adversary, bound, fpbound, fptable, fsem

WORST = Path(__file__).parent / "results" / "m1" / "worst"


def check(module: str, btext: str, steps: int = 400, seeds: tuple[int, ...] = (0, 1, 2, 3)) -> dict:
    lib = {m: (p, prog) for m, p, prog in fptable.library()}
    problem, prog = lib[module]
    b = bound.parse(btext, problem)
    found = fpbound.attack(prog, b, steps=steps, seeds=seeds, problem=problem)
    # the worst rows of M1, as they are and after a climb
    extra = []
    path = WORST / f"{module}-float32.npz"
    if path.exists():
        el = {k: np.asarray(v, dtype=np.float64) for k, v in np.load(path).items()}
        q, info = fpbound.ratio(problem, prog, b, el)
        extra.append({"start": "M1 worst row", "ratio": q, **info})
        F = fsem.FORMATS["float32"]
        rng = np.random.default_rng(11)
        cur = q
        for _ in range(steps):
            cand = adversary.mutate(problem, el, F, rng, 256)
            q2, info2 = fpbound.ratio(problem, prog, b, cand)
            if math.isfinite(q2) and q2 > cur:
                el, cur, info = cand, q2, info2
        extra.append({"start": "M1 worst row, climbed", "ratio": cur, **info})
    worst = max([found, *extra], key=lambda r: r["ratio"])
    return {
        "program": module,
        "B": btext,
        "steps": steps,
        "seeds": list(seeds),
        "worst_ratio": worst["ratio"],
        "F1_violated": worst["ratio"] > 1.0,
        "tightness": 1.0 / worst["ratio"] if worst["ratio"] > 0 else math.inf,
        "worst": {k: v for k, v in worst.items() if k != "x"},
        "attack": {k: v for k, v in found.items() if k != "x"},
        "m1_rows": extra,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--program", required=True)
    ap.add_argument("--bound", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default="discovery/results/m3/f1")
    args = ap.parse_args(argv)
    res = check(args.program, args.bound, args.steps)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"{args.program}.json").write_text(
        json.dumps(res, indent=1), encoding="utf-8"
    )
    print(json.dumps({k: res[k] for k in ("program", "worst_ratio", "F1_violated", "tightness")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
