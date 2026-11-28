"""Inputs that maximise a program's floating-point error.

Starts from structured families of inputs, then climbs: a move changes a few elements by a few
ulps (the ``nextafter`` move), shifts a whole block, makes ties with a block's maximum, or plants
a new maximum; a move is kept when the error grows. The objective is ``fsem``'s error with the
float64 reference, fast enough for thousands of runs; the worst input found is measured again
with the exact reference (mpmath or rationals) before it is reported or saved.

    python -m discovery.adversary --problem softmax --program known:online --steps 400
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np

from discovery import fsem, ir, problems

# --------------------------------------------------------------------------- families


def _staircase(
    rng: np.random.Generator, n: int, block: int, base: float, step: float
) -> np.ndarray:
    """Every block's maximum above the previous one: the kernel rescales at every merge."""
    x = base + rng.standard_normal(n)
    for b, s in enumerate(range(0, n, block)):
        x[s : s + block] += b * step
    return x


def _twolevel(rng: np.random.Generator, n: int, base: float, gap: float) -> np.ndarray:
    """Most elements ``gap`` below a few maxima: many equal terms in every sum."""
    x = np.full(n, base - gap) + 1e-3 * rng.standard_normal(n)
    x[rng.choice(n, size=max(1, n // 512), replace=False)] = base
    return x


SCORE_FAMILIES = {
    "normal0": lambda rng, n, b: rng.standard_normal(n),
    "normal100": lambda rng, n, b: 100 + rng.standard_normal(n),
    "normal1e4": lambda rng, n, b: 1e4 + rng.standard_normal(n),
    "spread": lambda rng, n, b: 30 * rng.standard_normal(n),
    "staircase": lambda rng, n, b: _staircase(rng, n, b, 0.0, 0.75),
    "twolevel": lambda rng, n, b: _twolevel(rng, n, 0.0, 1.0),
}

VARIANCE_FAMILIES = {
    "normal": lambda rng, n, b: rng.standard_normal(n),
    "offset": lambda rng, n, b: 1e4 + 0.1 * rng.standard_normal(n),
    "tight": lambda rng, n, b: 100 + 1e-3 * rng.standard_normal(n),
    "drift": lambda rng, n, b: _staircase(rng, n, b, 1e3, 0.5) * 1.0,
}


def families(problem: str) -> list[str]:
    return list(VARIANCE_FAMILIES if problem == "variance" else SCORE_FAMILIES)


def generate(problem: str, family: str, rng: np.random.Generator, n: int, block: int) -> dict:
    """Float64 inputs of one row (rounded to the format by ``fsem.measure``)."""
    if problem == "variance":
        return {"x": VARIANCE_FAMILIES[family](rng, n, block)[:, None]}
    s = SCORE_FAMILIES[family](rng, n, block)[:, None]
    if problem == "softmax":
        return {"x": s}
    return {"s": s, "v": rng.standard_normal((n, problems.ATT_DIM))}


# --------------------------------------------------------------------------- moves


def _ulp(F: fsem.Format, v: np.ndarray) -> np.ndarray:
    """Spacing of the format's numbers at ``v`` (normal range; subnormal floor)."""
    a = np.abs(np.asarray(v, dtype=np.float64))
    e = np.floor(np.log2(np.where(a > 0, a, 1.0)))
    e = np.maximum(e, F.emin)
    return np.ldexp(1.0, (e - (F.p - 1)).astype(int))


def mutate(problem: str, elems: dict, F: fsem.Format, rng: np.random.Generator, block: int) -> dict:
    """One random move on a copy of ``elems`` (float64 copies of values in the format)."""
    key = "s" if problem == "attention" else "x"
    out = {k: v.copy() for k, v in elems.items()}
    x = out[key][:, 0]
    n = x.shape[0]
    kind = rng.choice(["ulp", "ulp", "ulp", "block", "tie", "spike", "value"])
    if kind == "ulp" or (kind == "value" and problem != "attention"):
        idx = rng.choice(n, size=int(rng.integers(1, 9)), replace=False)
        steps = rng.integers(-4, 5, size=idx.size)
        x[idx] = x[idx] + steps * _ulp(F, x[idx])
    elif kind == "block":
        s = int(rng.integers(0, max(1, n // block))) * block
        scale = max(float(np.std(x)), 1e-3)
        x[s : s + block] += float(rng.choice([-2, -1, -0.5, 0.5, 1, 2])) * scale
    elif kind == "tie":
        s = int(rng.integers(0, max(1, n // block))) * block
        seg = x[s : s + block]
        idx = rng.choice(seg.size, size=int(rng.integers(1, 9)), replace=False)
        seg[idx] = seg.max()
    elif kind == "spike":
        i = int(rng.integers(0, n))
        s = (i // block) * block
        x[i] = x[s : s + block].max() + float(rng.choice([0.25, 1, 4, 16]))
    else:  # attention: change some value vectors
        v = out["v"]
        idx = rng.choice(n, size=int(rng.integers(1, 9)), replace=False)
        v[idx] = v[idx] * rng.choice([-1.0, 2.0, 0.5, 8.0]) + rng.standard_normal(v[idx].shape)
    out[key] = x[:, None]
    return {k: np.asarray(F.value(v), dtype=np.float64) for k, v in out.items()}


# --------------------------------------------------------------------------- the climb


@dataclasses.dataclass
class Found:
    program: str
    family: str
    seed: int
    start: float  # error of the starting input, units of u (float64 reference)
    error: float  # error of the worst input, units of u, exact reference
    errors: dict  # every metric of the worst input, exact reference
    evaluations: int
    elems: dict  # the worst input, float64 copies of format values


def climb(
    problem: str,
    prog: ir.Program,
    start: dict,
    fmt: str = "float32",
    steps: int = 300,
    rng: np.random.Generator | None = None,
    **runkw,
) -> tuple[dict, float, int]:
    """Hill-climb from ``start``; return (worst input, its error in u (float64 ref), runs)."""
    rng = rng or np.random.default_rng(0)
    F = fsem.FORMATS[fmt]
    metric = fsem.METRIC[problem]
    best = {k: np.asarray(F.value(v), dtype=np.float64) for k, v in start.items()}
    best_e = fsem.measure(problem, prog, best, fmt, exact=False, **runkw)[metric]
    runs = 1
    block = runkw.get("block", 256)
    for _ in range(steps):
        cand = mutate(problem, best, F, rng, block)
        e = fsem.measure(problem, prog, cand, fmt, exact=False, **runkw)[metric]
        runs += 1
        if math.isfinite(e) and e > best_e:
            best, best_e = cand, e
    return best, best_e, runs


def attack(
    problem: str,
    prog: ir.Program,
    fmt: str = "float32",
    n: int = 4096,
    steps: int = 300,
    seeds: tuple[int, ...] = (0, 1),
    family_names: list[str] | None = None,
    **runkw,
) -> list[Found]:
    """Climb from every family and seed; each result is re-measured with the exact reference."""
    out = []
    block = runkw.get("block", 256)
    for fam in family_names or families(problem):
        for seed in seeds:
            rng = np.random.default_rng(1000 * seed + 17)
            start = generate(problem, fam, rng, n, block)
            e0 = fsem.measure(problem, prog, start, fmt, exact=False, **runkw)[fsem.METRIC[problem]]
            best, _, runs = climb(problem, prog, start, fmt, steps, rng, **runkw)
            exact = fsem.measure(problem, prog, best, fmt, exact=True, **runkw)
            out.append(
                Found(prog.name, fam, seed, e0, exact[fsem.METRIC[problem]], exact, runs, best)
            )
    return out


def save_input(path: Path, found: Found) -> None:
    """The worst input, compressed, in float32 (which holds the values of every format exactly)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in found.elems.items()}
    np.savez_compressed(path, **arrays)


# --------------------------------------------------------------------------- command line


def _program(problem: problems.Problem, spec: str) -> ir.Program:
    if spec.startswith("known:"):
        return problem.parse(problem.known[spec[6:]])
    if spec == "seed":
        return problem.parse(problem.seed)
    return problem.parse(Path(spec).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--problem", choices=sorted(problems.PROBLEMS), required=True)
    ap.add_argument("--program", default="seed", help="seed, known:NAME or a file")
    ap.add_argument("--format", default="float32", choices=sorted(fsem.FORMATS))
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--order", default="left", choices=fsem.ORDERS)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args(argv)
    problem = problems.PROBLEMS[args.problem]
    prog = _program(problem, args.program)
    found = attack(
        args.problem,
        prog,
        args.format,
        args.n,
        args.steps,
        tuple(args.seeds),
        block=args.block,
        order=args.order,
    )
    rows = [
        {k: v for k, v in dataclasses.asdict(f).items() if k != "elems"}
        for f in sorted(found, key=lambda f: -f.error)
    ]
    print(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
