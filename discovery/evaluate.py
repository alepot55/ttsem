"""The checks a candidate goes through, cheapest first: parse, exact tests, floating point, cost.

The exact tests run the program in float64 and sample blockings and merge trees: many small
blocks on short rows, blocks of the kernel's size (2048) and a few random large ones on long
rows, rows of repeated values, and the floating-point families' inputs, where float64 itself
may lose a few digits on an ill-conditioned problem. They do not cover every blocking: only a
proof does (`prove.py`). The verdict's ``message`` is what the proposer reads about its program:
the first failure with its counterexample, or the scores of a program that passed.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from discovery import cost, ir, semantics
from discovery.problems import REPEATED, Problem
from discovery.semantics import Arith, Schedule

EXACT_TOL = 1e-9  # float64 runs of a program that is exact over the reals stay near 1e-15
FP_TOL = 1e-3  # float32 error above this, on any input family, is `inaccurate`
# float64 on the adversarial families: an exact but ill-conditioned form (the one-pass sum of
# squares) stays near 1e-6 there, a program wrong over the reals is off by order 1
REAL_TOL = 1e-3
SIZES = (1, 2, 3, 4, 7, 16, 33, 100, 257)
LONG = (4099, 6149)  # rows of two and three kernel blocks
KERNEL_BLOCK = 2048
FP_ROWS = 4  # rows per family in the floating-point tests; the error reported is their median


@dataclasses.dataclass
class Verdict:
    stage: str  # parse | exact | passed
    message: str
    exact_err: float = math.inf
    fp: dict[str, dict[str, float]] = dataclasses.field(default_factory=dict)
    fp_class: str = ""  # stable | inaccurate | nonfinite
    cost: dict = dataclasses.field(default_factory=dict)
    passes: int = 0
    state_fields: int = 0
    nodes: int = 0
    digest: str = ""
    canonical: str = ""  # the digest up to renaming (ir.Program.canonical_digest)
    name: str = ""
    text: str = ""
    # merge(block A, block B) == block(A ++ B) on random splits: what a proof needs
    homomorphic: bool = False

    @property
    def correct(self) -> bool:
        return self.stage == "passed"

    def cell(self) -> tuple[int, int, str]:
        """The MAP-Elites cell: passes over memory, state fields, floating-point class."""
        return (self.passes, self.state_fields, self.fp_class)

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["exact_err"] = None if not math.isfinite(self.exact_err) else self.exact_err
        d["fp"] = {
            f: {k: (None if not math.isfinite(v) else v) for k, v in r.items()}
            for f, r in self.fp.items()
        }
        return d


def _trials(problem: Problem) -> list[tuple[str, int, int, str, str, int, float]]:
    """(family, N, draw, blocking kind, order, block size, tolerance) of the exact stage."""
    out = []
    for n in SIZES:
        for draw in range(2):
            kinds = [("one", "left"), ("random", "left"), ("random", "right")]
            kinds += [("random", "tree"), ("random", "tree")]
            if n <= 64:
                kinds += [("single", "left"), ("single", "tree")]
            out += [("normal", n, draw, k, o, 0, EXACT_TOL) for k, o in kinds]
    for n in LONG:
        for k, o in [("fixed", "left"), ("fixed", "right"), ("fixed", "tree"), ("few", "left")]:
            out.append(("normal", n, 0, k, o, KERNEL_BLOCK, EXACT_TOL))
        out.append(("normal", n, 0, "few", "tree", 0, EXACT_TOL))
    for fam in REPEATED:
        for n in (1, 5, 300):
            for k, o, size in [("one", "left", 0), ("random", "tree", 0), ("fixed", "left", 128)]:
                out.append((fam, n, 0, k, o, size, EXACT_TOL))
    for fam in problem.fp_families:
        if fam != "normal":
            for k, o, size in [("random", "tree", 0), ("fixed", "left", 128), ("few", "right", 0)]:
                out.append((fam, 600, 0, k, o, size, REAL_TOL))
    return out


def _show(a) -> str:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 0:
        return f"{float(a):.6g}"
    flat = a.reshape(-1)
    body = ", ".join(f"{v:.6g}" for v in flat[:8])
    return f"[{body}{', ...' if flat.size > 8 else ''}]"


_HOW = {
    "one": "one block",
    "single": "one block per element",
    "random": "random blocks",
    "few": "a few random large blocks",
}
_COMBINED = {"left": "left to right", "right": "right to left", "tree": "as a random tree"}


def exact_check(
    prog: ir.Program, problem: Problem, rng: np.random.Generator
) -> tuple[float, str | None]:
    """Run ``prog`` in float64 over the sampled blockings and merge trees against the reference.
    Returns the largest error on the ``normal`` rows and, on a failure, a counterexample."""
    worst = 0.0
    rows: dict[tuple[str, int, int], dict] = {}
    for fam, n, draw, kind, order, size, tol in _trials(problem):
        if (fam, n, draw) not in rows:
            rows[fam, n, draw] = problem.generate(rng, n, fam)
        el = rows[fam, n, draw]
        ref = problem.reference(el)
        sch = Schedule(kind, order, size=size or int(rng.integers(1, max(2, n))))
        try:
            out = semantics.run(prog, el, Arith("float64"), sch, rng)
            err = problem.error(out, ref, el)
        except (semantics.EvalError, ValueError, FloatingPointError) as exc:
            return math.inf, f"N={n} ({fam} inputs): the program fails to run ({exc})"
        if fam == "normal":
            worst = max(worst, err)
        if not err <= tol:
            inputs = ", ".join(
                f"{k}={_show(v[:, 0] if v.shape[1] == 1 else v)}" for k, v in el.items()
            )
            got = ", ".join(f"{k}={_show(v)}" for k, v in out.items())
            want = ", ".join(f"{k}={_show(v)}" for k, v in ref.items())
            how = f"blocks of {sch.size}" if kind == "fixed" else _HOW[kind]
            return max(worst, err), (
                f"wrong result for N={n}, {fam} inputs ({inputs}), {how} combined "
                f"{_COMBINED[order]}: expected {want}, got {got} (relative error {err:.3g}, "
                "float64)"
            )
    return worst, None


def homomorphism_check(prog: ir.Program, problem: Problem, rng: np.random.Generator) -> bool:
    """Whether every pass's merge of two adjacent block summaries equals the summary of the two
    blocks together, on random rows split at random points: the statement a proof needs. A
    program can be right end to end without it (a state that is not canonical)."""
    A = Arith("float64")
    for _ in range(3):
        el = problem.generate(rng, 50, "normal")
        finished: dict = {}
        with np.errstate(all="ignore"):
            for p in prog.passes:
                k = int(rng.integers(1, 50))
                whole = semantics.block_summary(p, el, finished, A)
                left = semantics.block_summary(p, {f: v[:k] for f, v in el.items()}, finished, A)
                right = semantics.block_summary(p, {f: v[k:] for f, v in el.items()}, finished, A)
                merged = semantics.merge(p, left, right, finished, A)
                for f in p.fields:
                    a, b = np.asarray(merged[f], float), np.asarray(whole[f], float)
                    scale = max(float(np.max(np.abs(b))), 1.0)
                    if not float(np.max(np.abs(a - b))) <= 1e-9 * scale:
                        return False
                finished.update({f"{p.name}.{f}": v for f, v in whole.items()})
    return True


def fp_check(
    prog: ir.Program, problem: Problem, rng: np.random.Generator, n: int = 2048
) -> tuple[dict, str]:
    """Run ``prog`` in float32, float16 and bfloat16 (blocks of 128, combined in order) on
    :data:`FP_ROWS` rows of each input family, inputs rounded to float32 as they are stored, and
    report the median error of the rows (inf if any row is not finite). ``rng`` should not
    depend on the program, so that every candidate sees the same rows. Returns the errors and
    the float32 class. Float16 and bfloat16 round the inputs to their own format too, so on the
    shifted families their error has a floor no algorithm avoids; only float32 sets the class."""
    res: dict[str, dict[str, float]] = {}
    for fam in problem.fp_families:
        rows = []
        for _ in range(FP_ROWS):
            drawn = problem.generate(rng, n, fam)
            rows.append({k: v.astype(np.float32).astype(np.float64) for k, v in drawn.items()})
        res[fam] = {}
        for dt in ("float32", "float16", "bfloat16"):
            errs = []
            for el in rows:
                try:
                    out = semantics.run(prog, el, Arith(dt), Schedule("fixed", "left", 128), rng)
                    errs.append(problem.error(out, problem.reference(el), el))
                except (semantics.EvalError, ValueError, FloatingPointError):
                    errs.append(math.inf)
            finite = all(math.isfinite(e) for e in errs)
            res[fam][dt] = float(np.median(errs)) if finite else math.inf
    f32 = [r["float32"] for r in res.values()]
    cls = (
        "nonfinite"
        if any(not math.isfinite(e) for e in f32)
        else "inaccurate"
        if max(f32) > FP_TOL
        else "stable"
    )
    return res, cls


def evaluate(
    text: str, problem: Problem, seed: int = 0, device: cost.Device = cost.A1000
) -> Verdict:
    """The full cascade for one candidate text."""
    rng = np.random.default_rng(seed)
    try:
        prog = problem.parse(text)
    except ir.IRError as exc:
        return Verdict("parse", f"does not parse: {exc}", text=text)
    v = Verdict(
        "exact",
        "",
        name=prog.name,
        text=prog.text(),
        digest=prog.digest(),
        canonical=prog.canonical_digest(),
        nodes=ir.size(prog),
    )
    v.passes = sum(1 for fields in ir.reads_elements(prog) if fields)
    v.state_fields = sum(len(p.fields) for p in prog.passes)
    v.exact_err, failure = exact_check(prog, problem, rng)
    if failure is not None:
        v.message = failure
        return v
    v.stage = "passed"
    v.homomorphic = homomorphism_check(prog, problem, rng)
    # the floating-point rows do not depend on the program: every candidate sees the same ones
    v.fp, v.fp_class = fp_check(prog, problem, np.random.default_rng([seed, 1]))
    v.cost = cost.estimate(prog, problem, device)
    worst = ", ".join(f"{fam} {r['float32']:.1e}" for fam, r in v.fp.items())
    proof = "yes" if v.homomorphic else "no, so it cannot be proved by the proof tier"
    v.message = (
        f"correct on every float64 test (error {v.exact_err:.1e} on the normal rows, over "
        f"sampled blockings and merge trees); merge(block A, block B) = block(A ++ B): {proof}; "
        f"{v.passes} pass(es) over memory, {v.state_fields} state field(s); float32 {v.fp_class} "
        f"(median error per input family: {worst}); estimated {v.cost['time_us']:.0f} us on "
        f"{v.cost['device']}, set by the {v.cost['bytes_per_elem']:.0f} bytes moved per element"
    )
    return v
