"""The problems the search runs on: a specification, a seed program, inputs and a reference.

The seed is the textbook multi-pass program; it is also the specification, since a program run
on one block with no merge is the operator's definition. ``known`` holds the published
reformulations, for the tests and to recognise a rediscovery; the proposer's prompt never
quotes them (a proved lemma of the library can state one: the control runs use no library).

Every generator also gives rows of repeated values (``zeros``, ``threes``, ``small-int``): a
merge that divides by a difference of maxima or means is undefined there.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import numpy as np

from discovery import ir


@dataclasses.dataclass
class Problem:
    name: str
    description: str  # what the operator computes, for the proposer
    blind_description: str  # the same with the operator's name and wording removed
    elems: dict[str, str]  # element field -> scalar | vec
    outputs: str | dict[str, str]  # "elementwise", or output name -> type
    seed: str
    reference: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]
    generate: Callable[[np.random.Generator, int, str], dict[str, np.ndarray]]
    fp_families: list[str]  # input families of the floating-point tests
    field_bytes: dict[str, int]  # bytes read from memory per element for each field
    error: Callable[[dict, dict, dict], float]
    known: dict[str, str] = dataclasses.field(default_factory=dict)
    dim: int = 1  # D, the length of a vector field in the semantics tests
    lean_elem: str = "ℝ"  # the element type in the Lean export

    def parse(self, text: str) -> ir.Program:
        """Parse ``text`` and check it against this problem's elements and outputs."""
        prog = ir.parse(text)
        if set(prog.inputs) != set(self.elems):
            raise ir.IRError(
                f"(input ...) must name this problem's element fields ({' '.join(self.elems)}), "
                f"not ({' '.join(prog.inputs)})"
            )
        ir.check(prog, self.elems)
        if self.outputs == "elementwise":
            if prog.output.kind != "elementwise":
                raise ir.IRError("this problem's output is (output elementwise EXPR)")
        else:
            assert isinstance(self.outputs, dict)
            names = [n for n, _ in prog.output.exprs]
            if prog.output.kind != "final" or sorted(names) != sorted(self.outputs):
                want = " ".join(f"({n} EXPR)" for n in self.outputs)
                raise ir.IRError(f"this problem's output is (output final {want})")
        return prog


def _normwise(out: dict, ref: dict, elems: dict) -> float:
    worst = 0.0
    for k, r in ref.items():
        a = np.asarray(out.get(k), dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)
        if a.shape != r.shape or not np.all(np.isfinite(a)):
            return math.inf
        worst = max(worst, float(np.max(np.abs(a - r))) / max(float(np.max(np.abs(r))), 1e-300))
    return worst


# --------------------------------------------------------------------------- softmax


def _softmax_ref(el: dict) -> dict:
    x = el["x"][:, 0].astype(np.float64)
    e = np.exp(x - x.max())
    return {"y": e / math.fsum(e.tolist())}


REPEATED = ("zeros", "threes", "small-int")


def _repeated(rng: np.random.Generator, n: int, family: str) -> np.ndarray:
    """Rows whose values repeat and are exact in every format: all 0, all 3, or small integers."""
    if family == "zeros":
        return np.zeros(n)
    if family == "threes":
        return np.full(n, 3.0)
    return rng.integers(-3, 4, n).astype(np.float64)


def _logits(rng: np.random.Generator, n: int, family: str) -> np.ndarray:
    """Logits of one row: ``normal``, ``large`` (exp overflows in float32 without a shift),
    ``negative`` (exp underflows to zero without a shift), ``spread`` (a wide range)."""
    if family in REPEATED:
        return _repeated(rng, n, family)
    if family == "normal":
        return rng.standard_normal(n) * 3
    if family == "large":
        return rng.standard_normal(n) * 10 + 200
    if family == "negative":
        return rng.standard_normal(n) * 10 - 200
    if family == "spread":
        return rng.standard_normal(n) * 60
    raise ValueError(family)


SOFTMAX = Problem(
    name="softmax",
    description=(
        "Row softmax. The elements of one row are x_1..x_N (float32 in memory). Output, for every "
        "element: y_i = exp(x_i - c) / sum_j exp(x_j - c), with c = max_j x_j."
    ),
    blind_description=(
        "Operator P. The elements of one row are x_1..x_N (float32 in memory). Output, for every "
        "element i: y_i = E(x_i - c) / S, where c = max_j x_j, S = sum_j E(x_j - c) and E is exp."
    ),
    elems={"x": "scalar"},
    outputs="elementwise",
    seed="""(program softmax-3pass
  (input x)
  (pass a (state m) (block (m (rmax x))) (merge (m (max L.m R.m))))
  (pass b (state d) (block (d (sum (exp (- x a.m))))) (merge (d (+ L.d R.d))))
  (output elementwise (/ (exp (- x a.m)) b.d)))""",
    reference=_softmax_ref,
    generate=lambda rng, n, fam: {"x": _logits(rng, n, fam)[:, None]},
    fp_families=["normal", "large", "negative", "spread"],
    field_bytes={"x": 4},
    error=_normwise,
    known={
        "online": """(program softmax-online
  (input x)
  (pass a
    (state m d)
    (block (m (rmax x)) (d (sum (exp (- x m)))))
    (merge (m (max L.m R.m))
           (d (+ (* L.d (exp (- L.m m))) (* R.d (exp (- R.m m)))))))
  (output elementwise (/ (exp (- x a.m)) a.d)))""",
        # exact over the reals, but exp overflows or underflows with no shift
        "unshifted": """(program softmax-unshifted
  (input x)
  (pass a (state d) (block (d (sum (exp x)))) (merge (d (+ L.d R.d))))
  (output elementwise (/ (exp x) a.d)))""",
        # one state field, the log of the sum, with temporaries for the maxima
        "log-sum-exp": """(program softmax-lse
  (input x)
  (pass a
    (state s)
    (block (m (rmax x)) (s (+ m (log (sum (exp (- x m)))))))
    (merge (m (max L.s R.s)) (s (+ m (log (+ (exp (- L.s m)) (exp (- R.s m))))))))
  (output elementwise (exp (- x a.s))))""",
        # wrong: adds the sums without rescaling them to the common maximum
        "no-rescale": """(program softmax-no-rescale
  (input x)
  (pass a
    (state m d)
    (block (m (rmax x)) (d (sum (exp (- x m)))))
    (merge (m (max L.m R.m)) (d (+ L.d R.d))))
  (output elementwise (/ (exp (- x a.m)) a.d)))""",
    },
)

# --------------------------------------------------------------------------- attention

ATT_DIM = 8


def _attention_ref(el: dict) -> dict:
    s = el["s"][:, 0].astype(np.float64)
    v = el["v"].astype(np.float64)
    p = np.exp(s - s.max())
    return {"o": (p / math.fsum(p.tolist())) @ v}


def _attention_gen(rng: np.random.Generator, n: int, fam: str, dim: int = ATT_DIM) -> dict:
    return {"s": _logits(rng, n, fam)[:, None], "v": rng.standard_normal((n, dim))}


ATTENTION = Problem(
    name="attention",
    description=(
        "Attention for one query. Element j has a score s_j (q.k_j/sqrt(D), computed from a key "
        "row of D float32 values in memory) and a value vector v_j (D float32 values). Output: "
        "o = sum_j p_j v_j, a vector of length D, with p_j = exp(s_j - c) / sum_k exp(s_k - c) and "
        "c = max_k s_k."
    ),
    blind_description=(
        "Operator Q. Element j has a scalar s_j (computed from D float32 values in memory) and a "
        "vector v_j (D float32 values). Output: o = sum_j w_j v_j with w_j = E(s_j - c) / S, "
        "c = max_k s_k, S = sum_k E(s_k - c), E = exp."
    ),
    elems={"s": "scalar", "v": "vec"},
    outputs={"o": "vec"},
    seed="""(program attention-3pass
  (input s v)
  (pass a (state m) (block (m (rmax s))) (merge (m (max L.m R.m))))
  (pass b (state d) (block (d (sum (exp (- s a.m))))) (merge (d (+ L.d R.d))))
  (pass c (state o) (block (o (sum (* (exp (- s a.m)) v)))) (merge (o (+ L.o R.o))))
  (output final (o (/ c.o b.d))))""",
    reference=_attention_ref,
    generate=_attention_gen,
    fp_families=["normal", "large", "negative", "spread"],
    field_bytes={"s": 4 * 64, "v": 4 * 64},  # a key row and a value row of D=64 float32
    error=_normwise,
    known={
        "online": """(program attention-online
  (input s v)
  (pass a
    (state m d o)
    (block (m (rmax s)) (d (sum (exp (- s m)))) (o (sum (* (exp (- s m)) v))))
    (merge (m (max L.m R.m))
           (d (+ (* L.d (exp (- L.m m))) (* R.d (exp (- R.m m)))))
           (o (+ (* L.o (exp (- L.m m))) (* R.o (exp (- R.m m)))))))
  (output final (o (/ a.o a.d))))""",
    },
    dim=ATT_DIM,
    lean_elem="ℝ × V",
)

# --------------------------------------------------------------------------- variance


def _variance_ref(el: dict) -> dict:
    x = el["x"][:, 0].astype(np.float64).tolist()
    mean = math.fsum(x) / len(x)
    return {
        "mean": np.float64(mean),
        "var": np.float64(math.fsum((t - mean) ** 2 for t in x) / len(x)),
    }


def _variance_gen(rng: np.random.Generator, n: int, fam: str) -> dict:
    """``normal``; ``offset`` (mean 1e4, spread 0.1) and ``tight`` (mean 100, spread 0.001):
    a sum of squares cancels in float32 on both, while the data still have a hundred float32
    ulps of spread, so a stable algorithm can get it right."""
    if fam in REPEATED:
        x = _repeated(rng, n, fam)
    elif fam == "normal":
        x = rng.standard_normal(n)
    elif fam == "offset":
        x = 1e4 + 0.1 * rng.standard_normal(n)
    elif fam == "tight":
        x = 100 + 1e-3 * rng.standard_normal(n)
    else:
        raise ValueError(fam)
    return {"x": x[:, None]}


def _variance_error(out: dict, ref: dict, elems: dict) -> float:
    try:
        mean, var = float(out["mean"]), float(out["var"])
    except (KeyError, TypeError, ValueError):
        return math.inf
    if not (math.isfinite(mean) and math.isfinite(var)):
        return math.inf
    rm, rv = float(ref["mean"]), float(ref["var"])
    spread = math.sqrt(rv) if rv > 0 else 1e-300
    # a zero variance (a constant row) is compared with a floor at the rounding of the mean
    floor = 1e-15 * rm * rm + 1e-300
    return max(abs(mean - rm) / max(abs(rm), spread), abs(var - rv) / max(rv, floor))


VARIANCE = Problem(
    name="variance",
    description=(
        "Mean and variance of a row. The elements are x_1..x_N (float32 in memory). Outputs: "
        "mean = (1/N) sum_i x_i and var = (1/N) sum_i (x_i - mean)^2 (population variance)."
    ),
    blind_description=(
        "Operator R. The elements are x_1..x_N (float32 in memory). Outputs: mean = (1/N) "
        "sum_i x_i and var = (1/N) sum_i (x_i - mean)^2."
    ),
    elems={"x": "scalar"},
    outputs={"mean": "scalar", "var": "scalar"},
    seed="""(program variance-2pass
  (input x)
  (pass a (state n t) (block (n (sum 1)) (t (sum x))) (merge (n (+ L.n R.n)) (t (+ L.t R.t))))
  (pass b
    (state q)
    (block (q (sum (* (- x (/ a.t a.n)) (- x (/ a.t a.n))))))
    (merge (q (+ L.q R.q))))
  (output final (mean (/ a.t a.n)) (var (/ b.q a.n))))""",
    reference=_variance_ref,
    generate=_variance_gen,
    fp_families=["normal", "offset", "tight"],
    field_bytes={"x": 4},
    error=_variance_error,
    known={
        "welford": """(program variance-welford
  (input x)
  (pass a
    (state n mu q)
    (block (n (sum 1)) (mu (/ (sum x) n)) (q (sum (* (- x mu) (- x mu)))))
    (merge (n (+ L.n R.n))
           (mu (+ L.mu (* (- R.mu L.mu) (/ R.n n))))
           (q (+ L.q R.q (* (- R.mu L.mu) (- R.mu L.mu) (/ (* L.n R.n) n))))))
  (output final (mean a.mu) (var (/ a.q a.n))))""",
        # exact over the reals, cancels catastrophically in float32
        "sum-of-squares": """(program variance-sumsq
  (input x)
  (pass a
    (state n t u)
    (block (n (sum 1)) (t (sum x)) (u (sum (* x x))))
    (merge (n (+ L.n R.n)) (t (+ L.t R.t)) (u (+ L.u R.u))))
  (output final (mean (/ a.t a.n)) (var (- (/ a.u a.n) (* (/ a.t a.n) (/ a.t a.n))))))""",
    },
)

PROBLEMS: dict[str, Problem] = {p.name: p for p in (SOFTMAX, ATTENTION, VARIANCE)}
