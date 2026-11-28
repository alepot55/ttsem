"""Floating-point semantics of a program: one rounding per operation, in the kernel's order.

``semantics.run`` reads the IR with numpy's reductions: a block's ``sum`` adds in numpy's order,
and in bfloat16 it rounds once per reduction. Here every operation of the IR is one rounding in
the chosen format (float32, float16, bfloat16), a block's ``sum`` adds along an explicit tree
(``pairwise``, balanced like ``tl.sum``, or ``sequential``), the block summaries merge along any
tree (``left`` is the generated kernel's loop), and the variadic ``+`` and ``*`` fold left to
right as the generated kernel does.

The reference is the specification computed exactly: rationals for the variance, mpmath for the
exponentials. ``exact=False`` uses float64 with ``math.fsum`` instead, fast enough for the
adversary's inner loop; ``tests/test_fsem.py`` checks it against the exact one.

Errors are reported in units of the format's unit roundoff ``u`` (2^-24 for float32):

* softmax: ``normalizer``, the relative error of the normalizer that the final state encodes,
  read by evaluating the output expression exactly at an element equal to the row's maximum; and
  ``output``, the l1 error of the probability vector;
* attention: ``output``, max over the D components of |o^ - o| / sum_j p_j |v_j|;
* variance: ``var``, the relative error of the variance, and ``mean``, the error of the mean
  against max(|mean|, standard deviation).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from fractions import Fraction

import numpy as np

from discovery import ir, semantics

# --------------------------------------------------------------------------- formats


@dataclasses.dataclass(frozen=True)
class Format:
    """A binary floating-point format: ``p`` bits of precision (with the implicit one), ``emin``
    the exponent of the smallest normal number. Values are held in ``dtype``; bfloat16 is held in
    float32 and rounded after every operation, which is exact rounding for + - * / since
    24 >= 2*8 + 2."""

    name: str
    p: int
    emin: int
    dtype: type

    @property
    def u(self) -> float:
        """Unit roundoff: the largest relative error of one correctly rounded operation."""
        return 2.0 ** (-self.p)

    @property
    def eta(self) -> float:
        """Half the spacing of the subnormals: the largest absolute error of one correctly
        rounded operation whose result is subnormal."""
        return 2.0 ** (self.emin - self.p)

    def fix(self, a) -> np.ndarray:
        a = np.asarray(a, dtype=self.dtype)
        return semantics.round_bf16(a) if self.name == "bfloat16" else a

    def value(self, a) -> np.ndarray:
        """Round values given in float64 to the format (inputs and constants)."""
        a = np.asarray(a, dtype=np.float64)
        if self.name == "bfloat16":
            # float64 -> float32 -> bfloat16 can round twice; round from float64 directly
            return _bf16_from64(a)
        return a.astype(self.dtype)


def _bf16_from64(a: np.ndarray) -> np.ndarray:
    """Round float64 values to the nearest bfloat16 (ties to even), held in float32."""
    a = np.asarray(a, dtype=np.float64)
    out = np.empty(a.shape, dtype=np.float32)
    flat_in, flat_out = a.reshape(-1), out.reshape(-1)
    for i, v in enumerate(flat_in.tolist()):
        if not math.isfinite(v) or v == 0.0:
            flat_out[i] = np.float32(v)
            continue
        m, e = math.frexp(v)  # v = m * 2^e, 0.5 <= |m| < 1
        e_eff = max(e, -125)  # below the normal range the spacing is fixed (subnormals)
        q = math.ldexp(1.0, e_eff - 8)  # spacing of bfloat16 numbers near v
        k = v / q
        r = round(k)  # Python rounds half to even
        flat_out[i] = np.float32(r * q)
    return out


FORMATS: dict[str, Format] = {
    "float32": Format("float32", 24, -126, np.float32),
    "float16": Format("float16", 11, -14, np.float16),
    "bfloat16": Format("bfloat16", 8, -126, np.float32),
}

_FN: dict[str, Callable] = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": np.divide,
    "max": np.maximum,
    "min": np.minimum,
    "neg": np.negative,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
}

INNER = ("pairwise", "sequential")
ORDERS = ("left", "right", "balanced", "random")

# --------------------------------------------------------------------------- evaluation


def _tree_sum(F: Format, a: np.ndarray, inner: str) -> np.ndarray:
    """Sum the rows of ``a`` (shape (b, k)) with one rounding per addition."""
    if inner == "sequential":
        acc = a[0]
        for i in range(1, a.shape[0]):
            acc = F.fix(np.add(acc, a[i]))
        return acc
    while a.shape[0] > 1:  # adjacent pairs, level by level: depth ceil(log2 b)
        h = a.shape[0] // 2
        s = F.fix(np.add(a[0 : 2 * h : 2], a[1 : 2 * h : 2]))
        a = np.concatenate([s, a[2 * h :]]) if a.shape[0] % 2 else s
    return a[0]


def evaluate(
    e: ir.Expr, env: dict, elems: dict | None, F: Format, inner: str = "pairwise"
) -> np.ndarray:
    if isinstance(e, ir.Num):
        return F.value(e.value)
    if isinstance(e, ir.Var):
        if elems is not None and e.name in elems:
            return elems[e.name]
        return env[e.name]
    if isinstance(e, ir.Reduce):
        assert elems is not None
        b = next(iter(elems.values())).shape[0]
        body = evaluate(e.body, env, elems, F, inner)
        if body.ndim < 2:
            body = np.broadcast_to(body, (b, 1) if body.ndim == 0 else (b, *body.shape[-1:]))
        if e.op == "sum":
            out = _tree_sum(F, np.asarray(body), inner)
        elif e.op == "rmax":
            out = np.max(body, axis=0)
        else:
            out = np.min(body, axis=0)
        out = F.fix(out)
        return out.reshape(()) if out.shape == (1,) else out
    assert isinstance(e, ir.Call)
    args = [evaluate(a, env, elems, F, inner) for a in e.args]
    if e.op == "ifgt":  # a comparison and a selection: exact
        return F.fix(np.where(args[0] > args[1], args[2], args[3]))
    fn = _FN[e.op]
    acc = F.fix(fn(*args[:2])) if len(args) >= 2 else F.fix(fn(args[0]))
    for a in args[2:]:  # (+ a b c) is (a + b) + c, as the kernel folds it
        acc = F.fix(fn(acc, a))
    return acc


def _block(p: ir.Pass, elems: dict, finished: dict, F: Format, inner: str) -> dict:
    out: dict = {}
    for name, e in p.block:
        out[name] = evaluate(e, {**finished, **out}, elems, F, inner)
    return {f: out[f] for f in p.fields}


def _merge(p: ir.Pass, L: dict, R: dict, finished: dict, F: Format) -> dict:
    env = dict(finished)
    env.update({f"L.{k}": v for k, v in L.items()})
    env.update({f"R.{k}": v for k, v in R.items()})
    out: dict = {}
    for name, e in p.merge:
        out[name] = evaluate(e, {**env, **out}, None, F)
    return {f: out[f] for f in p.fields}


def combine(states: list, merge: Callable, order: str, rng: np.random.Generator):
    """Merge the block summaries in sequence order along a tree; return (state, depth), and set
    ``combine.rdepth`` to the tree's right depth (a leaf 0, a node max(left, right + 1))."""

    def go(lo: int, hi: int):
        if hi - lo == 1:
            return states[lo], 0, 0
        if order == "left":
            k = hi - 1
        elif order == "right":
            k = lo + 1
        elif order == "balanced":
            k = (lo + hi + 1) // 2
        else:
            k = int(rng.integers(lo + 1, hi))
        a, da, ra = go(lo, k)
        b, db, rb = go(k, hi)
        return merge(a, b), max(da, db) + 1, max(ra, rb + 1)

    if order == "left":  # iterative, so that thousands of blocks do not recurse
        acc = states[0]
        for s in states[1:]:
            acc = merge(acc, s)
        combine.rdepth = 1 if len(states) > 1 else 0
        return acc, len(states) - 1
    state, depth, rdepth = go(0, len(states))
    combine.rdepth = rdepth
    return state, depth


@dataclasses.dataclass
class Run:
    outputs: dict[str, np.ndarray]  # float64 copies of the computed outputs
    finished: dict[str, np.ndarray]  # "p.f" -> final state field, in the format
    depth: int  # depth of the merge tree of the last pass
    blocks: int
    rdepth: int = 0  # right depth of that tree: a leaf 0, a node max(left, right + 1)


def run(
    prog: ir.Program,
    elems: dict[str, np.ndarray],
    fmt: str = "float32",
    block: int = 256,
    inner: str = "pairwise",
    order: str = "left",
    rng: np.random.Generator | None = None,
    blocking: list[tuple[int, int]] | None = None,
    params: dict[str, float] | None = None,
) -> Run:
    """Run ``prog`` on ``elems`` (field -> (N, 1) or (N, D) values already in the format);
    ``params`` overrides the program's named constants."""
    F = FORMATS[fmt]
    rng = rng or np.random.default_rng(0)
    elems = {k: F.fix(v) for k, v in elems.items()}
    n = next(iter(elems.values())).shape[0]
    cuts = blocking or [(i, min(i + block, n)) for i in range(0, n, block)]
    finished: dict = {k: F.value(v) for k, v in {**prog.params, **(params or {})}.items()}
    depth = 0
    with np.errstate(all="ignore"):
        for p in prog.passes:
            states = [
                _block(p, {k: v[s:t] for k, v in elems.items()}, finished, F, inner)
                for s, t in cuts
            ]
            final, depth = combine(
                states, lambda L, R, p=p: _merge(p, L, R, finished, F), order, rng
            )
            for f, v in final.items():
                finished[f"{p.name}.{f}"] = v
        if prog.output.kind == "elementwise":
            y = evaluate(prog.output.exprs[0][1], finished, elems, F, inner)
            y = np.broadcast_to(y, (n, 1)) if y.ndim < 2 else y
            outs = {"y": np.asarray(y, dtype=np.float64).reshape(n, -1).squeeze(-1)}
        else:
            outs = {
                name: np.asarray(evaluate(e, finished, None, F, inner), dtype=np.float64)
                for name, e in prog.output.exprs
            }
    return Run(outs, finished, depth, len(cuts), getattr(combine, "rdepth", 0))


# --------------------------------------------------------------------------- the reference


def _mp():
    import mpmath

    mpmath.mp.prec = 160
    return mpmath


def reference(problem: str, elems: dict[str, np.ndarray], exact: bool = True) -> dict:
    """The specification on the (already rounded) inputs: exact, or float64 with fsum."""
    if problem == "softmax":
        x = elems["x"][:, 0].astype(np.float64)
        m = float(x.max())
        if exact:
            mp = _mp()
            terms = [mp.exp(mp.mpf(float(t)) - mp.mpf(m)) for t in x.tolist()]
            D = mp.fsum(terms)
            y = np.array([float(t / D) for t in terms])
            return {"m": m, "D": D, "y": y, "exact": True}
        terms = np.exp(x - m)
        D = math.fsum(terms.tolist())
        return {"m": m, "D": D, "y": terms / D, "exact": False}
    if problem == "attention":
        s = elems["s"][:, 0].astype(np.float64)
        v = elems["v"].astype(np.float64)
        m = float(s.max())
        if exact:
            mp = _mp()
            p = [mp.exp(mp.mpf(float(t)) - mp.mpf(m)) for t in s.tolist()]
            D = mp.fsum(p)
            o, scale = [], []
            for k in range(v.shape[1]):
                col = v[:, k].tolist()
                o.append(float(mp.fsum(pj * c for pj, c in zip(p, col, strict=True)) / D))
                scale.append(float(mp.fsum(pj * abs(c) for pj, c in zip(p, col, strict=True)) / D))
            return {"o": np.array(o), "scale": np.array(scale), "exact": True}
        p = np.exp(s - m)
        D = math.fsum(p.tolist())
        o = np.array([math.fsum((p * v[:, k]).tolist()) / D for k in range(v.shape[1])])
        scale = np.array([math.fsum((p * np.abs(v[:, k])).tolist()) / D for k in range(v.shape[1])])
        return {"o": o, "scale": scale, "exact": False}
    if problem == "variance":
        x = elems["x"][:, 0].astype(np.float64).tolist()
        n = len(x)
        if exact:
            fx = [Fraction(t) for t in x]
            mean = sum(fx, Fraction(0)) / n
            var = sum(((t - mean) ** 2 for t in fx), Fraction(0)) / n
            return {"mean": float(mean), "var": float(var), "mean_q": mean, "var_q": var}
        mean = math.fsum(x) / n
        var = math.fsum((t - mean) ** 2 for t in x) / n
        return {"mean": mean, "var": var}
    raise ValueError(problem)


# --------------------------------------------------------------------------- errors


def _normalizer(prog: ir.Program, r: Run, m: float) -> float:
    """The normalizer the final state encodes, relative to exp(m): 1 / y(m), with the output
    expression evaluated in float64 at an element equal to ``m``."""
    F64 = Format("float64", 53, -1022, np.float64)
    env = {k: np.asarray(v, dtype=np.float64) for k, v in r.finished.items()}
    elems = {prog.inputs[0]: np.array([[m]])}
    with np.errstate(all="ignore"):
        y = evaluate(prog.output.exprs[0][1], env, elems, F64)
    y0 = float(np.asarray(y).reshape(-1)[0])
    return 1.0 / y0 if y0 != 0.0 and math.isfinite(y0) else math.inf  # inf: the state overflowed


def errors(problem: str, prog: ir.Program, r: Run, ref: dict, fmt: str) -> dict[str, float]:
    """Errors of a run against the reference, in units of the format's u (inf if not finite)."""
    u = FORMATS[fmt].u
    if problem == "softmax":
        D = float(ref["D"])
        zhat = _normalizer(prog, r, ref["m"])
        y = r.outputs["y"]
        if not (math.isfinite(zhat) and np.all(np.isfinite(y))):
            return {"normalizer": math.inf, "output": math.inf}
        return {
            "normalizer": abs(zhat / D - 1.0) / u,
            "output": float(np.sum(np.abs(y - ref["y"]))) / u,
        }
    if problem == "attention":
        o = r.outputs["o"].reshape(-1)
        if not np.all(np.isfinite(o)):
            return {"output": math.inf}
        return {"output": float(np.max(np.abs(o - ref["o"]) / ref["scale"])) / u}
    if problem == "variance":
        mean, var = float(r.outputs["mean"]), float(r.outputs["var"])
        if not (math.isfinite(mean) and math.isfinite(var)):
            return {"var": math.inf, "mean": math.inf}
        rv, rm = ref["var"], ref["mean"]
        e_var = abs(var - rv) / rv if rv > 0 else (0.0 if var == 0 else math.inf)
        return {
            "var": e_var / u,
            "mean": abs(mean - rm) / max(abs(rm), math.sqrt(max(rv, 0.0)), 1e-300) / u,
        }
    raise ValueError(problem)


METRIC = {"softmax": "normalizer", "attention": "output", "variance": "var"}


def measure(
    problem: str,
    prog: ir.Program,
    elems64: dict[str, np.ndarray],
    fmt: str = "float32",
    exact: bool = True,
    **kw,
) -> dict[str, float]:
    """Round the inputs to the format, run, and return the errors in units of u."""
    F = FORMATS[fmt]
    elems = {k: F.value(v) for k, v in elems64.items()}
    r = run(prog, elems, fmt, **kw)
    ref = reference(problem, {k: np.asarray(v, dtype=np.float64) for k, v in elems.items()}, exact)
    out = errors(problem, prog, r, ref, fmt)
    out["depth"] = float(r.depth)
    return out
