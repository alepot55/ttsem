"""The numpy reading of a program: any blocking, any merge tree, any floating-point format.

Shapes: a block-level scalar is ``()``, a block-level vector ``(D,)``; inside a reduction an
element scalar is ``(b, 1)`` and an element vector ``(b, D)``, so numpy's broadcasting is the
IR's. The floating-point format is a parameter (:class:`Arith`): float64 for the exact tests,
float32, float16, or bfloat16 emulated by rounding a float32 result after every operation.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np

from discovery import ir

# --------------------------------------------------------------------------- arithmetic


def round_bf16(a: np.ndarray) -> np.ndarray:
    """Round float32 values to the nearest bfloat16 (ties to even), kept as float32."""
    a = np.asarray(a, dtype=np.float32)
    bits = a.view(np.uint32).astype(np.uint64)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)
    return np.where(np.isfinite(a), rounded, a).astype(np.float32)


class Arith:
    """The format every value of a run is stored in, and the rounding after each operation."""

    def __init__(self, name: str = "float64") -> None:
        self.name = name
        if name == "bfloat16":
            self.dtype: type = np.float32
            self.round: Callable[[np.ndarray], np.ndarray] | None = round_bf16
        else:
            self.dtype = {"float64": np.float64, "float32": np.float32, "float16": np.float16}[name]
            self.round = None

    def value(self, a) -> np.ndarray:
        a = np.asarray(a, dtype=self.dtype)
        return self.round(a) if self.round is not None else a


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


class EvalError(Exception):
    pass


def evaluate(
    e: ir.Expr, env: dict[str, np.ndarray], elems: dict[str, np.ndarray] | None, A: Arith
) -> np.ndarray:
    if isinstance(e, ir.Num):
        return A.value(e.value)
    if isinstance(e, ir.Var):
        if elems is not None and e.name in elems:
            return elems[e.name]
        try:
            return env[e.name]
        except KeyError:
            raise EvalError(f"unbound name {e.name}") from None
    if isinstance(e, ir.Reduce):
        assert elems is not None
        b = next(iter(elems.values())).shape[0]
        body = evaluate(e.body, env, elems, A)
        body = (
            np.broadcast_to(body, (b, 1) if body.ndim == 0 else (b, *body.shape[-1:]))
            if body.ndim < 2
            else body
        )
        if e.op == "sum":
            out = np.sum(body, axis=0, dtype=A.dtype)
        elif e.op == "rmax":
            out = np.max(body, axis=0)
        else:
            out = np.min(body, axis=0)
        out = A.value(out)
        return out.reshape(()) if out.shape == (1,) else out
    assert isinstance(e, ir.Call)
    args = [evaluate(a, env, elems, A) for a in e.args]
    fn = _FN[e.op]
    acc = fn(*args) if len(args) <= 2 else _fold(fn, args)
    return A.value(acc)


def _fold(fn: Callable, args: list) -> np.ndarray:
    acc = args[0]
    for a in args[1:]:
        acc = fn(acc, a)
    return acc


# --------------------------------------------------------------------------- passes


def block_summary(
    p: ir.Pass, elems: dict[str, np.ndarray], finished: dict, A: Arith
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, e in p.block:
        out[name] = evaluate(e, {**finished, **out}, elems, A)
    return {f: out[f] for f in p.fields}


def merge(p: ir.Pass, L: dict, R: dict, finished: dict, A: Arith) -> dict[str, np.ndarray]:
    env = dict(finished)
    env.update({f"L.{k}": v for k, v in L.items()})
    env.update({f"R.{k}": v for k, v in R.items()})
    out: dict[str, np.ndarray] = {}
    for name, e in p.merge:
        out[name] = evaluate(e, {**env, **out}, None, A)
    return {f: out[f] for f in p.fields}


def blocking(n: int, kind: str, rng: np.random.Generator, size: int = 128) -> list[tuple[int, int]]:
    """Consecutive nonempty blocks covering ``range(n)``: ``one``, ``single``, ``fixed`` (of
    ``size``, the last one shorter), ``random`` cut points (many small blocks) or ``few`` (one to
    three random cuts, so a few large blocks)."""
    if kind == "one" or n == 1:
        return [(0, n)]
    if kind == "single":
        return [(i, i + 1) for i in range(n)]
    if kind == "fixed":
        return [(i, min(i + size, n)) for i in range(0, n, size)]
    many = int(rng.integers(1, min(n, 4))) if kind == "few" else int(rng.integers(1, n))
    cuts = sorted(set(rng.choice(np.arange(1, n), size=many, replace=False).tolist()))
    edges = [0, *cuts, n]
    return list(zip(edges[:-1], edges[1:], strict=True))


def fold(states: list, combine: Callable, order: str, rng: np.random.Generator):
    """Combine the block summaries in sequence order: ``left``, ``right`` or a random ``tree``."""
    if order == "left":
        acc = states[0]
        for s in states[1:]:
            acc = combine(acc, s)
        return acc
    if order == "right":
        acc = states[-1]
        for s in reversed(states[:-1]):
            acc = combine(s, acc)
        return acc

    def tree(lo: int, hi: int):
        if hi - lo == 1:
            return states[lo]
        k = int(rng.integers(lo + 1, hi))
        return combine(tree(lo, k), tree(k, hi))

    return tree(0, len(states))


@dataclasses.dataclass
class Schedule:
    """How every pass of one run splits and combines: the same for all passes, or drawn per pass."""

    kind: str = "fixed"  # one | single | fixed | random
    order: str = "left"  # left | right | tree
    size: int = 128


def run(
    prog: ir.Program,
    elems: dict[str, np.ndarray],
    A: Arith | None = None,
    schedule: Schedule | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Run ``prog`` on ``elems`` (field -> array of shape (N, 1) or (N, D))."""
    A = A or Arith("float64")
    schedule = schedule or Schedule()
    rng = rng or np.random.default_rng(0)
    elems = {k: A.value(v) for k, v in elems.items()}
    n = next(iter(elems.values())).shape[0]
    finished: dict[str, np.ndarray] = {}
    with np.errstate(all="ignore"):
        for p in prog.passes:
            blocks = blocking(n, schedule.kind, rng, schedule.size)
            states = [
                block_summary(p, {k: v[s:t] for k, v in elems.items()}, finished, A)
                for s, t in blocks
            ]
            final = fold(states, lambda L, R, p=p: merge(p, L, R, finished, A), schedule.order, rng)
            for f, v in final.items():
                finished[f"{p.name}.{f}"] = v
        if prog.output.kind == "elementwise":
            y = evaluate(prog.output.exprs[0][1], finished, elems, A)
            y = np.broadcast_to(y, (n, 1)) if y.ndim < 2 else y
            return {
                "y": np.asarray(y).reshape(n, -1).squeeze(-1) if y.shape[-1] == 1 else np.asarray(y)
            }
        return {name: np.asarray(evaluate(e, finished, None, A)) for name, e in prog.output.exprs}
