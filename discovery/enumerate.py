"""The search's baseline with no language model: the merge of a pass, synthesised bottom-up.

The block summaries of a program are taken as given (a known form with its merge thrown away,
say) and the merge is searched field by field, in the order of the block. For field ``f`` the
expressions are built from ``L.g`` and ``R.g`` (every state field ``g``), the fields already
merged, the constants 0 and 1, and the operators ``+ - * / exp max min neg``. Each expression is
evaluated on a fixed set of sample pairs: the summaries ``L = block(A)`` and ``R = block(B)`` of
two random adjacent blocks of a random row, whose target is ``block(A ++ B).f``. Two expressions
with the same values on every sample are one (observational equivalence): the bank keeps one
expression per value.

Expressions are enumerated by size. Up to ``K = (size_cap - 1) // 2`` nodes every one is built.
Bigger ones are reached at their root only, by meeting in the middle: for every expression ``a``
of the bank and every invertible root operator, the other child the target needs (``T - a`` under
``+``, ``T / a`` under ``*``, ...) is computed and looked up among the bank's values. So sizes
``K + 1`` to ``size_cap`` cover the expressions whose root is ``+ - * /`` (or ``exp``, ``neg``)
over children of at most ``K`` nodes, not those with a bigger child. The first expression that
matches the target on every sample is kept, and the whole program is then confirmed by
:func:`discovery.evaluate.evaluate`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections.abc import Iterator

import numpy as np

from discovery import ir, semantics
from discovery.evaluate import Verdict, evaluate
from discovery.problems import PROBLEMS, Problem

UNARY = {"neg": np.negative, "exp": np.exp}
BINARY = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": np.divide,
    "max": np.maximum,
    "min": np.minimum,
}
OPS = [*UNARY, *BINARY]
COMMUTATIVE = {"+", "*", "max", "min"}
SCALAR_ONLY = {"max", "min"}

# For a root ``a op b = t``: the right child given the left one, and the left given the right.
SOLVE_RIGHT = {
    "+": lambda t, a: t - a,
    "-": lambda t, a: a - t,
    "*": lambda t, a: t / a,
    "/": lambda t, a: a / t,
}
SOLVE_LEFT = {
    "+": lambda t, b: t - b,
    "-": lambda t, b: t + b,
    "*": lambda t, b: t / b,
    "/": lambda t, b: t * b,
}

BIG = 1e30  # values beyond this are dropped: they only feed overflow
TINY = 1e-12  # values below this count as 0 when expressions are compared
RTOL, ATOL = 1e-6, 1e-9  # a match: every sample point this close to the target
CHUNK = 1 << 20  # expressions evaluated at once
MAX_FLOATS = 60_000_000  # the memory guard: values kept over all sizes (480 MB)

_MULT = np.random.default_rng(20261128).integers(1, 2**63, 4096, dtype=np.uint64) | np.uint64(1)


def _hash(values: np.ndarray) -> np.ndarray:
    """One 64-bit key per row: its values rounded to float32 (about 7 digits), tiny ones to 0."""
    rounded = np.where(np.abs(values) < TINY, 0.0, values).astype(np.float32)
    bits = rounded.view(np.uint32).astype(np.uint64)
    return (bits * _MULT[: values.shape[1]]).sum(axis=1, dtype=np.uint64)


def _usable(values: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(values) & (np.abs(values) < BIG), axis=1)


class GiveUp(Exception):
    pass


@dataclasses.dataclass
class _Level:
    """The expressions of one size, one per observational class."""

    values: np.ndarray  # (n, P): the value of each expression at every sample point
    vec: np.ndarray  # (n,) the expression's type is vec
    op: np.ndarray  # (n,) index into OPS, -1 for a leaf
    left: np.ndarray  # (n,) the left (or only) child's index in its level, or the leaf's index
    lsize: np.ndarray  # (n,) the left child's size
    right: np.ndarray  # (n,) the right child's index in its level
    hashes: np.ndarray  # (n,) sorted keys, for lookups
    order: np.ndarray  # (n,) the row of each sorted key


class _Search:
    """The bottom-up search for one field's merge expression."""

    def __init__(
        self,
        leaves: list[tuple[ir.Expr, np.ndarray, bool]],
        target: np.ndarray,
        target_vec: bool,
        size_cap: int,
        deadline: float,
    ) -> None:
        self.leaves = [e for e, _, _ in leaves]
        self.leaf_values = np.stack([v for _, v, _ in leaves])
        self.leaf_vec = np.array([t for _, _, t in leaves])
        self.target = target
        self.target_vec = target_vec
        self.size_cap = size_cap
        self.depth = max(1, (size_cap - 1) // 2)
        self.deadline = deadline
        self.levels: list[_Level] = []
        self.seen = np.zeros(0, np.uint64)  # sorted keys of every expression kept
        self.pending: list[tuple[np.ndarray, ...]] = []  # batches of the level being built
        self.found: ir.Expr | None = None
        self.size = 1
        self.enumerated = self.kept = self.lookups = 0

    # ------------------------------------------------------------------ the bank

    def _check_time(self) -> None:
        if time.time() > self.deadline:
            raise GiveUp("time cap reached")

    def _admit(self, values, vec, op, left, lsize, right) -> None:
        """Keep the rows of one batch that are finite and new; stop at one matching the target."""
        self.enumerated += len(vec)
        ok = np.flatnonzero(_usable(values))
        keys = _hash(values[ok])
        uniq, first = np.unique(keys, return_index=True)
        if len(self.seen):
            pos = np.minimum(np.searchsorted(self.seen, uniq), len(self.seen) - 1)
            fresh = self.seen[pos] != uniq
        else:
            fresh = np.ones(len(uniq), dtype=bool)
        rows = ok[first[fresh]]
        self.seen = np.union1d(self.seen, uniq[fresh])
        self.pending.append(
            (
                values[rows],
                vec[rows],
                op[rows].astype(np.int8),
                left[rows].astype(np.int32),
                lsize[rows].astype(np.int8),
                right[rows].astype(np.int32),
            )
        )
        self.kept += len(rows)
        match = np.flatnonzero(
            (vec[rows] == self.target_vec)
            & np.all(np.isclose(values[rows], self.target, rtol=RTOL, atol=ATOL), axis=1)
        )
        if len(match):
            before = sum(len(b[1]) for b in self.pending[:-1])
            self._close_level()
            self.found = self._expr(len(self.levels), before + int(match[0]))
            return
        self._check_time()
        if self.kept * values.shape[1] > MAX_FLOATS:
            raise GiveUp(f"memory guard ({self.kept} expressions kept)")

    def _close_level(self) -> None:
        columns = zip(*self.pending, strict=True)
        values, vec, op, left, lsize, right = (np.concatenate(c) for c in columns)
        self.pending = []
        keys = _hash(values)
        order = np.argsort(keys, kind="stable")
        self.levels.append(_Level(values, vec, op, left, lsize, right, keys[order], order))

    def _grow(self) -> None:
        """Build every expression of the next size from the smaller ones."""
        s = len(self.levels) + 1
        child = self.levels[s - 2]
        idx = np.arange(len(child.vec))
        for name, fn in UNARY.items():
            with np.errstate(all="ignore"):
                values = fn(child.values)
            code = np.full(len(idx), OPS.index(name))
            self._admit(values, child.vec, code, idx, np.full(len(idx), s - 1), idx)
            if self.found is not None:
                return
        for i in range(1, s - 1):
            j = s - 1 - i
            A, B = self.levels[i - 1], self.levels[j - 1]
            for name, fn in BINARY.items():
                if name in COMMUTATIVE and i > j:
                    continue
                ia, ib = np.arange(len(A.vec)), np.arange(len(B.vec))
                if name in SCALAR_ONLY:
                    ia, ib = ia[~A.vec], ib[~B.vec]
                for a, b in _pairs(ia, ib, triangle=name in COMMUTATIVE and i == j):
                    with np.errstate(all="ignore"):
                        values = fn(A.values[a], B.values[b])
                    code = np.full(len(a), OPS.index(name))
                    self._admit(values, A.vec[a] | B.vec[b], code, a, np.full(len(a), i), b)
                    if self.found is not None:
                        return
        self._close_level()

    def _expr(self, size: int, k: int) -> ir.Expr:
        level = self.levels[size - 1]
        code = int(level.op[k])
        if code < 0:
            return self.leaves[int(level.left[k])]
        name = OPS[code]
        if name in UNARY:
            return ir.Call(name, (self._expr(size - 1, int(level.left[k])),))
        i = int(level.lsize[k])
        left = self._expr(i, int(level.left[k]))
        return ir.Call(name, (left, self._expr(size - 1 - i, int(level.right[k]))))

    # ------------------------------------------------------------------ meeting in the middle

    def _find(self, level: _Level, need: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The rows of ``need`` whose values are in ``level``, and their rows there."""
        self.lookups += len(need)
        rows = np.flatnonzero(_usable(need)) if len(level.hashes) else np.zeros(0, np.int64)
        keys = _hash(need[rows])
        pos = np.minimum(np.searchsorted(level.hashes, keys), len(level.hashes) - 1)
        hit = level.hashes[pos] == keys
        return rows[hit], level.order[pos[hit]]

    def _root(self, s: int) -> ir.Expr | None:
        """An expression of size ``s``: an invertible root over two expressions of the bank."""
        t = self.target
        for name in SOLVE_RIGHT:
            for i in range(1, s - 1):
                j = s - 1 - i
                if i > self.depth or j > self.depth or (name in COMMUTATIVE and i > j):
                    continue
                A, B = self.levels[i - 1], self.levels[j - 1]
                with np.errstate(all="ignore"):
                    if len(A.vec) <= len(B.vec):  # look the smaller level's partners up
                        ka, kb = self._find(B, SOLVE_RIGHT[name](t, A.values))
                    else:
                        kb, ka = self._find(A, SOLVE_LEFT[name](t, B.values))
                    for a, b in zip(ka.tolist(), kb.tolist(), strict=True):
                        value = BINARY[name](A.values[a], B.values[b])
                        if bool(A.vec[a] or B.vec[b]) == self.target_vec and np.allclose(
                            value, t, rtol=RTOL, atol=ATOL
                        ):
                            return ir.Call(name, (self._expr(i, a), self._expr(j, b)))
                self._check_time()
        if s == self.depth + 1:
            top = self.levels[self.depth - 1]
            with np.errstate(all="ignore"):
                needs = {"neg": -t, "exp": np.log(t)}
            for name, need in needs.items():
                for k in self._find(top, need[None, :])[1].tolist():
                    if bool(top.vec[k]) == self.target_vec:
                        return ir.Call(name, (self._expr(self.depth, k),))
        return None

    def run(self) -> ir.Expr | None:
        """The smallest expression found that matches the target; None at the size cap."""
        n = len(self.leaves)
        self._admit(
            self.leaf_values, self.leaf_vec, np.full(n, -1), np.arange(n), np.zeros(n), np.zeros(n)
        )
        if self.found is None:
            self._close_level()
        while self.found is None and len(self.levels) < self.depth:
            self.size = len(self.levels) + 1
            self._grow()
        if self.found is not None:
            return self.found
        for s in range(self.depth + 1, self.size_cap + 1):
            self.size = s
            found = self._root(s)
            if found is not None:
                return found
        return None


def _pairs(ia: np.ndarray, ib: np.ndarray, triangle: bool) -> Iterator[tuple[np.ndarray, ...]]:
    """Index pairs (a, b) in chunks; only a <= b when the operator commutes within one level."""
    step = max(1, CHUNK // max(1, len(ib)))
    for start in range(0, len(ia), step):
        part = ia[start : start + step]
        a, b = np.repeat(part, len(ib)), np.tile(ib, len(part))
        if triangle:
            keep = a <= b
            a, b = a[keep], b[keep]
        yield a, b


# --------------------------------------------------------------------------- samples


def _samples(
    problem: Problem, prog: ir.Program, index: int, count: int, rng: np.random.Generator
) -> list[tuple[dict, dict, dict]]:
    """``(L, R, T)`` for random adjacent blocks A and B of random rows: the summaries of pass
    ``index`` over A, over B and over A ++ B, the earlier passes finished over the whole row."""
    p = prog.passes[index]
    A = semantics.Arith("float64")
    out = []
    while len(out) < count:
        n = int(rng.integers(2, 13))
        cut = int(rng.integers(1, n))
        row = problem.generate(rng, n, "normal")
        finished: dict[str, np.ndarray] = {}
        with np.errstate(all="ignore"):
            for q in prog.passes[:index]:
                for f, v in semantics.block_summary(q, row, finished, A).items():
                    finished[f"{q.name}.{f}"] = v
            L = semantics.block_summary(p, {k: v[:cut] for k, v in row.items()}, finished, A)
            R = semantics.block_summary(p, {k: v[cut:] for k, v in row.items()}, finished, A)
            T = semantics.block_summary(p, row, finished, A)
        if all(np.all(np.isfinite(v)) for s in (L, R, T) for v in s.values()):
            out.append((L, R, T))
    return out


def _points(samples: list[tuple[dict, dict, dict]], side: int, field: str, coords: int):
    """A field's value at every sample point: ``coords`` coordinates of a vector per sample (a
    scalar repeats). The operators act coordinate by coordinate, so a few coordinates suffice."""
    cols = []
    for s in samples:
        v = np.asarray(s[side][field], dtype=np.float64)
        cols.append(np.full(coords, float(v)) if v.ndim == 0 else v[:coords])
    return np.concatenate(cols)


# --------------------------------------------------------------------------- the synthesis


@dataclasses.dataclass
class FieldResult:
    pass_name: str
    field: str
    expr: str  # empty when nothing was found
    size: int  # the size found, or the size being searched when it stopped
    seconds: float
    enumerated: int  # expressions built and evaluated on the samples
    distinct: int  # observational classes kept in the bank
    lookups: int  # root lookups, each standing for every expression `a op b` over one level


@dataclasses.dataclass
class Synthesis:
    problem: str
    fields: list[FieldResult]
    text: str | None  # the program with its merges synthesised, when every field was found
    verdict: Verdict | None
    seconds: float
    gave_up: str = ""

    def to_json(self) -> dict:
        return {
            "problem": self.problem,
            "fields": [dataclasses.asdict(f) for f in self.fields],
            "text": self.text,
            "verdict": self.verdict.to_json() if self.verdict else None,
            "seconds": round(self.seconds, 2),
            "gave_up": self.gave_up,
        }


def synthesise(
    problem: Problem,
    text: str,
    size_cap: int = 15,
    time_cap: float = 120.0,
    samples: int = 16,
    seed: int = 0,
) -> Synthesis:
    """Rebuild every merge of ``text`` from its block summaries alone, then evaluate it."""
    prog = problem.parse(text)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    deadline = t0 + time_cap
    coords = min(problem.dim, 2) if "vec" in prog.types.values() else 1
    results: list[FieldResult] = []
    for index, p in enumerate(prog.passes):
        pairs = _samples(problem, prog, index, samples, rng)
        is_vec = {f: prog.types[f"{p.name}.{f}"] == "vec" for f in p.fields}
        merged: list[tuple[str, ir.Expr]] = []
        for f in p.fields:  # temporaries of the block need no merge
            started = time.time()
            leaves = [
                (ir.Var(f"{side}.{g}"), _points(pairs, k, g, coords), is_vec[g])
                for g in p.fields
                for k, side in ((0, "L"), (1, "R"))
            ]
            leaves += [(ir.Var(g), _points(pairs, 2, g, coords), is_vec[g]) for g, _ in merged]
            leaves += [(ir.Num(c), np.full(samples * coords, c), False) for c in (0.0, 1.0)]
            if not is_vec[f]:  # merges have no reductions: a vector never helps make a scalar
                leaves = [leaf for leaf in leaves if not leaf[2]]
            search = _Search(leaves, _points(pairs, 2, f, coords), is_vec[f], size_cap, deadline)
            try:
                found = search.run()
                reason = (
                    f"nothing up to size {size_cap} (every expression up to size {search.depth} "
                    "built; above it only + - * / exp neg over two of those)"
                )
            except GiveUp as stop:
                found, reason = None, f"{stop} while at size {search.size}"
            results.append(
                FieldResult(
                    p.name,
                    f,
                    ir.unparse_expr(found) if found is not None else "",
                    search.size,
                    round(time.time() - started, 2),
                    search.enumerated,
                    search.kept,
                    search.lookups,
                )
            )
            if found is None:
                gave_up = f"pass {p.name}, field {f}: {reason}"
                return Synthesis(problem.name, results, None, None, time.time() - t0, gave_up)
            merged.append((f, found))
        p.merge = merged
    out = prog.text()
    return Synthesis(problem.name, results, out, evaluate(out, problem, seed), time.time() - t0)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--problem", choices=sorted(PROBLEMS), default="softmax")
    ap.add_argument("--form", help="a known form whose block summaries are kept (not its merge)")
    ap.add_argument("--program", help="a file holding the program whose block summaries are kept")
    ap.add_argument("--size-cap", type=int, default=15)
    ap.add_argument("--time-cap", type=float, default=120.0)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    args = ap.parse_args(argv)
    problem = PROBLEMS[args.problem]
    if args.program:
        with open(args.program, encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = problem.known[args.form] if args.form else problem.seed
    res = synthesise(problem, text, args.size_cap, args.time_cap, args.samples, args.seed)
    if args.json:
        print(json.dumps(res.to_json(), indent=2))
        return
    for f in res.fields:
        print(
            f"{f.pass_name}.{f.field} = {f.expr or '(not found)'}: size {f.size}, "
            f"{f.seconds:.1f} s, {f.enumerated} built, {f.distinct} distinct, "
            f"{f.lookups} root lookups"
        )
    if res.verdict is not None:
        print(f"{res.verdict.stage} {res.verdict.cell()}: {res.verdict.message}")
    else:
        print(f"gave up: {res.gave_up}")
    print(f"{res.seconds:.1f} s in all")


if __name__ == "__main__":
    main()
