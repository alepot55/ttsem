"""The grammar of a proposed error bound ``B``, its value on a run, and its Lean term.

``B`` is a sum of products of nonnegative numbers and atoms, with parentheses and small integer
powers; no subtraction, so ``B`` grows with every atom. The atoms:

=======  =====================================================  ===========================
atom     meaning                                                Lean
=======  =====================================================  ===========================
u        relative error of + - * / sqrt                          ``F.u``
ue       relative error of exp                                   ``F.ue``
ul       relative error of log                                   ``F.ul``
eta      absolute error of * / below the normal range           ``F.η``
etae     absolute error of exp below the normal range           ``F.ηe``
D        bound on the depth of the trees of a block's sums       ``(D : ℝ)``
d        depth of the merge tree                                 ``(t.depth : ℝ)``
n        number of elements of the row                           ``(X.length : ℝ)``
R        width of the data's range, ``hi - lo``                  ``(hi - lo)``
M        magnitude of the data's range, ``max |lo| |hi|``        ``(max |lo| |hi|)``
S2       the row's exact variance (variance only)                ``(vr X)``
Q        the row's exact mean of squares (variance only)         ``(msq X)``
S        the attention's scale Σ p|v| / Σ p (attention only)    ``(attnScale X)``
V        the bound on |v| (attention only)                       ``Vmax``
=======  =====================================================  ===========================

Numbers are written as integers, decimals or ``p/q`` with integer ``p`` and ``q``; a decimal is
read as the exact rational it writes.
"""

from __future__ import annotations

import dataclasses
import math
import re
from fractions import Fraction

ATOMS = {
    "u": "F.u",
    "ue": "F.ue",
    "ul": "F.ul",
    "eta": "F.η",
    "etae": "F.ηe",
    "D": "(D : ℝ)",
    "d": "(t.depth : ℝ)",
    "n": "(X.length : ℝ)",
    "R": "(hi - lo)",
    "M": "(max |lo| |hi|)",
}

# Atoms of one problem only: the variance's bound may use the exact variance and mean square.
EXTRA = {
    "variance": {"S2": "(vr X)", "Q": "(msq X)"},
    "attention": {"S": "(attnScale X)", "V": "Vmax"},
    "policy": {"tau": "tau", "rd": "(rdepth t : ℝ)"},
}

MAX_LENGTH = 400
MAX_NODES = 60
MAX_POWER = 4

_TOKEN = re.compile(r"\s*(?:(\d+(?:\.\d+)?)|([A-Za-z][A-Za-z0-9]*)|(\*\*|[+*/^()]))")


class BoundError(Exception):
    """A bound outside the grammar; the message is meant for the model."""


@dataclasses.dataclass(frozen=True)
class Num:
    value: Fraction


@dataclasses.dataclass(frozen=True)
class Atom:
    name: str


@dataclasses.dataclass(frozen=True)
class Op:
    op: str  # + * ^
    args: tuple


Node = Num | Atom | Op


def _tokens(text: str) -> list[str]:
    out, pos = [], 0
    text = text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise BoundError(f"unexpected character at {text[pos : pos + 12]!r}")
        out.append(next(g for g in m.groups() if g is not None))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, toks: list[str], atoms: dict[str, str]) -> None:
        self.toks = toks
        self.atoms = atoms
        self.i = 0

    def peek(self) -> str | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> str:
        t = self.peek()
        if t is None:
            raise BoundError("the expression ends too early")
        self.i += 1
        return t

    def sum(self) -> Node:
        terms = [self.product()]
        while self.peek() == "+":
            self.take()
            terms.append(self.product())
        return terms[0] if len(terms) == 1 else Op("+", tuple(terms))

    def product(self) -> Node:
        factors = [self.power()]
        while self.peek() in ("*", "/"):
            op = self.take()
            f = self.power()
            if op == "/":
                if not isinstance(f, Num) or f.value == 0:
                    raise BoundError("divide only by a positive number")
                f = Num(1 / f.value)
            factors.append(f)
        return factors[0] if len(factors) == 1 else Op("*", tuple(factors))

    def power(self) -> Node:
        base = self.atom()
        if self.peek() in ("^", "**"):
            self.take()
            k = self.take()
            if not k.isdigit() or not 1 <= int(k) <= MAX_POWER:
                raise BoundError(f"a power is an integer from 1 to {MAX_POWER}, not {k}")
            return Op("^", (base, int(k)))
        return base

    def atom(self) -> Node:
        t = self.take()
        if t == "(":
            node = self.sum()
            if self.take() != ")":
                raise BoundError("a ( is not closed")
            return node
        if re.fullmatch(r"\d+(?:\.\d+)?", t):
            return Num(Fraction(t))
        if t in self.atoms:
            return Atom(t)
        raise BoundError(f"unknown name {t!r}; the atoms are {' '.join(self.atoms)}")


def _count(node: Node) -> int:
    if isinstance(node, Op):
        return 1 + sum(_count(a) for a in node.args if not isinstance(a, int))
    return 1


@dataclasses.dataclass(frozen=True)
class Bound:
    text: str
    node: Node
    problem: str = ""

    def value(self, env: dict[str, float]) -> float:
        return _value(self.node, env)

    def lean(self) -> str:
        return _lean(self.node, {**ATOMS, **EXTRA.get(self.problem, {})})

    def atoms(self) -> set[str]:
        out: set[str] = set()
        _atoms(self.node, out)
        return out


def atoms_of(problem: str = "") -> dict[str, str]:
    return {**ATOMS, **EXTRA.get(problem, {})}


def parse(text: str, problem: str = "") -> Bound:
    text = text.strip().strip("`").strip()
    if not text:
        raise BoundError("an empty bound")
    if len(text) > MAX_LENGTH:
        raise BoundError(f"a bound has at most {MAX_LENGTH} characters")
    p = _Parser(_tokens(text), atoms_of(problem))
    node = p.sum()
    if p.peek() is not None:
        raise BoundError(f"unexpected {p.peek()!r} after a complete expression")
    if _count(node) > MAX_NODES:
        raise BoundError(f"a bound has at most {MAX_NODES} operations and atoms")
    return Bound(text, node, problem)


def _value(node: Node, env: dict[str, float]) -> float:
    if isinstance(node, Num):
        return float(node.value)
    if isinstance(node, Atom):
        return float(env[node.name])
    if node.op == "+":
        return math.fsum(_value(a, env) for a in node.args)
    if node.op == "*":
        out = 1.0
        for a in node.args:
            out *= _value(a, env)
        return out
    base, k = node.args
    return _value(base, env) ** k


def _lean(node: Node, atoms: dict[str, str]) -> str:
    if isinstance(node, Num):
        v = node.value
        return (
            f"({v.numerator} : ℝ)"
            if v.denominator == 1
            else f"(({v.numerator} : ℝ) / {v.denominator})"
        )
    if isinstance(node, Atom):
        return atoms[node.name]
    if node.op == "+":
        return "(" + " + ".join(_lean(a, atoms) for a in node.args) + ")"
    if node.op == "*":
        return "(" + " * ".join(_lean(a, atoms) for a in node.args) + ")"
    base, k = node.args
    return f"({_lean(base, atoms)} ^ {k})"


def _atoms(node: Node, out: set[str]) -> None:
    if isinstance(node, Atom):
        out.add(node.name)
    elif isinstance(node, Op):
        for a in node.args:
            if not isinstance(a, int):
                _atoms(a, out)
