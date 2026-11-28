"""The IR of the search: a program is a sequence of blocked folds over a stream of elements.

A program reads as one S-expression::

    (program NAME
      (input FIELD ...)                     ; the fields of one element: x, or s and v
      (pass NAME                            ; one pass over the elements, one or more
        (state FIELD ...)                   ; what a block reduces to
        (block (FIELD EXPR) ...)            ; the summary of one nonempty block
        (merge (FIELD EXPR) ...))           ; two adjacent summaries, L then R, into one
      (output elementwise EXPR)             ; one value per element, or
      (output final (NAME EXPR) ...))       ; named values from the finished passes

A ``block`` and a ``merge`` bind every state field once, in any order. A binding whose name is
not a state field is a temporary: later bindings of the same list can read it, and it is not
part of the summary.

Expressions are numbers, names and calls. Names are:

* an element field (``x``), inside a reduction of a ``block`` or in an elementwise output;
* a name bound earlier in the same ``block`` or ``merge`` list (``m``);
* ``L.f`` and ``R.f`` in a ``merge``: state field ``f`` of the left and of the right summary;
* ``p.f``: state field ``f`` of the finished pass ``p``, any pass before the one being defined.

Calls: ``+ - * /`` (``+`` and ``*`` take two or more arguments, ``-`` and ``/`` two),
``neg exp log sqrt abs`` (one), ``max min`` (two), and the reductions over the elements of the
block, ``sum rmax rmin`` (one argument; anywhere in a ``block`` binding, but not inside another
reduction).

Values have a type, ``scalar`` or ``vec`` (a vector of the problem's dimension ``D``); the
arithmetic operators broadcast a scalar over a vector, ``rmax``, ``rmin``, ``max``, ``min`` and
``log``/``sqrt`` take scalars, ``sum`` takes either.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
from collections.abc import Iterator

# --------------------------------------------------------------------------- reader

_TOKEN = re.compile(r"\s+|;[^\n]*|\(|\)|[^\s();]+")


class IRError(Exception):
    """A program that does not parse or does not type. The message is meant for the proposer."""


def _tokens(text: str) -> Iterator[str]:
    for match in _TOKEN.finditer(text):
        tok = match.group(0)
        if tok[0].isspace() or tok[0] == ";":
            continue
        yield tok


def read(text: str) -> list:
    """Parse ``text`` into nested lists of atoms (str or float). Exactly one top-level form."""
    stack: list[list] = [[]]
    for tok in _tokens(text):
        if tok == "(":
            stack.append([])
        elif tok == ")":
            if len(stack) == 1:
                raise IRError("unbalanced ')'")
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(_atom(tok))
    if len(stack) != 1:
        raise IRError(f"{len(stack) - 1} unclosed '('")
    forms = stack[0]
    if len(forms) != 1:
        raise IRError(f"expected one top-level form, found {len(forms)}")
    return forms[0]


def _atom(tok: str) -> str | float:
    """A number, or else a name. ``inf``, ``nan`` and ``1e999`` stay names, so that the parser
    rejects them: the programs are read over the reals."""
    try:
        value = float(tok)
    except ValueError:
        return tok
    return value if math.isfinite(value) else tok


# --------------------------------------------------------------------------- AST


@dataclasses.dataclass(frozen=True)
class Num:
    value: float


@dataclasses.dataclass(frozen=True)
class Var:
    name: str  # x, m, L.m, R.m, a.m


@dataclasses.dataclass(frozen=True)
class Call:
    op: str
    args: tuple[Expr, ...]


@dataclasses.dataclass(frozen=True)
class Reduce:
    op: str  # sum | rmax | rmin
    body: Expr


Expr = Num | Var | Call | Reduce

UNARY = {"neg", "exp", "log", "sqrt", "abs"}
BINARY = {"-", "/", "max", "min"}
VARIADIC = {"+", "*"}
REDUCTIONS = {"sum", "rmax", "rmin"}
SCALAR_ONLY = {"max", "min", "log", "sqrt"}


@dataclasses.dataclass
class Pass:
    name: str
    fields: list[str]
    block: list[tuple[str, Expr]]
    merge: list[tuple[str, Expr]]


@dataclasses.dataclass
class Output:
    kind: str  # elementwise | final
    exprs: list[tuple[str, Expr]]  # elementwise: one entry named "y"


@dataclasses.dataclass
class Program:
    name: str
    inputs: list[str]
    passes: list[Pass]
    output: Output
    types: dict[str, str] = dataclasses.field(default_factory=dict)  # "p.f" -> scalar | vec

    def text(self) -> str:
        return unparse(self)

    def digest(self) -> str:
        """Identity of the program up to its name and layout."""
        body = unparse(dataclasses.replace(self, name="_"))
        return hashlib.sha256(body.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- parsing


def _expect_list(form, what: str) -> list:
    if not isinstance(form, list) or not form:
        raise IRError(f"expected ({what} ...), found {_show(form)}")
    return form


def _show(form) -> str:
    if isinstance(form, list):
        return "(" + " ".join(_show(f) for f in form) + ")"
    if isinstance(form, float):
        return _num(form)
    return str(form)


def _num(v: float) -> str:
    if not math.isfinite(v):
        return repr(v)
    return str(int(v)) if v == int(v) and abs(v) < 1e15 else repr(v)


def parse_expr(form) -> Expr:
    if isinstance(form, float):
        return Num(form)
    if isinstance(form, str):
        if form in ("inf", "+inf", "-inf", "nan"):
            raise IRError(f"`{form}` is not a real number; the programs are read over the reals")
        return Var(form)
    if not form:
        raise IRError("empty call ()")
    head, *rest = form
    if not isinstance(head, str):
        raise IRError(f"a call starts with an operator, found {_show(form)}")
    args = tuple(parse_expr(a) for a in rest)
    if head in REDUCTIONS:
        if len(args) != 1:
            raise IRError(f"({head} ...) takes one argument: {_show(form)}")
        return Reduce(head, args[0])
    if head in UNARY and len(args) != 1:
        raise IRError(f"({head} ...) takes one argument: {_show(form)}")
    if head in BINARY and len(args) != 2:
        raise IRError(f"({head} ...) takes two arguments: {_show(form)}")
    if head in VARIADIC and len(args) < 2:
        raise IRError(f"({head} ...) takes two or more arguments: {_show(form)}")
    if head not in UNARY | BINARY | VARIADIC:
        known = ", ".join(sorted(UNARY | BINARY | VARIADIC | REDUCTIONS))
        raise IRError(f"unknown operator `{head}` (known: {known})")
    return Call(head, args)


def _bindings(form, what: str) -> list[tuple[str, Expr]]:
    out = []
    for item in form[1:]:
        if not (isinstance(item, list) and len(item) == 2 and isinstance(item[0], str)):
            raise IRError(f"each entry of ({what} ...) is (NAME EXPR), found {_show(item)}")
        out.append((item[0], parse_expr(item[1])))
    return out


def parse(text: str) -> Program:
    """Parse and type a program. Raises :class:`IRError` with a message for the proposer."""
    form = _expect_list(read(text), "program")
    if form[0] != "program" or len(form) < 4 or not isinstance(form[1], str):
        raise IRError("a program is (program NAME (input ...) (pass ...)... (output ...))")
    name = form[1]
    inp = _expect_list(form[2], "input")
    if inp[0] != "input" or not all(isinstance(f, str) for f in inp[1:]) or len(inp) < 2:
        raise IRError("(input FIELD ...) names the fields of one element")
    passes: list[Pass] = []
    for item in form[3:-1]:
        p = _expect_list(item, "pass")
        if p[0] != "pass" or len(p) != 5 or not isinstance(p[1], str):
            raise IRError(f"a pass is (pass NAME (state ...) (block ...) (merge ...)): {_show(p)}")
        parts = zip(p[2:], ("state", "block", "merge"), strict=True)
        st, bl, mg = (_expect_list(x, w) for x, w in parts)
        if st[0] != "state" or bl[0] != "block" or mg[0] != "merge":
            raise IRError(
                f"pass {p[1]}: expected (state ...) (block ...) (merge ...) in that order"
            )
        if not all(isinstance(f, str) for f in st[1:]) or len(st) < 2:
            raise IRError(f"pass {p[1]}: (state FIELD ...) names one field or more")
        passes.append(Pass(p[1], list(st[1:]), _bindings(bl, "block"), _bindings(mg, "merge")))
    if not passes:
        raise IRError("a program has one pass or more")
    out = _expect_list(form[-1], "output")
    if out[0] != "output" or len(out) < 3 or out[1] not in ("elementwise", "final"):
        raise IRError(
            "the last form is (output elementwise EXPR) or (output final (NAME EXPR) ...)"
        )
    if out[1] == "elementwise":
        if len(out) != 3:
            raise IRError("(output elementwise EXPR) has one expression")
        output = Output("elementwise", [("y", parse_expr(out[2]))])
    else:
        output = Output("final", _bindings(["final", *out[2:]], "output final"))
    prog = Program(name, list(inp[1:]), passes, output)
    check(prog)
    return prog


# --------------------------------------------------------------------------- typing


def check(prog: Program, elem_types: dict[str, str] | None = None) -> None:
    """Check scoping and types; fill ``prog.types``. ``elem_types`` maps the element fields to
    their types (default: all scalar except a field named ``v``)."""
    elem = elem_types or {f: ("vec" if f == "v" else "scalar") for f in prog.inputs}
    for f in prog.inputs:
        if f not in elem:
            raise IRError(f"input field `{f}` is not a field of this problem's elements")
    reserved = set(prog.inputs) | {"L", "R"}
    seen_pass: set[str] = set()
    finished: dict[str, str] = {}  # "p.f" -> type
    for p in prog.passes:
        if p.name in reserved or p.name in seen_pass or "." in p.name:
            raise IRError(f"pass name `{p.name}` is taken or invalid")
        if len(set(p.fields)) != len(p.fields):
            raise IRError(f"pass {p.name}: a state field is listed twice")
        for f in p.fields:
            if f in reserved or "." in f:
                raise IRError(f"pass {p.name}: `{f}` cannot be a state field")
        for what, binds in (("block", p.block), ("merge", p.merge)):
            names = [n for n, _ in binds]
            if len(set(names)) != len(names):
                raise IRError(f"pass {p.name}: ({what} ...) binds a name twice ({' '.join(names)})")
            missing = [f for f in p.fields if f not in names]
            if missing:
                raise IRError(
                    f"pass {p.name}: ({what} ...) must bind every state field "
                    f"({' '.join(p.fields)}); it does not bind {' '.join(missing)}"
                )
            for n in names:
                if n not in p.fields and (n in reserved or "." in n or n in seen_pass):
                    raise IRError(f"pass {p.name}: `{n}` cannot be the name of a temporary")
        local: dict[str, str] = {}
        for n, e in p.block:
            local[n] = _type(e, "block", elem, local, finished, None, p)
        state = {f: local[f] for f in p.fields}
        merged: dict[str, str] = {}
        sides = {f"{s}.{f}": t for s in "LR" for f, t in state.items()}
        for n, e in p.merge:
            t = _type(e, "merge", elem, merged, finished, sides, p)
            if n in state and t != state[n]:
                raise IRError(f"pass {p.name}: merge gives `{n}` type {t}, block gives {state[n]}")
            merged[n] = t
        for f, t in state.items():
            finished[f"{p.name}.{f}"] = t
        seen_pass.add(p.name)
    for _, e in prog.output.exprs:
        where = "elementwise" if prog.output.kind == "elementwise" else "final"
        _type(e, where, elem, {}, finished, None, None)
    prog.types = finished


def _type(
    e: Expr,
    where: str,
    elem: dict[str, str],
    local: dict[str, str],
    finished: dict[str, str],
    sides: dict[str, str] | None,
    p: Pass | None,
    in_reduce: bool = False,
) -> str:
    """The type of ``e`` in a ``where`` context: block, merge, elementwise or final."""
    ctx = f"pass {p.name} {where}" if p is not None else f"output {where}"
    if isinstance(e, Num):
        return "scalar"
    if isinstance(e, Var):
        n = e.name
        if n in elem:
            if (where == "block" and in_reduce) or where == "elementwise":
                return elem[n]
            raise IRError(
                f"{ctx}: element field `{n}` can be read only inside (sum ...), (rmax ...) or "
                f"(rmin ...) of a block, or in an elementwise output"
            )
        if n in local:
            return local[n]
        if sides is not None and n in sides:
            return sides[n]
        if n in finished:
            return finished[n]
        if "." in n and n.split(".", 1)[0] in ("L", "R") and sides is None:
            raise IRError(f"{ctx}: `{n}` exists only inside a merge")
        raise IRError(f"{ctx}: unknown name `{n}`")
    if isinstance(e, Reduce):
        if where != "block" or in_reduce:
            raise IRError(
                f"{ctx}: ({e.op} ...) is allowed only in a block binding, "
                "not inside another reduction"
            )
        t = _type(e.body, where, elem, local, finished, sides, p, True)
        if e.op != "sum" and t != "scalar":
            raise IRError(f"{ctx}: ({e.op} ...) takes a scalar per element")
        return t
    assert isinstance(e, Call)
    ts = [_type(a, where, elem, local, finished, sides, p, in_reduce) for a in e.args]
    if e.op in SCALAR_ONLY and any(t != "scalar" for t in ts):
        raise IRError(f"{ctx}: ({e.op} ...) takes scalars")
    return "vec" if "vec" in ts else "scalar"


# --------------------------------------------------------------------------- printing


def unparse_expr(e: Expr) -> str:
    if isinstance(e, Num):
        return _num(e.value)
    if isinstance(e, Var):
        return e.name
    if isinstance(e, Reduce):
        return f"({e.op} {unparse_expr(e.body)})"
    return "(" + " ".join([e.op, *(unparse_expr(a) for a in e.args)]) + ")"


def unparse(prog: Program) -> str:
    lines = [f"(program {prog.name}", f"  (input {' '.join(prog.inputs)})"]
    for p in prog.passes:
        lines.append(f"  (pass {p.name}")
        lines.append(f"    (state {' '.join(p.fields)})")
        lines.append(
            "    (block "
            + "\n           ".join(f"({n} {unparse_expr(e)})" for n, e in p.block)
            + ")"
        )
        lines.append(
            "    (merge "
            + "\n           ".join(f"({n} {unparse_expr(e)})" for n, e in p.merge)
            + "))"
        )
    if prog.output.kind == "elementwise":
        lines.append(f"  (output elementwise {unparse_expr(prog.output.exprs[0][1])}))")
    else:
        outs = " ".join(f"({n} {unparse_expr(e)})" for n, e in prog.output.exprs)
        lines.append(f"  (output final {outs}))")
    return "\n".join(lines)


def walk(e: Expr) -> Iterator[Expr]:
    yield e
    if isinstance(e, Call):
        for a in e.args:
            yield from walk(a)
    elif isinstance(e, Reduce):
        yield from walk(e.body)


def size(prog: Program) -> int:
    """Number of expression nodes, a measure of the program's complexity."""
    n = 0
    for p in prog.passes:
        for _, e in [*p.block, *p.merge]:
            n += sum(1 for _ in walk(e))
    for _, e in prog.output.exprs:
        n += sum(1 for _ in walk(e))
    return n


def reads_elements(prog: Program) -> list[set[str]]:
    """For each pass over memory, the element fields it reads: one entry per pass, then one for
    an elementwise output."""
    out = []
    for p in prog.passes:
        out.append(
            {
                e.name
                for _, b in p.block
                for e in walk(b)
                if isinstance(e, Var) and e.name in prog.inputs
            }
        )
    if prog.output.kind == "elementwise":
        out.append(
            {
                e.name
                for _, b in prog.output.exprs
                for e in walk(b)
                if isinstance(e, Var) and e.name in prog.inputs
            }
        )
    return out
