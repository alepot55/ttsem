"""The Lean reading of a program: its passes as definitions over ℝ, and the theorems that make it
correct, each proof left as a slot for the prover.

``export`` writes one Lean file per candidate, in the namespace ``Discovery.Library.<Module>``:

* per pass ``p``: a structure ``State_p`` (its state), ``block_p`` (the summary of one block, a
  list of elements; the finished passes it reads are parameters) and ``merge_p``;
* ``output``: the candidate with every pass reading all of ``X`` as one block; ``Spec.output``,
  the same for the specification (the problem's seed program unless another is given);
* ``blocked``: the candidate as a kernel runs it, every pass folding its blocks along a merge tree
  of its own;
* the obligations, each a theorem whose proof is a slot: per pass ``merge_block_p``, merging the
  summaries of two nonempty blocks gives the summary of both, and ``spec_eq``, on one block the
  candidate's output is the specification's;
* ``blocked_eq_spec``, proved here from the obligations and the framework theorem of
  ``Discovery/Blocked.lean``: over any blocking and any merge trees, the candidate computes the
  specification.

The element type is ℝ for one scalar field and a product for several (``ℝ × V`` for attention,
``V`` any real vector space). The statements come from the IR alone. Each one is also fixed as a
``Prop`` (``X_statement``) before the first slot and checked against its theorem after the last
one (``theorem X_holds : X_statement := X``). The file is one text for the prover, and two modules
for the judge (``prove.check``): ``statements``, everything before the first slot, compiled from
the generated text alone, and ``proofs``, the rest, which imports it. The judge's checker then
reads, outside any elaboration of the model's text, that every ``X_holds`` has exactly the type
``X_statement`` of the first module.
"""

from __future__ import annotations

import dataclasses
import math
import re
import textwrap
from decimal import Decimal
from pathlib import Path

from discovery import ir
from discovery.problems import Problem

LEAN_DIR = Path(__file__).resolve().parent / "lean"
FRAMEWORK = LEAN_DIR / "Discovery" / "Blocked.lean"

IMPORTS = (
    "Discovery.Blocked",
    "Mathlib.Algebra.BigOperators.Ring.List",
    "Mathlib.Algebra.Module.BigOperators",
    "Mathlib.Analysis.Complex.Exponential",
    "Mathlib.Analysis.Real.Sqrt",
    "Mathlib.Analysis.SpecialFunctions.Log.Basic",
    "Mathlib.Tactic.FieldSimp",
    "Mathlib.Tactic.LinearCombination",
    "Mathlib.Tactic.Linarith",
    "Mathlib.Tactic.Module",
    "Mathlib.Tactic.Positivity",
    "Mathlib.Tactic.Ring",
)

HELPERS = "helpers"  # the slot for the prover's helper lemmas
MAX_TERM = 4000  # characters of one field's expression, once its bound fields are written out


class ExportError(Exception):
    """A program the Lean reading cannot state; the message names the construct."""


# --------------------------------------------------------------------------- names

_KEYWORDS = {
    "at", "by", "calc", "class", "def", "deriving", "do", "else", "end", "example", "exists",
    "from", "fun", "have", "if", "import", "in", "inductive", "instance", "let", "local", "match",
    "mutual", "namespace", "open", "private", "protected", "section", "show", "structure",
    "suffices", "then", "theorem", "universe", "variable", "where", "with", "Prop", "Sort", "Type",
}  # fmt: skip
# the names Lean gives a structure's own declarations
_STRUCTURE = {"mk", "rec", "recOn", "casesOn", "noConfusion", "noConfusionType", "ext", "ext_iff"}
# what the generated text refers to by name, so no binder may take it
_GLOBAL = {
    "L", "R", "V", "Real", "List", "Discovery", "Tree", "Spec", "Out", "output", "blocked", "max",
    "min", "lmax", "lmin",
}  # fmt: skip
_SIMPLE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*\Z")


def ident(name: str) -> str:
    """``name`` as a Lean identifier: as it is when it is one, else between « »."""
    if _SIMPLE.match(name) and name not in _KEYWORDS:
        return name
    if "«" in name or "»" in name:
        raise ExportError(f"the name `{name}` cannot be written in Lean")
    return f"«{name}»"


def _field(name: str) -> str:
    return ident(name + "_" if name in _STRUCTURE else name)


class _Binders:
    """Lean names for the binders of one file: distinct from each other and from every name the
    file refers to."""

    def __init__(self) -> None:
        self.taken = _GLOBAL | _KEYWORDS

    def fresh(self, want: str) -> str:
        base = want if _SIMPLE.match(want) else "v"
        name, k = base, 1
        while name in self.taken or name.startswith(("State_", "block_", "merge_")):
            name, k = f"{base}{k}", k + 1
        self.taken.add(name)
        return name


def module_name(problem: Problem, prog: ir.Program) -> str:
    """``SoftmaxOnline`` for the program ``softmax-online`` of the problem ``softmax``."""
    rest = prog.name
    if rest.lower().startswith(problem.name.lower()):
        rest = rest[len(problem.name) :]
    words = [w for w in re.split(r"[^A-Za-z0-9]+", f"{problem.name} {rest}") if w]
    name = "".join(w[0].upper() + w[1:] for w in words)
    return name if name[:1].isalpha() else "P" + name


# --------------------------------------------------------------------------- expressions

_ATOM, _APP, _NEG, _SMUL, _MUL, _ADD = 1024, 1000, 75, 73, 70, 65


@dataclasses.dataclass(frozen=True)
class _Term:
    """A Lean expression, the precedence of its outermost operator, and whether it is a vector."""

    text: str
    prec: int
    vec: bool

    def at(self, prec: int) -> str:
        return self.text if self.prec >= prec else f"({self.text})"


@dataclasses.dataclass
class _Rows:
    """How a reduction ranges over a block: the list, the element binder and its fields."""

    list: str
    var: str
    fields: dict[str, _Term]
    whole: str | None  # the element field that is the whole element, when there is one


def _num(v: float) -> str:
    if not math.isfinite(v):
        raise ExportError(f"`{v!r}` is not a real number")
    text = str(int(v)) if v == int(v) and abs(v) < 1e15 else format(Decimal(repr(v)), "f")
    return f"({text} : ℝ)"


def _vector_error(what: str) -> ExportError:
    return ExportError(
        f"{what} has no meaning in an abstract real vector space; the Lean reading has vector + "
        "vector, vector - vector, scalar * vector, vector / scalar, neg and sum of vectors"
    )


def _add(a: _Term, b: _Term, op: str) -> _Term:
    if a.vec != b.vec:
        raise _vector_error(f"({op} ...) of a scalar and a vector (componentwise)")
    return _Term(f"{a.at(_ADD)} {op} {b.at(_ADD + 1)}", _ADD, a.vec)


def _mul(a: _Term, b: _Term) -> _Term:
    if a.vec and b.vec:
        raise _vector_error("(* ...) of two vectors (componentwise)")
    if not a.vec and not b.vec:
        return _Term(f"{a.at(_MUL)} * {b.at(_MUL + 1)}", _MUL, False)
    s, v = (b, a) if a.vec else (a, b)
    return _Term(f"{s.at(_SMUL + 1)} • {v.at(_SMUL)}", _SMUL, True)


def _div(a: _Term, b: _Term) -> _Term:
    if b.vec:
        raise _vector_error("(/ ...) by a vector")
    if a.vec:
        return _Term(f"{b.at(_ATOM)}⁻¹ • {a.at(_SMUL)}", _SMUL, True)
    return _Term(f"{a.at(_MUL)} / {b.at(_MUL + 1)}", _MUL, False)


def _term(e: ir.Expr, names: dict[str, _Term], rows: _Rows | None = None) -> _Term:
    """Translate ``e``; ``names`` gives the Lean term of every IR name in scope."""
    if isinstance(e, ir.Num):
        return _Term(_num(e.value), _ATOM, False)
    if isinstance(e, ir.Var):
        if e.name not in names:
            raise ExportError(f"`{e.name}` is not in scope here")
        return names[e.name]
    if isinstance(e, ir.Reduce):
        if rows is None:
            raise ExportError(f"({e.op} ...) outside a block")
        body = _term(e.body, {**names, **rows.fields})
        if rows.whole is not None and e.body == ir.Var(rows.whole):
            listed = _Term(rows.list, _ATOM, body.vec)
        else:
            uses = any(isinstance(n, ir.Var) and n.name in rows.fields for n in ir.walk(e.body))
            var = rows.var if uses else "_"
            listed = _Term(f"{rows.list}.map fun {var} => {body.text}", 0, body.vec)
        if e.op == "sum":
            return _Term(f"{listed.at(_ATOM)}.sum", _ATOM, body.vec)
        fn = {"rmax": "lmax", "rmin": "lmin"}[e.op]
        return _Term(f"{fn} {listed.at(_ATOM)}", _APP, False)
    assert isinstance(e, ir.Call)
    args = [_term(a, names, rows) for a in e.args]
    op = e.op
    if op in ("+", "*"):
        acc = args[0]
        for a in args[1:]:
            acc = _add(acc, a, "+") if op == "+" else _mul(acc, a)
        return acc
    if op == "-":
        return _add(args[0], args[1], "-")
    if op == "/":
        return _div(args[0], args[1])
    if op == "neg":
        return _Term(f"-{args[0].at(_ATOM)}", _NEG, args[0].vec)
    if args[0].vec:
        raise _vector_error(f"({op} ...) of a vector")
    if op in ("exp", "log", "sqrt"):
        return _Term(f"Real.{op} {args[0].at(_ATOM)}", _APP, False)
    if op == "abs":  # `||` would read as Boolean or
        inner = args[0].text
        return _Term(f"|({inner})|" if "|" in (inner[0], inner[-1]) else f"|{inner}|", _ATOM, False)
    return _Term(f"{op} {args[0].at(_ATOM)} {args[1].at(_ATOM)}", _APP, False)  # max, min


# --------------------------------------------------------------------------- programs


@dataclasses.dataclass
class _Elements:
    """The element type of a problem and how each field of an element is read."""

    type: str
    fields: dict[str, bool]  # IR field -> is a vector

    def of(self, var: str) -> dict[str, _Term]:
        names = list(self.fields)
        if len(names) == 1:
            return {names[0]: _Term(var, _ATOM, self.fields[names[0]])}
        out = {}
        for i, f in enumerate(names):
            path = ".2" * i + (".1" if i < len(names) - 1 else "")
            out[f] = _Term(var + path, _ATOM, self.fields[f])
        return out


def _elements(problem: Problem) -> _Elements:
    fields = {f: t == "vec" for f, t in problem.elems.items()}
    typ = " × ".join("V" if v else "ℝ" for v in fields.values())
    if typ != problem.lean_elem:
        raise ExportError(
            f"problem {problem.name}: lean_elem {problem.lean_elem!r}, fields {typ!r}"
        )
    return _Elements(typ, fields)


def _reads(binds: list[tuple[str, ir.Expr]], passes: list[str]) -> list[str]:
    """The finished passes a list of bindings reads, in program order."""
    used = {
        n.name.split(".", 1)[0]
        for _, e in binds
        for n in ir.walk(e)
        if isinstance(n, ir.Var) and "." in n.name
    }
    return [p for p in passes if p in used]


def _doc(text: str) -> str:
    return text.replace("-/", "- /")


def _structure(fields: list[tuple[str, _Term]]) -> str:
    for f, t in fields:
        if len(t.text) > MAX_TERM:
            raise ExportError(
                f"the expression of `{f}` is too long once its fields are written out"
            )
    return "{ " + ",\n    ".join(f"{_field(f)} := {t.text}" for f, t in fields) + " }"


@dataclasses.dataclass
class _Pass:
    binder: str  # the Lean name of its finished state
    state: str  # the Lean type of its state
    block_reads: list[str]  # the finished passes its block reads
    merge_reads: list[str]  # the finished passes its merge reads
    reads: list[str]  # both, in program order


def _apply(fn: str, args: list[str]) -> str:
    return " ".join([fn, *args])


def _arg(text: str) -> str:
    """``text`` as an argument of an application."""
    return f"({text})" if " " in text else text


class _Writer:
    """The Lean definitions of one program: the candidate, or the specification under `Spec`."""

    def __init__(self, prog: ir.Program, el: _Elements) -> None:
        self.prog, self.el = prog, el
        self.names = _Binders()
        self.A, self.B, self.X = (self.names.fresh(n) for n in "ABX")
        self.vec = {k: t == "vec" for k, t in prog.types.items()}
        order = [p.name for p in prog.passes]
        self.passes: dict[str, _Pass] = {}
        for p in prog.passes:
            binder = self.names.fresh(p.name)
            has_vec = any(self.vec[f"{p.name}.{f}"] for f in p.fields)
            br, mr = _reads(p.block, order), _reads(p.merge, order)
            state = ident(f"State_{p.name}") + (" V" if has_vec else "")
            self.passes[p.name] = _Pass(binder, state, br, mr, [q for q in order if q in br + mr])
        self.item = self.names.fresh(next(iter(el.fields)) if len(el.fields) == 1 else "e")

    def finished(self, upto: str | None = None) -> dict[str, _Term]:
        """The Lean term of every field `p.f` of the passes before `upto`."""
        out = {}
        for p in self.prog.passes:
            if p.name == upto:
                break
            b = self.passes[p.name].binder
            for f in p.fields:
                out[f"{p.name}.{f}"] = _Term(f"{b}.{_field(f)}", _ATOM, self.vec[f"{p.name}.{f}"])
        return out

    def block(self, p: str) -> str:
        """`block_p` applied to the finished passes it reads."""
        return _apply(
            ident(f"block_{p}"), [self.passes[q].binder for q in self.passes[p].block_reads]
        )

    def merge(self, p: str) -> str:
        """`merge_p` applied to the finished passes it reads."""
        return _apply(
            ident(f"merge_{p}"), [self.passes[q].binder for q in self.passes[p].merge_reads]
        )

    def params(self, reads: list[str]) -> str:
        return "".join(f"({self.passes[q].binder} : {self.passes[q].state}) " for q in reads)

    def pass_defs(self, p: ir.Pass, merges: bool) -> list[str]:
        info = self.passes[p.name]
        vec_param = " (V : Type*)" if info.state.endswith(" V") else ""
        out = [
            f"/-- The state of pass `{p.name}`. -/",
            f"@[ext] structure {ident('State_' + p.name)}{vec_param} where",
            *(f"  {_field(f)} : {'V' if self.vec[f'{p.name}.{f}'] else 'ℝ'}" for f in p.fields),
        ]
        whole = next(iter(self.el.fields)) if len(self.el.fields) == 1 else None
        rows = _Rows(self.A, self.item, self.el.of(self.item), whole)
        names = self.finished(p.name)
        local: dict[str, _Term] = {}
        for f, e in p.block:
            local[f] = _term(e, {**names, **local}, rows)
        ir_text = " ".join(f"({f} {ir.unparse_expr(e)})" for f, e in p.block)
        out += [
            "",
            f"/-- Pass `{p.name}` on one block `{self.A}`: {_doc(ir_text)} -/",
            f"noncomputable def {ident('block_' + p.name)} {self.params(info.block_reads)}"
            f"({self.A} : List {_arg(self.el.type)}) : {info.state} :=",
            "  " + _structure([(f, local[f]) for f in p.fields]),
        ]
        if not merges:
            return out
        sides = {
            f"{s}.{f}": _Term(f"{s}.{_field(f)}", _ATOM, self.vec[f"{p.name}.{f}"])
            for s in "LR"
            for f in p.fields
        }
        local = {}
        for f, e in p.merge:
            local[f] = _term(e, {**names, **sides, **local})
        ir_text = " ".join(f"({f} {ir.unparse_expr(e)})" for f, e in p.merge)
        return out + [
            "",
            f"/-- Pass `{p.name}`, two adjacent summaries `L` then `R` into one: "
            f"{_doc(ir_text)} -/",
            f"noncomputable def {ident('merge_' + p.name)} {self.params(info.merge_reads)}"
            f"(L R : {info.state}) : {info.state} :=",
            "  " + _structure([(f, local[f]) for f in p.fields]),
        ]

    def element(self) -> str:
        """The binder of the element an elementwise output is computed for, or ''."""
        if self.prog.output.kind != "elementwise":
            return ""
        return f" ({self.item} : {self.el.type})"

    def value(self) -> _Term:
        """The output from the finished passes (and the element, for an elementwise output)."""
        o = self.prog.output
        if o.kind == "elementwise":
            return _term(o.exprs[0][1], {**self.finished(), **self.el.of(self.item)})
        terms = [(n, _term(e, self.finished())) for n, e in o.exprs]
        return _Term(_structure(terms), _ATOM, any(t.vec for _, t in terms))

    def output(self, name: str, binders: str, out_type: str, values: dict[str, str]) -> list[str]:
        """A definition of the output in which pass `p` is bound to `values[p]`."""
        return [
            f"noncomputable def {name} {binders}{self.element()} : {out_type} :=",
            *(f"  let {self.passes[p].binder} := {values[p]}" for p in self.passes),
            f"  {self.value().text}",
        ]


def _out_type(problem: Problem, cand: _Writer) -> tuple[str, list[str]]:
    """The type of the output and, for named outputs, the structure that holds them."""
    if problem.outputs == "elementwise":
        return ("V" if cand.value().vec else "ℝ"), []
    assert isinstance(problem.outputs, dict)
    has_vec = "vec" in problem.outputs.values()
    return ("Out V" if has_vec else "Out"), [
        "/-- The named outputs. -/",
        "@[ext] structure Out" + (" (V : Type*) where" if has_vec else " where"),
        *(f"  {_field(n)} : {'V' if t == 'vec' else 'ℝ'}" for n, t in problem.outputs.items()),
        "",
    ]


# --------------------------------------------------------------------------- the file


@dataclasses.dataclass(frozen=True)
class _Binder:
    names: str  # one name, or several separated by spaces
    type: str
    hyp: bool = False  # a hypothesis: named in a theorem, anonymous in a statement


def _named(binders: list[_Binder]) -> str:
    return " ".join(f"({b.names} : {b.type})" for b in binders)


def _quantified(binders: list[_Binder]) -> str:
    """The binders as the prefix of a proposition: `∀ (A B : List ℝ), A ≠ [] → B ≠ [] →`."""
    out: list[str] = []
    group: list[str] = []
    for b in binders:
        if not b.hyp:
            group.append(f"({b.names} : {b.type})")
            continue
        if group:
            out.append(f"∀ {' '.join(group)},")
            group = []
        out.append(f"({b.type}) →" if b.type.startswith("∀") else f"{b.type} →")
    if group:
        out.append(f"∀ {' '.join(group)},")
    return " ".join(out)


@dataclasses.dataclass
class Obligation:
    name: str
    statement: str  # the theorem's header, without its proof
    meaning: str  # what it says, for the prover


@dataclasses.dataclass(frozen=True)
class _Slot:
    name: str
    indent: int


@dataclasses.dataclass
class LeanFile:
    module: str  # the file's place in the library: Discovery/Library/<module>.lean
    namespace: str
    program: str  # the candidate's IR text
    spec: str  # the specification's program name
    digest: str
    imports: tuple[str, ...]
    parts: list[str | _Slot]
    obligations: list[Obligation]
    composed: str  # the statement of blocked_eq_spec, proved here from the obligations
    checks: dict[str, str]  # theorem whose axioms are printed -> the theorem it checks
    split: int = 1  # parts[:split]: the definitions and statements, before any slot
    reopen: str = ""  # what parts[split:] needs in scope once in a module of its own

    @property
    def header(self) -> str:
        return "".join(f"import {m}\n" for m in self.imports)

    @property
    def prefix(self) -> str:
        """The text before the first slot: imports, definitions and statements."""
        parts = self.parts[: self.split]
        assert all(isinstance(p, str) for p in parts)
        return self.header + "".join(str(p) for p in parts)

    @property
    def statements(self) -> str:
        """The definitions and statements as a module of their own: no slot, no model text."""
        return f"{self.prefix}\nend {self.namespace}\n"

    def proofs(self, text: str, statements: str) -> tuple[str, int]:
        """The spliced ``text`` after its prefix, as a module importing the module
        ``statements``, and what to add to a line number of that module to get the same line of
        ``text``."""
        if not text.startswith(self.prefix):
            raise ValueError("the text does not start with the generated definitions")
        preamble = f"import {statements}\n{self.reopen}"
        offset = self.prefix.count("\n") - preamble.count("\n")
        return preamble + text[len(self.prefix) :], offset

    def slots(self, text: str) -> dict[str, str] | None:
        """What fills each slot of ``text``, a file spliced from this one (``HELPERS`` included),
        or None when ``text`` is not one."""
        pattern = re.escape(self.header)
        for part in self.parts:
            pattern += re.escape(part) if isinstance(part, str) else r"(.*?)\n"
        m = re.fullmatch(pattern, text, re.S)
        if m is None:
            return None
        names = [p.name for p in self.parts if isinstance(p, _Slot)]
        return dict(zip(names, m.groups(), strict=True))

    @property
    def text(self) -> str:
        """The file with ``sorry`` in every proof slot."""
        return self.splice({o.name: "sorry" for o in self.obligations})

    @property
    def body(self) -> str:
        """``text`` without its imports."""
        return self.text[len(self.header) :]

    def splice(self, proofs: dict[str, str], helpers: str = "") -> str:
        """The file with ``proofs[name]``, a tactic block, as the proof of each obligation, and
        ``helpers`` before the first theorem."""
        return self.layout(proofs, helpers)[0]

    def layout(
        self, proofs: dict[str, str], helpers: str = ""
    ) -> tuple[str, dict[str, tuple[int, int]]]:
        """The spliced text, and the first and last line (from 1) of every slot in it."""
        names = {o.name for o in self.obligations}
        if set(proofs) != names:
            raise ValueError(
                f"proofs missing for {sorted(names - set(proofs))}, "
                f"unknown proofs {sorted(set(proofs) - names)}"
            )
        text = self.header
        spans: dict[str, tuple[int, int]] = {}
        for part in self.parts:
            if isinstance(part, str):
                text += part
                continue
            raw = helpers if part.name == HELPERS else _tactics(proofs[part.name])
            filled = _indent(raw, part.indent)
            first = text.count("\n") + 1
            text += filled + "\n"
            spans[part.name] = (first, first + filled.count("\n"))
        return text, spans


def _tactics(body: str) -> str:
    """A proof body as a tactic block: dedented, without a leading `by`."""
    text = textwrap.dedent(body).strip("\n").rstrip()
    head = text.lstrip()
    if head == "by" or head.startswith(("by\n", "by ")):
        text = textwrap.dedent(head[2:].lstrip(" ").lstrip("\n")).rstrip()
    if not text.strip():
        raise ValueError("an empty proof")
    return text


def _indent(text: str, n: int) -> str:
    text = textwrap.dedent(text).strip("\n").rstrip()
    return "\n".join((" " * n + line) if line.strip() else "" for line in text.splitlines())


def export(
    prog: ir.Program, problem: Problem, spec: ir.Program | None = None, module: str | None = None
) -> LeanFile:
    """The Lean file of ``prog``, its obligations stated against ``spec`` (default: the seed)."""
    spec = spec or problem.parse(problem.seed)
    el = _elements(problem)
    module = module or module_name(problem, prog)
    namespace = f"Discovery.Library.{module}"
    cand, sw = _Writer(prog, el), _Writer(spec, el)
    out_type, out_lines = _out_type(problem, cand)
    has_vec = any(el.fields.values()) or out_type.endswith(" V")
    passes = [p.name for p in prog.passes]
    X, n = cand.X, cand.names
    hA, hB, hX = n.fresh("hA"), n.fresh("hB"), n.fresh("hX")
    trees = {p: n.fresh(f"t_{cand.passes[p].binder}") for p in passes}
    flat = {p: n.fresh(f"h_{cand.passes[p].binder}") for p in passes}
    leaves = {p: n.fresh(f"l_{cand.passes[p].binder}") for p in passes}
    blk, item = n.fresh("blk"), cand.element()
    arg = f" {cand.item}" if item else ""
    tree_type = f"Discovery.Tree {_arg(el.type)}"

    lines = [
        "",
        "/-!",
        f"# `{prog.name}` against `{spec.name}`",
        "",
        f"Generated by `discovery/lean_export.py` from the program below (digest {prog.digest()}).",
        "",
        "```",
        _doc(prog.text()),
        "```",
        "-/",
        "",
    ]
    # the scope every later declaration is read in, reopened by the proofs' own module
    scope = [f"namespace {namespace}", ""]
    if has_vec:
        scope += ["variable {V : Type*} [AddCommGroup V] [Module ℝ V]", ""]
    lines += [*scope, *out_lines, "/-! ## The candidate -/", ""]
    for p in prog.passes:
        lines += [*cand.pass_defs(p, merges=True), ""]
    lines += [
        "/-- The candidate's output, every pass reading all of `X` as one block. -/",
        *cand.output(
            "output",
            f"({X} : List {_arg(el.type)})",
            out_type,
            {p: f"{cand.block(p)} {X}" for p in passes},
        ),
        "",
        "/-- The candidate as a kernel runs it: every pass summarises the leaves of its own merge "
        "tree and merges the summaries along the tree. -/",
        *cand.output(
            "blocked",
            " ".join(f"({trees[p]} : {tree_type})" for p in passes),
            out_type,
            {p: f"{trees[p]}.eval {_arg(cand.block(p))} {_arg(cand.merge(p))}" for p in passes},
        ),
        "",
        f"/-! ## The specification: `{spec.name}`, every pass reading all of `X` as one block -/",
        "",
        "namespace Spec",
        "",
    ]
    for p in spec.passes:
        lines += [*sw.pass_defs(p, merges=False), ""]
    lines += [
        "/-- The specification's output. -/",
        *sw.output(
            "output",
            f"({sw.X} : List {_arg(el.type)})",
            out_type,
            {q: f"{sw.block(q)} {sw.X}" for q in sw.passes},
        ),
        "",
        "end Spec",
        "",
    ]

    elem = [_Binder(cand.item, el.type)] if item else []
    theorems: list[tuple[str, list[_Binder], str, str]] = []  # name, binders, claim, meaning
    for p in passes:
        info = cand.passes[p]
        binders = [_Binder(cand.passes[q].binder, cand.passes[q].state) for q in info.reads]
        binders += [_Binder(f"{cand.A} {cand.B}", f"List {_arg(el.type)}")]
        binders += [_Binder(hA, f"{cand.A} ≠ []", True), _Binder(hB, f"{cand.B} ≠ []", True)]
        theorems.append(
            (
                f"merge_block_{p}",
                binders,
                f"{cand.merge(p)} ({cand.block(p)} {cand.A}) ({cand.block(p)} {cand.B}) = "
                f"{cand.block(p)} ({cand.A} ++ {cand.B})",
                f"pass `{p}`: merging the summaries of two nonempty blocks gives the summary of "
                "the two blocks together"
                + (", whatever the finished passes it reads" if info.reads else ""),
            )
        )
    on_block = [_Binder(X, f"List {_arg(el.type)}"), _Binder(hX, f"{X} ≠ []", True)]
    theorems.append(
        (
            "spec_eq",
            on_block + elem,
            f"output {X}{arg} = Spec.output {X}{arg}",
            f"on one nonempty block, the candidate's output equals the specification's "
            f"(`{spec.name}`)" + (", for every element value" if item else ""),
        )
    )
    e2e_binders = list(on_block)
    for p in passes:
        t = trees[p]
        e2e_binders += [
            _Binder(t, tree_type),
            _Binder(flat[p], f"{t}.flat = {X}", True),
            _Binder(leaves[p], f"∀ {blk} ∈ {t}.leaves, {blk} ≠ []", True),
        ]
    e2e_binders += elem
    e2e_claim = f"blocked {' '.join(trees[p] for p in passes)}{arg} = Spec.output {X}{arg}"

    lines += ["/-! ## The statements, fixed before any proof text -/", ""]
    for name, binders, claim in [
        *(t[:3] for t in theorems),
        ("blocked_eq_spec", e2e_binders, e2e_claim),
    ]:
        lines += [f"def {ident(name + '_statement')} : Prop :=", f"  {_quantified(binders)}"]
        lines += [f"    {claim}", ""]

    obligations = [
        Obligation(name, f"theorem {ident(name)} {_named(bs)} :\n    {claim}", meaning)
        for name, bs, claim, meaning in theorems
    ]
    parts: list[str | _Slot] = ["\n".join(lines), "\n/-! ## The proofs -/\n\n-- helper lemmas\n"]
    parts += [_Slot(HELPERS, 0), "\n"]
    for o in obligations:
        parts += [f"/-- {o.meaning[0].upper()}{o.meaning[1:]}. -/\n{o.statement} := by\n"]
        parts += [_Slot(o.name, 2), "\n"]

    # blocked_eq_spec: each pass's tree evaluates to its summary of X, then spec_eq
    on_x: dict[str, str] = {}
    proof = []
    for i, p in enumerate(passes):
        info = cand.passes[p]
        blk_x = _apply(ident(f"block_{p}"), [f"({on_x[q]})" for q in info.block_reads])
        mrg_x = _apply(ident(f"merge_{p}"), [f"({on_x[q]})" for q in info.merge_reads])
        lemma = _apply(ident(f"merge_block_{p}"), [f"({on_x[q]})" for q in info.reads])
        on_x[p] = f"{blk_x} {X}"
        framework = f"Discovery.Tree.eval_eq_block {_arg(lemma)} {trees[p]} {leaves[p]}"
        proof += [
            f"  have e_{i} : {trees[p]}.eval {_arg(blk_x)} {_arg(mrg_x)} = {on_x[p]} := by",
            f"    rw [{framework}, {flat[p]}]",
        ]
    proof += [
        f"  rw [← spec_eq {X} {hX}{arg}]",
        f"  simp only [blocked, output, {', '.join(f'e_{i}' for i in range(len(passes)))}]",
    ]
    checks = {f"{o.name}_holds": o.name for o in obligations}
    checks["blocked_eq_spec_holds"] = "blocked_eq_spec"
    vec_arg = " (V := V)" if has_vec else ""
    tail = [
        "/-- Over any blocking and any merge trees, the candidate computes the specification: the "
        "framework theorem for every pass, then `spec_eq`. -/",
        f"theorem blocked_eq_spec {_named(e2e_binders)} :\n    {e2e_claim} := by",
        *proof,
        "",
        "/-! ## Checks: every theorem against its statement fixed above, and its axioms -/",
        "",
    ]
    for c, t in checks.items():
        tail.append(f"theorem {ident(c)} : {ident(t + '_statement')}{vec_arg} := {ident(t)}")
    tail += ["", *(f"#print axioms {ident(c)}" for c in checks), "", f"end {namespace}", ""]
    parts.append("\n".join(tail))
    return LeanFile(
        module=module,
        namespace=namespace,
        program=prog.text(),
        spec=spec.name,
        digest=prog.digest(),
        imports=IMPORTS,
        parts=parts,
        obligations=obligations,
        composed=f"theorem blocked_eq_spec {_named(e2e_binders)} :\n    {e2e_claim}",
        checks=checks,
        split=1,
        reopen="\n".join(scope) + "\n",
    )
