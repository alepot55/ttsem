"""Floating-point statements of a program: generated from its IR, the bound ``B`` left open.

For a one-pass program of the proof library, ``export`` writes a Lean file in the namespace
``Discovery.FP.<Module>`` that imports the program's library module (its exact semantics and the
merge lemma proved there) and ``Discovery.Approx`` (the rounding model and the depth law), and
states, for a bound ``B`` that the model proposes::

    ∀ F D lo hi X, X ≠ [] → (∀ x ∈ X, lo ≤ x ≤ hi) → small F →
      n (η + ηe) ≤ u → ∀ t, t.flat = X → (∀ b ∈ t.leaves, b ≠ []) →
      (d + D + 1)(u + ue + ul) ≤ 1/16 → ∀ s, FEval (fblock F D) (fmerge F) t s →
      |value s - spec X| ≤ B

* ``fblock`` and ``fmerge`` are the program's block and merge read operation by operation in the
  rounding model: every ``+ - * / exp log sqrt`` is one rounding (a relation, with its own error
  variables), ``max min neg abs`` are exact, and a block's ``sum`` adds along any tree of depth at
  most ``D`` (``fsum``). A count, ``(sum 1)`` merged by ``(+ L.n R.n)``, is exact: the statement
  then assumes ``n u ≤ 1``, under which floating-point sums of ones are exact;
* the certified quantity, per problem: the softmax's normalizer, read from the final state through
  the output expression at the element 0, as ``|log Ẑ − log Σ exp x| ≤ B``; one component of the
  attention's output, rounded as the output expression rounds it, as ``|ô − o| ≤ B`` with the
  scale ``S = Σ p|v| / Σ p`` and the bound ``V`` on ``|v|`` among the atoms of ``B`` (a purely
  relative bound is false); the variance, rounded likewise, as ``|v̂ − var| ≤ B``.

The statement's form is fixed here; the model gives only ``B``, an expression of the grammar of
:mod:`discovery.bound`, and the proofs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import textwrap

from discovery import bound, ir
from discovery.problems import Problem

IMPORTS = ("Discovery.Approx",)

# a small rounding: float32, float16 and bfloat16 all satisfy it, and second-order terms fit
SMALL = "F.u ≤ 1 / 256 ∧ F.ue ≤ 1 / 64 ∧ F.ul ≤ 1 / 64 ∧ F.η ≤ 1 / 2 ^ 100 ∧ F.ηe ≤ 1 / 2 ^ 100"

# The depth times the rounding is small, the classical hypothesis k·u < 1 of the γ_k bounds.
# Version 1 of the statement had no such hypothesis, and the prover refuted it (diary, 10:00 UTC):
# with u = 1/256 and a merge tree of depth 2000 a rounding relative to the state compounds as
# (1 + u)^d, which no bound linear in d covers. Every float32 run has (d + D + 1)(u + ue + ul)
# below 1e-4.
DEEP = "((t.depth : ℝ) + D + 1) * (F.u + F.ue + F.ul) ≤ 1 / 16"

# The absolute errors are small next to the relative ones. Version 2 had no such hypothesis,
# and the prover refuted it (diary, 10:00 UTC): with u = 0 and η = ηe = 2^-100, a comb of
# about 2^100 leaves accumulates enough absolute error to bring a normalizer to 0. float32
# satisfies it up to n = 2^124.
TINY = "(X.length : ℝ) * (F.η + F.ηe) ≤ F.u"
VERSION = 4

# Version 4 changes the attention's statement only. In a linear quantity (|ô − o|, not a log) the
# rounding of `s − m` inside an exponential is a factor e^(Ru), not a term Ru; with u = 1/256 and
# R = 2048 that is e^8, and the prover refuted version 3 with it (diary, 10:35 UTC). A range
# hypothesis keeps the factor near 1; float32 satisfies it for logits within a range of 10^6.
NARROW = "(hi - lo) * F.u ≤ 1 / 16"


class FPExportError(Exception):
    pass


@dataclasses.dataclass
class FPFile:
    module: str  # Discovery/FP/<module>.lean
    namespace: str
    program: str
    problem: str
    library: str  # the library module imported for the exact semantics
    definitions: str  # imports, namespace, generated definitions (no statement)
    statement_template: str  # the statement's proposition with `{B}` where the bound goes

    def statements(self, b: bound.Bound) -> str:
        """The statements module: definitions and `fp_bound_statement` for this ``B``."""
        prop = self.statement_template.replace("{B}", b.lean())
        return (
            f"{self.definitions}\n/-! ## The statement, fixed before any proof text -/\n\n"
            f"def fp_bound_statement : Prop :=\n{prop}\n\nend {self.namespace}\n"
        )

    def proofs(self, b: bound.Bound, helpers: str, proof: str, statements: str) -> str:
        """The proofs module: imports the statements module, the helpers, the theorem, the check."""
        prop = self.statement_template.replace("{B}", b.lean())
        body = _indent(_tactics(proof), 2)
        opened = f"Discovery Discovery.Library.{self.library}" if self.library else "Discovery"
        return (
            f"import {statements}\n\nnamespace {self.namespace}\n\n"
            f"open {opened}\n\n"
            f"{helpers.strip()}\n\n"
            f"theorem fp_bound : {prop.strip()} := by\n{body}\n\n"
            "theorem fp_bound_holds : fp_bound_statement := fp_bound\n\n"
            "#print axioms fp_bound_holds\n\n"
            f"end {self.namespace}\n"
        )

    def key(self, b: bound.Bound) -> str:
        return hashlib.sha256(self.statements(b).encode()).hexdigest()[:16]


def _tactics(body: str) -> str:
    text = textwrap.dedent(body).strip("\n").rstrip()
    head = text.lstrip()
    if head == "by" or head.startswith(("by\n", "by ")):
        text = textwrap.dedent(head[2:].lstrip(" ").lstrip("\n")).rstrip()
    if not text.strip():
        raise FPExportError("an empty proof")
    return text


def _indent(text: str, n: int) -> str:
    text = textwrap.dedent(text).strip("\n").rstrip()
    return "\n".join((" " * n + line) if line.strip() else "" for line in text.splitlines())


# --------------------------------------------------------------------------- expressions


def _num(v: float) -> str:
    if v == int(v):
        return f"({int(v)} : ℝ)"
    raise FPExportError(f"the constant {v} is not an integer; it is not exact in floating point")


def _is_count(e: ir.Expr) -> bool:
    return (
        isinstance(e, ir.Reduce)
        and e.op == "sum"
        and isinstance(e.body, ir.Num)
        and e.body.value == 1
    )


def _counts(p: ir.Pass) -> set[str]:
    """State fields that count elements: ``(sum 1)`` in the block, ``(+ L.f R.f)`` in the merge."""
    block, merge = dict(p.block), dict(p.merge)
    out = set()
    for f in p.fields:
        m = merge.get(f)
        if (
            _is_count(block.get(f))
            and isinstance(m, ir.Call)
            and m.op == "+"
            and len(m.args) == 2
            and {a.name for a in m.args if isinstance(a, ir.Var)} == {f"L.{f}", f"R.{f}"}
        ):
            out.add(f)
    return out


class _Rel:
    """Builds the relation of one block, merge or output: fresh variables and their conjuncts."""

    def __init__(
        self, prefix: str, elems: dict[str, str] | None = None, whole: bool = True
    ) -> None:
        self.prefix = prefix
        self.elems = elems  # IR element field -> Lean term over the element `x` (blocks only)
        self.whole = whole  # one element field: a reduction over it is over `A` itself
        self.count = 0
        self.vars: list[str] = []
        self.conj: list[str] = []

    def fresh(self) -> str:
        self.count += 1
        v = f"{self.prefix}{self.count}"
        self.vars.append(v)
        return v

    def reduce(self, e: ir.Reduce, env: dict[str, str]) -> str:
        """A fresh variable for a block's reduction, and its relation."""
        assert self.elems is not None
        v = self.fresh()
        if e.op in ("rmax", "rmin"):
            if not (isinstance(e.body, ir.Var) and e.body.name in self.elems):
                raise FPExportError("rmax/rmin of an expression is not supported")
            fn = "lmax" if e.op == "rmax" else "lmin"
            if self.whole:
                self.conj.append(f"{v} = {fn} A")
            else:
                field = self.elems[e.body.name].replace("x", "e")
                self.conj.append(f"{v} = {fn} (A.map fun e => {field})")
            return v
        if _is_count(e):
            self.conj.append(f"{v} = (A.length : ℝ)")  # exact: the statement assumes n u ≤ 1
            return v
        inner = _Rel("q")
        body = inner.term(e.body, {**env, **self.elems})
        parts = [*inner.conj, f"y = {body}"]
        ex = f"∃ {' '.join(inner.vars)}, " if inner.vars else ""
        self.conj.append(f"Discovery.fsum F D (fun x y => {ex}{' ∧ '.join(parts)}) A {v}")
        return v

    def term(self, e: ir.Expr, env: dict[str, str]) -> str:
        """A Lean term for the float value of ``e``, adding the roundings it takes."""
        if isinstance(e, ir.Num):
            return _num(e.value)
        if isinstance(e, ir.Var):
            if e.name not in env:
                raise FPExportError(f"unbound name {e.name}")
            return env[e.name]
        if isinstance(e, ir.Reduce):
            if self.elems is None:
                raise FPExportError("a reduction outside a block")
            return self.reduce(e, env)
        args = [self.term(a, env) for a in e.args]
        op = e.op
        if op == "ifgt":  # a comparison of float values and a selection: exact
            return f"(if {args[0]} > {args[1]} then {args[2]} else {args[3]})"
        if op in ("max", "min"):
            return f"({op} {args[0]} {args[1]})"
        if op == "neg":
            return f"(-{args[0]})"
        if op == "abs":
            return f"|{args[0]}|"
        if op in ("+", "*"):
            acc = args[0]
            for a in args[1:]:  # left to right, as the kernel folds it
                v = self.fresh()
                rel = "add" if op == "+" else "mul"
                self.conj.append(f"F.{rel} ({acc} {op} {a}) {v}")
                acc = v
            return acc
        if op in ("-", "/"):
            v = self.fresh()
            rel = "add" if op == "-" else "mul"
            self.conj.append(f"F.{rel} ({args[0]} {op} {args[1]}) {v}")
            return v
        if op in ("exp", "log", "sqrt"):
            v = self.fresh()
            self.conj.append(f"F.{op} {args[0]} {v}")
            return v
        raise FPExportError(f"operator {op} has no floating-point reading")


def _exact(e: ir.Expr, env: dict[str, str]) -> str:
    """The exact (real) Lean term of ``e``."""
    if isinstance(e, ir.Num):
        return _num(e.value)
    if isinstance(e, ir.Var):
        return env[e.name]
    if isinstance(e, ir.Reduce):
        raise FPExportError("a reduction in an output is not supported")
    a = [_exact(x, env) for x in e.args]
    if e.op == "ifgt":
        return f"(if {a[0]} > {a[1]} then {a[2]} else {a[3]})"
    if e.op in ("+", "*"):
        acc = a[0]
        for x in a[1:]:
            acc = f"({acc} {e.op} {x})"
        return acc
    if e.op in ("-", "/"):
        return f"({a[0]} {e.op} {a[1]})"
    if e.op in ("max", "min"):
        return f"({e.op} {a[0]} {a[1]})"
    if e.op == "neg":
        return f"(-{a[0]})"
    if e.op == "abs":
        return f"|{a[0]}|"
    fn = {"exp": "Real.exp", "log": "Real.log", "sqrt": "Real.sqrt"}[e.op]
    return f"({fn} {a[0]})"


# --------------------------------------------------------------------------- the file


def _close(rel: _Rel, binds: list[str], tail: str) -> str:
    rel.conj.append(tail)
    names = " ".join([*binds, *rel.vars])
    body = " ∧\n    ".join(rel.conj)
    return f"  ∃ {names},\n    {body}\n" if names else f"  {body}\n"


def _binders(params: tuple[str, ...]) -> str:
    return "".join(f" ({n} : ℝ)" for n in params)


def _block(
    p: ir.Pass, elems: dict[str, str], whole: bool, etype: str, stype: str, params=()
) -> str:
    """The relation `fblock_<p> F D A st`."""
    rel = _Rel("r", elems, whole)
    env: dict[str, str] = {n: n for n in params}
    binds: list[str] = []
    for name, e in p.block:
        v = f"v_{name}"
        binds.append(v)
        rel.conj.append(f"{v} = {rel.term(e, env)}")
        env[name] = v
    state = ", ".join(f"v_{f}" for f in p.fields)
    return (
        f"/-- Pass `{p.name}` on one block `A`, in floating point: `st` is one possible summary,\n"
        f"every in-block sum along a tree of depth at most `D`. -/\n"
        f"def fblock_{p.name} (F : Discovery.Rounding) (D : ℕ){_binders(params)} "
        f"(A : List ({etype})) (st : {stype}) "
        f": Prop :=\n" + _close(rel, binds, f"st = ⟨{state}⟩")
    )


def _merge(p: ir.Pass, counts: set[str], stype: str, params=()) -> str:
    rel = _Rel("r")
    env = {f"{s}.{f}": f"{s}.{f}" for s in "LR" for f in p.fields}
    env.update({n: n for n in params})
    binds: list[str] = []
    for name, e in p.merge:
        v = f"v_{name}"
        binds.append(v)
        if name in counts:
            rel.conj.append(f"{v} = L.{name} + R.{name}")  # a count: exact
        else:
            rel.conj.append(f"{v} = {rel.term(e, env)}")
        env[name] = v
    state = ", ".join(f"v_{f}" for f in p.fields)
    return (
        f"/-- Pass `{p.name}`, two adjacent float summaries `L` then `R` merged in floating\n"
        f"point: `st` is one possible result. -/\n"
        f"def fmerge_{p.name} (F : Discovery.Rounding){_binders(params)} (L R st : {stype}) "
        f": Prop :=\n" + _close(rel, binds, f"st = ⟨{state}⟩")
    )


def _fout(p: ir.Pass, name: str, e: ir.Expr, stype: str) -> str:
    """The relation `fout_<name> F st w`: `w` is one possible float value of the output."""
    rel = _Rel("r")
    env = {f"{p.name}.{f}": f"st.{f}" for f in p.fields}
    t = rel.term(e, env)
    return (
        f"/-- The output `{name}`, rounded as its expression rounds it: `w` is one possible\n"
        f"value. -/\n"
        f"def fout_{name} (F : Discovery.Rounding) (st : {stype}) (w : ℝ) : Prop :=\n"
        + _close(rel, [], f"w = {t}")
    )


_HEAD = """
/-!
# `{name}` in floating point

Generated by `discovery/fp_export.py` (statement version {version}) from the program below
(digest {digest}).

```
{text}
```
-/

namespace {namespace}

open Discovery Discovery.Library.{library}

"""

_SMALL_DEF = f"""
/-- The rounding is small: float32, float16 and bfloat16 all satisfy this. -/
def small (F : Discovery.Rounding) : Prop := {SMALL}
"""

_ATTN = """/-- The exact output (one component): `Σ exp s · v / Σ exp s`. -/
noncomputable def attn (X : List (ℝ × ℝ)) : ℝ :=
  (X.map fun e => Real.exp e.1 * e.2).sum / (X.map fun e => Real.exp e.1).sum

/-- The scale of the output's error: `Σ exp s · |v| / Σ exp s`. -/
noncomputable def attnScale (X : List (ℝ × ℝ)) : ℝ :=
  (X.map fun e => Real.exp e.1 * |e.2|).sum / (X.map fun e => Real.exp e.1).sum
"""

_VAR = """/-- The exact (population) variance of a row. -/
noncomputable def vr (X : List ℝ) : ℝ :=
  (X.map fun x => (x - X.sum / X.length) ^ 2).sum / X.length

/-- The exact mean of the squares of a row. -/
noncomputable def msq (X : List ℝ) : ℝ := (X.map fun x => x ^ 2).sum / X.length
"""


def export(prog: ir.Program, problem: Problem, library: str) -> FPFile:
    """The floating-point statement file of a one-pass ``prog``."""
    if len(prog.passes) != 1:
        raise FPExportError("only one-pass programs have a floating-point statement")
    p = prog.passes[0]
    counts = _counts(p)
    if problem.name == "attention":
        elems, whole, etype, stype = {"s": "x.1", "v": "x.2"}, False, "ℝ × ℝ", f"(State_{p.name} ℝ)"
    elif problem.name in ("softmax", "variance"):
        elems, whole, etype, stype = {prog.inputs[0]: "x"}, True, "ℝ", f"State_{p.name}"
    else:
        raise FPExportError(f"no floating-point statement for {problem.name}")
    namespace = f"Discovery.FP.{library}"
    parts = [
        "".join(f"import {m}\n" for m in (*IMPORTS, f"Discovery.Library.{library}")),
        _HEAD.format(
            name=prog.name,
            version=VERSION,
            digest=prog.digest(),
            text=prog.text(),
            namespace=namespace,
            library=library,
        ),
        _block(p, elems, whole, etype, stype),
        "\n",
        _merge(p, counts, stype),
        "\n",
    ]
    lead = f"""  ∀ (F : Discovery.Rounding) (D : ℕ) (lo hi : ℝ) (X : List ({etype})), X ≠ [] →"""
    tree = f"""    ∀ (t : Discovery.Tree ({etype})), t.flat = X → (∀ b ∈ t.leaves, b ≠ []) →
    {DEEP} →
    ∀ (st : {stype}), Discovery.Tree.FEval (fblock_{p.name} F D) (fmerge_{p.name} F) t st →"""
    if problem.name == "softmax":
        out_expr = prog.output.exprs[0][1]
        env = {prog.inputs[0]: "(0 : ℝ)", **{f"{p.name}.{f}": f"st.{f}" for f in p.fields}}
        parts.append(
            "/-- The normalizer a state encodes: one over the output expression read at the "
            "element `0`. -/\n"
            f"noncomputable def znorm (st : {stype}) : ℝ := 1 / {_exact(out_expr, env)}\n"
        )
        template = f"""{lead}
    (∀ x ∈ X, lo ≤ x ∧ x ≤ hi) → small F → {TINY} →
{tree}
      |Real.log (znorm st) - Real.log ((X.map Real.exp).sum)| ≤ {{B}}"""
    elif problem.name == "attention":
        name, e = prog.output.exprs[0]
        parts += [_fout(p, name, e, stype), "\n", _ATTN]
        template = f"""{lead}
    (∀ e ∈ X, lo ≤ e.1 ∧ e.1 ≤ hi) → ∀ (Vmax : ℝ), (∀ e ∈ X, |e.2| ≤ Vmax) →
    small F → {TINY} → {NARROW} →
{tree}
    ∀ (w : ℝ), fout_{name} F st w →
      |w - attn X| ≤ {{B}}"""
    else:
        name, e = next((n, x) for n, x in prog.output.exprs if n == "var")
        parts += [_fout(p, name, e, stype), "\n", _VAR]
        template = f"""{lead}
    (∀ x ∈ X, lo ≤ x ∧ x ≤ hi) → small F → {TINY} → (X.length : ℝ) * F.u ≤ 1 →
{tree}
    ∀ (w : ℝ), fout_{name} F st w →
      |w - vr X| ≤ {{B}}"""
    parts.append(_SMALL_DEF)
    return FPFile(
        module=f"Discovery/FP/{library}.lean",
        namespace=namespace,
        program=prog.text(),
        problem=problem.name,
        library=library,
        definitions="".join(parts),
        statement_template=template,
    )


# --------------------------------------------------------------------------- policies (M4)

_RDEPTH = """/-- The right depth of a merge tree: a leaf `0`, a node `max (left) (right + 1)`.
A stale
reference can lag the maximum by `τ` at each right step. -/
def rdepth {E : Type*} : Discovery.Tree E → ℕ
  | .leaf _ => 0
  | .node l r => max (rdepth l) (rdepth r + 1)
"""


def _exact_red(e: ir.Expr, env: dict[str, str], elem: str) -> str:
    """The exact Lean term of a block expression, reductions included."""
    if isinstance(e, ir.Reduce):
        if e.op in ("rmax", "rmin"):
            if not (isinstance(e.body, ir.Var) and e.body.name == elem):
                raise FPExportError("rmax/rmin of an expression is not supported")
            return "(lmax A)" if e.op == "rmax" else "(lmin A)"
        return f"((A.map fun x => {_exact(e.body, {**env, elem: 'x'})}).sum)"
    if isinstance(e, ir.Call):
        args = [_exact_red(a, env, elem) for a in e.args]
        return _exact(
            ir.Call(e.op, tuple(ir.Var(f"__{i}") for i in range(len(args)))),
            {f"__{i}": a for i, a in enumerate(args)},
        )
    return _exact(e, env)


def _exact_defs(p: ir.Pass, elem: str, params: tuple[str, ...]) -> str:
    """The exact semantics of a one-pass softmax-like program, written out (no library module)."""
    fields = "\n".join(f"  {f} : ℝ" for f in p.fields)
    out = [
        f"/-- The state of pass `{p.name}`. -/\n@[ext] structure State_{p.name} where\n{fields}\n"
    ]
    env = {n: n for n in params}
    lets = []
    for name, e in p.block:
        lets.append(f"  let b_{name} := {_exact_red(e, env, elem)}")
        env[name] = f"b_{name}"
    state = ", ".join(f"b_{f}" for f in p.fields)
    out.append(
        f"/-- Pass `{p.name}` on one block, exactly. -/\n"
        f"noncomputable def block_{p.name}{_binders(params)} (A : List ℝ) : State_{p.name} :=\n"
        + "\n".join(lets)
        + f"\n  ⟨{state}⟩\n"
    )
    env = {f"{s}.{f}": f"{s}.{f}" for s in "LR" for f in p.fields}
    env.update({n: n for n in params})
    lets = []
    for name, e in p.merge:
        lets.append(f"  let b_{name} := {_exact(e, env)}")
        env[name] = f"b_{name}"
    out.append(
        f"/-- Pass `{p.name}`, two adjacent summaries merged, exactly. -/\n"
        f"noncomputable def merge_{p.name}{_binders(params)} (L R : State_{p.name}) : "
        f"State_{p.name} :=\n" + "\n".join(lets) + f"\n  ⟨{state}⟩\n"
    )
    return "\n".join(out)


POLICY_KINDS = ("exact", "overflow", "fp")


def export_policy(prog: ir.Program, problem: Problem, module: str, kind: str) -> FPFile:
    """Statements for a rescaling policy (a softmax program with parameters, M4): ``exact``, the
    normalizer is exact over ℝ for every tree; ``overflow``, the sum field is at most
    ``n · exp(τ · rdepth t)``; ``fp``, the floating-point bound ``B`` of version 3, parametric in
    the program's parameters."""
    if problem.name != "softmax" or len(prog.passes) != 1 or kind not in POLICY_KINDS:
        raise FPExportError("rescaling policies: one-pass softmax programs, kind in POLICY_KINDS")
    p = prog.passes[0]
    params = tuple(prog.params)
    elem = prog.inputs[0]
    namespace = f"Discovery.FP.{module}"
    args = "".join(f" {n}" for n in params)
    hyps = "".join(f"∀ ({n} : ℝ), 0 ≤ {n} → " for n in params)
    out_expr = prog.output.exprs[0][1]
    env = {elem: "(0 : ℝ)", **{f"{p.name}.{f}": f"st.{f}" for f in p.fields}}
    parts = [
        "import Discovery.Approx\n",
        _HEAD.format(
            name=prog.name,
            version=VERSION,
            digest=prog.digest(),
            text=prog.text(),
            namespace=namespace,
            library="",
        ).replace("open Discovery Discovery.Library.\n", "open Discovery\n"),
        _exact_defs(p, elem, params),
        "\n",
        _RDEPTH,
        "\n",
        _block(p, {elem: "x"}, True, "ℝ", f"State_{p.name}", params),
        "\n",
        _merge(p, set(), f"State_{p.name}", params),
        "\n",
        "/-- The normalizer a state encodes: one over the output expression read at the element "
        "`0`. -/\n"
        f"noncomputable def znorm (st : State_{p.name}) : ℝ := 1 / {_exact(out_expr, env)}\n",
        _SMALL_DEF,
    ]
    ev = f"(t.eval (block_{p.name}{args}) (merge_{p.name}{args}))"
    head = (
        "  ∀ (X : List ℝ), X ≠ [] → ∀ (t : Discovery.Tree ℝ), t.flat = X →\n"
        "    (∀ b ∈ t.leaves, b ≠ []) →"
    )
    if kind == "exact":
        foralls = "".join(f"∀ ({n} : ℝ), " for n in params)
        template = f"  {foralls}\n{head}\n      znorm {ev} = (X.map Real.exp).sum"
    elif kind == "overflow":
        sumf = next(n for n, e in p.block if isinstance(e, ir.Reduce) and e.op == "sum")
        tau = params[0] if params else "0"
        bound_ = f"(X.length : ℝ) * Real.exp ({tau} * (rdepth t : ℝ))"
        template = f"  {hyps}\n{head}\n      {ev}.{sumf} ≤ {bound_}"
    else:
        nonneg = "".join(f"0 ≤ {n} → " for n in params)
        feval = f"Discovery.Tree.FEval (fblock_{p.name} F D{args}) (fmerge_{p.name} F{args}) t st"
        lead = f"  ∀ (F : Discovery.Rounding) (D : ℕ){_binders(params)}, {nonneg}"
        # Policy version 4: a stale sum can reach n·e^(τρ) (the overflow theorem), and an
        # absolute error of exp multiplies it; both models refuted version 3 with that (diary,
        # 10:45 UTC). The hypothesis is the overflow theorem's quantity times the absolute errors.
        stale = (
            f"\n    (X.length : ℝ) * Real.exp ({params[0]} * (rdepth t : ℝ)) * (F.η + F.ηe) ≤ F.u →"
            if params
            else ""
        )
        template = f"""{lead}∀ (lo hi : ℝ) (X : List ℝ),
    X ≠ [] → (∀ x ∈ X, lo ≤ x ∧ x ≤ hi) → small F → {TINY} →
    ∀ (t : Discovery.Tree ℝ), t.flat = X → (∀ b ∈ t.leaves, b ≠ []) →
    {DEEP} →{stale}
    ∀ (st : State_{p.name}), {feval} →
      |Real.log (znorm st) - Real.log ((X.map Real.exp).sum)| ≤ {{B}}"""
    return FPFile(
        module=f"Discovery/FP/{module}.lean",
        namespace=namespace,
        program=prog.text(),
        problem="policy",
        library="",
        definitions="".join(parts),
        statement_template=template,
    )
