"""The Triton reading of a program: one kernel, one program instance per row, a loop over blocks.

``generate(prog, problem)`` returns the source of a Python module with one ``@triton.jit`` kernel
and a launcher ``run(inputs) -> outputs`` (dicts of torch tensors). Inside the kernel each pass
is a loop over blocks of ``BLOCK`` elements: the first block's summary is the state, and every
later block's summary (``R``) is merged into it (``L``, the state so far). That is the left fold
of the numpy semantics with blocks of ``BLOCK``, so a merge that only works for some blockings
fails here as it does there. The last block is masked: a reduction reduces
``tl.where(mask, body, identity)`` (0 for ``sum``, -inf for ``rmax``, +inf for ``rmin``), so
``(sum 1)`` counts the elements. An elementwise output is one more loop over the blocks.

The kernel is compiled without FMA contraction (``enable_fp_fusion=False``), so every ``+``,
``-`` and ``*`` of the program is one float32 operation rounded to nearest, as in the semantics.
With contraction, the one-pass variance ``u/n - (t/n)*(t/n)`` of a single element is the rounding
error of ``x*x`` instead of 0, a behaviour the floating-point class never saw. On the RTX A1000
this costs nothing on the memory-bound rows and up to 4% on attention.

The other operations are not rounded as numpy rounds them. Measured with Triton 3.8 on sm_86,
against the float64 result rounded to float32: ``/`` is ``div.full.f32``, within 2 ulp, where
numpy's quotient is correctly rounded; ``exp(x)`` is ``ex2.approx.f32`` of ``x * log2(e)``, and
since the product is rounded first its error grows with ``|x|``: up to 2 ulp for ``|x| < 1``, 15
for ``|x| < 16``, 64 at most; ``sqrt`` is ``sqrt.approx.ftz.f32``, within 1 ulp, with subnormal
inputs flushed to zero; ``log`` is a polynomial evaluated with FMAs, within 1 ulp.
numpy's own float32 ``exp`` and ``log`` are not correctly rounded either (2 and 3 ulp here), and
a block's ``sum`` adds in ``tl.sum``'s order, not numpy's. So the kernel does not reproduce the
float32 semantics op for op: ``discovery.gpu`` checks it against float64, and the tests check
that it falls in the floating-point class the semantics measured.

The source is written to be read, since a reviewer checks the algorithm in it: the program heads
the file, ``a_m`` is field ``m`` of pass ``a``, and the block and merge lines are the program's
bindings in order, with the program's names where Python allows them.

Layouts, float32 in memory:

* ``rows``, when every element field is a scalar (softmax, variance): each field is a
  ``(rows, N)`` tensor named after it; an elementwise output is ``(rows, N)``, a final output
  ``(rows,)``.
* ``attention``, for one scalar field and one vector field: ``q (rows, D)``, ``k (N, D)``,
  ``v (N, D)``, every query attending to the same keys. The scalar of key ``j`` is its score
  ``sum(q * k_j) / sqrt(D)``, computed in the kernel from the key block; the vector is the value
  row. ``D`` is a power of two. Outputs are ``(rows, D)`` for a vector, ``(rows,)`` for a scalar.
"""

from __future__ import annotations

import dataclasses
import keyword
import re
import textwrap

import numpy as np

from discovery import ir, semantics
from discovery.problems import Problem

# elements per block and warps per program instance, timed on the benchmark shapes of
# discovery.gpu on an RTX A1000 (sm_86) against BLOCK 512-4096 with 4-16 warps (rows) and 32-128
# with 1-8 warps (attention): within 2% of the best setting tried on every row shape, within 10%
# on attention, where no setting is best on all four
BLOCK = {"rows": 2048, "attention": 64}
NUM_WARPS = {"rows": 16, "attention": 2}

# what every generated module says, in its header, about how the kernel rounds (see the docstring)
ROUNDING = (
    "Rounding: no FMA contraction, so every + - * of the program is one float32 operation "
    "rounded to nearest, as in the semantics; a block's sum adds in tl.sum's order. The rest is "
    "not numpy's float32 (Triton 3.8 on sm_86, against the float64 result rounded to float32): "
    "/ is div.full.f32, within 2 ulp; exp(x) is ex2.approx.f32 of x*log2(e), up to 2 ulp for "
    "|x| < 1, 15 for |x| < 16, 64 at most; sqrt is sqrt.approx.ftz.f32, within 1 ulp, "
    "subnormal inputs flushed to zero; log is a polynomial evaluated with FMAs, within 1 ulp."
)

# precedence of the Python operators the kernel uses, loosest first
_ADD, _MUL, _UNARY, _ATOM = 1, 2, 3, 4
_BINARY = {"+": ("+", _ADD), "-": ("-", _ADD), "*": ("*", _MUL), "/": ("/", _MUL)}
_FUNCTIONS = {
    "exp": "tl.exp",
    "log": "tl.log",
    "sqrt": "tl.sqrt",
    "abs": "tl.abs",
    "max": "tl.maximum",
    "min": "tl.minimum",
}
_REDUCTIONS = {
    "sum": ("tl.sum", "0.0"),
    "rmax": ("tl.max", 'float("-inf")'),
    "rmin": ("tl.min", 'float("inf")'),
}
_F32 = semantics.Arith("float32")

# names the generated module uses for itself: at module level, in the kernel, in the launcher
_MODULE = {"torch", "triton", "tl", "run", "BLOCK", "NUM_WARPS", "float", "range"}
_KERNEL = {"N", "row", "offs", "mask", "start"}
_LAUNCHER = {"inputs", "rows", "n"}
# and those of the attention layout: the query, the keys, D (`d` in the launcher)
_ATTENTION = {"math", "D", "dims", "q", "k", "q_ptr", "k_ptr", "stride_q", "stride_k", "sqrt_d"}


def layout(problem: Problem) -> str:
    """``rows`` when every element field is a scalar in memory; ``attention`` for one scalar (the
    score, computed from a query and a key) and one vector (the value)."""
    kinds = sorted(problem.elems.values())
    if all(k == "scalar" for k in kinds):
        return "rows"
    if kinds == ["scalar", "vec"]:
        return "attention"
    raise ValueError(f"no kernel layout for the elements of {problem.name} ({problem.elems})")


def block_for(problem: Problem) -> int:
    return BLOCK[layout(problem)]


def generate(prog: ir.Program, problem: Problem, block: int | None = None) -> str:
    """The source of a module with the kernel of ``prog`` and its launcher ``run``."""
    lay = layout(problem)
    block = block or BLOCK[lay]
    if block < 1 or block & (block - 1):
        raise ValueError(f"BLOCK must be a power of two, not {block}")
    return _Kernel(prog, problem, lay, block).source()


# --------------------------------------------------------------------------- expressions


@dataclasses.dataclass(frozen=True)
class _Code:
    """A Python expression of the kernel and the shape of its value: ``()`` for one scalar per
    row, ``(D,)`` with ``vec``, ``(BLOCK,)`` with ``elem`` (one scalar per element of the block),
    ``(BLOCK, D)`` with both. ``value`` is set for a number known when the kernel is generated."""

    text: str
    elem: bool = False
    vec: bool = False
    prec: int = _ATOM
    value: float | None = None


def _literal(v: float) -> _Code:
    if v != v:
        return _Code('float("nan")', value=v)
    if v in (float("inf"), float("-inf")):
        return _Code(f'float("{v}")', value=v)
    text = repr(float(v))
    return _Code(text, prec=_UNARY if text.startswith("-") else _ATOM, value=v)


def _wrap(c: _Code, prec: int) -> str:
    return c.text if c.prec >= prec else f"({c.text})"


def _broadcast(c: _Code, elem: bool, vec: bool) -> _Code:
    """``c`` indexed to broadcast against a value with one vector per element: an element scalar
    becomes ``(BLOCK, 1)``, a row vector ``(1, D)``."""
    if elem and vec and c.elem != c.vec:
        index = "[:, None]" if c.elem else "[None, :]"
        return _Code(_wrap(c, _ATOM) + index, True, True)
    return c


def _call(op: str, args: list[_Code]) -> _Code:
    elem, vec = any(a.elem for a in args), any(a.vec for a in args)
    args = [_broadcast(a, elem, vec) for a in args]
    if op == "neg":
        return _Code("-" + _wrap(args[0], _UNARY + 1), elem, vec, _UNARY)
    if op in _FUNCTIONS:
        return _Code(f"{_FUNCTIONS[op]}({', '.join(a.text for a in args)})", elem, vec)
    sym, prec = _BINARY[op]
    acc = args[0]
    for b in args[1:]:  # left to right, as the semantics folds (+ a b c)
        right = f"({b.text})" if b.prec <= prec or b.prec == _UNARY else b.text
        acc = _Code(f"{_wrap(acc, prec)} {sym} {right}", elem, vec, prec)
    return acc


def _reduce(op: str, body: _Code) -> _Code:
    fn, identity = _REDUCTIONS[op]
    if body.vec:
        body, mask = _broadcast(body, True, True), "mask[:, None]"
    else:
        mask = "mask"
    return _Code(f"{fn}(tl.where({mask}, {body.text}, {identity}), axis=0)", vec=body.vec)


def _expr(e: ir.Expr, env: dict[str, _Code]) -> _Code:
    if isinstance(e, ir.Num):
        return _literal(e.value)
    if isinstance(e, ir.Var):
        return env[e.name]
    if isinstance(e, ir.Reduce):
        return _reduce(e.op, _expr(e.body, env))
    args = [_expr(a, env) for a in e.args]
    if all(a.value is not None for a in args):
        # a constant: folded in float32 as the semantics computes it (Triton cannot divide
        # Python numbers by zero, nor take tl.exp of one)
        with np.errstate(all="ignore"):
            return _literal(float(semantics.evaluate(e, {}, None, _F32)))
    return _call(e.op, args)


def _state_value(c: _Code) -> str:
    """A state field's value as a float32 tensor, so that the loop can carry it."""
    return f"tl.full((), {c.text}, tl.float32)" if c.value is not None else c.text


def _comment(text: str) -> str:
    return "".join(ch if ch.isprintable() else ch.encode("unicode_escape").decode() for ch in text)


# --------------------------------------------------------------------------- names


class _Names:
    """Python identifiers for the program's names: valid, close to the original, and never one
    identifier for two things."""

    def __init__(self, taken: set[str]) -> None:
        self.taken = set(taken)

    def new(self, base: str) -> str:
        ident = re.sub(r"[^0-9A-Za-z_]", "_", base)
        if not ident or ident[0].isdigit():
            ident = "_" + ident
        name, i = ident, 2
        while name in self.taken or keyword.iskeyword(name):
            name, i = f"{ident}_{i}", i + 1
        self.taken.add(name)
        return name

    def child(self) -> _Names:
        """Names for the body of one loop: they may repeat in another loop, never outside."""
        return _Names(self.taken)


# --------------------------------------------------------------------------- the kernel


class _Kernel:
    def __init__(self, prog: ir.Program, problem: Problem, lay: str, block: int) -> None:
        self.prog, self.problem, self.layout, self.block = prog, problem, lay, block
        if prog.output.kind == "elementwise" and lay != "rows":
            raise ValueError(f"an elementwise output has no layout for {problem.name}")
        # the names that live in more than one loop or function, unique in the whole module
        launcher_only = set(_LAUNCHER)
        reserved = _MODULE | _KERNEL | _LAUNCHER
        if lay == "attention":
            reserved |= _ATTENTION | {"d"}
            launcher_only.add("d")
        names = _Names(reserved)
        self.elem = {f: names.new(f) for f in problem.elems}
        self.ptr: dict[str, str] = {}
        self.stride: dict[str, str] = {}
        self.row_ptr: dict[str, str] = {}
        if lay == "rows":
            self.fields = list(problem.elems)
            for f in self.fields:
                self.ptr[f] = names.new(f"{self.elem[f]}_ptr")
                self.stride[f] = names.new(f"stride_{self.elem[f]}")
                self.row_ptr[f] = names.new(f"{self.elem[f]}_row")
        else:  # the score is not in memory: q and the key rows are
            self.score = next(f for f, t in problem.elems.items() if t == "scalar")
            self.value = next(f for f, t in problem.elems.items() if t == "vec")
            self.ptr[self.value] = names.new(f"{self.elem[self.value]}_ptr")
            self.stride[self.value] = names.new(f"stride_{self.elem[self.value]}")
        # outputs: (name, identifier, vec), the shape the problem gives them
        if prog.output.kind == "elementwise":
            self.outputs = [("y", names.new("y"), False)]
        else:
            assert isinstance(problem.outputs, dict)
            self.outputs = [(o, names.new(o), problem.outputs[o] == "vec") for o in problem.outputs]
        for o, ident, vec in self.outputs:
            self.ptr[o] = names.new(f"{ident}_ptr")
            if vec or prog.output.kind == "elementwise":
                self.stride[o] = names.new(f"stride_{ident}")
            if prog.output.kind == "elementwise":
                self.row_ptr[o] = names.new(f"{ident}_row")
        self.fn = names.new(prog.name)
        self.state = {
            (p.name, f): names.new(f"{p.name}_{f}") for p in prog.passes for f in p.fields
        }
        # the names of a loop's body avoid the module's and the kernel's, not the launcher's
        launcher_only |= {ident for _, ident, _ in self.outputs}
        self.kernel_names = _Names(names.taken - launcher_only)

    # ------------------------------------------------------------------ source

    def source(self) -> str:
        prog = self.prog
        head = [
            f"# Generated by discovery.codegen_triton from program {_comment(prog.name)}",
            f"# (digest {prog.digest()}), problem {self.problem.name}, layout {self.layout}:",
            "#",
            *(f"#   {_comment(line)}" for line in prog.text().splitlines()),
            "#",
            "# One program instance per row. Each pass is a loop over blocks of BLOCK elements:",
            "# the first block's summary is the state, every later block's summary (R) is merged",
            "# into the state so far (L), the left fold of the program's semantics. The last",
            "# block is masked: a reduction reduces tl.where(mask, body, identity).",
            "#",
            *(f"# {line}" for line in textwrap.wrap(ROUNDING, width=86)),
            "",
        ]
        imports = ["import math", ""] if self.layout == "attention" else []
        imports += ["import torch", "import triton", "import triton.language as tl", ""]
        consts = [f"BLOCK = {self.block}", f"NUM_WARPS = {NUM_WARPS[self.layout]}", "", ""]
        return (
            "\n".join([*head, *imports, *consts, *self.kernel(), "", "", *self.launcher()]) + "\n"
        )

    def kernel(self) -> list[str]:
        if self.layout == "rows":
            params = [self.ptr[f] for f in self.fields] + [self.ptr[o] for o, _, _ in self.outputs]
            params += ["N"] + [self.stride[f] for f in self.fields]
            params += [self.stride[o] for o, _, _ in self.outputs if o in self.stride]
            params += ["BLOCK: tl.constexpr"]
            body = ["row = tl.program_id(0).to(tl.int64)"]
            body += [
                f"{self.row_ptr[f]} = {self.ptr[f]} + row * {self.stride[f]}" for f in self.fields
            ]
            body += [
                f"{self.row_ptr[o]} = {self.ptr[o]} + row * {self.stride[o]}"
                for o, _, _ in self.outputs
                if o in self.row_ptr
            ]
        else:
            v = self.value
            params = ["q_ptr", "k_ptr", self.ptr[v]] + [self.ptr[o] for o, _, _ in self.outputs]
            params += ["N", "stride_q", "stride_k", self.stride[v]]
            params += [self.stride[o] for o, _, vec in self.outputs if vec]
            params += ["sqrt_d", "BLOCK: tl.constexpr", "D: tl.constexpr"]
            body = [
                "# one program instance per query; the elements are the N keys and their values",
                "row = tl.program_id(0).to(tl.int64)",
                "dims = tl.arange(0, D)",
                "q = tl.load(q_ptr + row * stride_q + dims)",
            ]
        finished: dict[str, _Code] = {}
        reads = ir.reads_elements(self.prog)
        for i, p in enumerate(self.prog.passes):
            body += ["", *self.fold(p, reads[i], finished)]
            for f in p.fields:
                vec = self.prog.types[f"{p.name}.{f}"] == "vec"
                finished[f"{p.name}.{f}"] = _Code(self.state[p.name, f], vec=vec)
        body += ["", *self.output(finished, reads[-1])]
        return ["@triton.jit", *_call_lines(f"def {self.fn}", params, ":")] + _indent(body)

    def load(self, fields: set[str], first: bool) -> list[str]:
        """The offsets and mask of one block, and the element fields ``fields`` of its elements."""
        lines = [
            "offs = tl.arange(0, BLOCK)" if first else "offs = start + tl.arange(0, BLOCK)",
            "mask = offs < N",
        ]
        if self.layout == "rows":
            lines += [
                f"{self.elem[f]} = tl.load({self.row_ptr[f]} + offs, mask=mask, other=0.0)"
                for f in self.fields
                if f in fields
            ]
            return lines
        tile = "offs[:, None] * {} + dims[None, :], mask=mask[:, None], other=0.0"
        if self.score in fields:
            lines.append(f"k = tl.load(k_ptr + {tile.format('stride_k')})")
            lines.append(f"{self.elem[self.score]} = tl.sum(q[None, :] * k, axis=1) / sqrt_d")
        if self.value in fields:
            v = self.value
            lines.append(f"{self.elem[v]} = tl.load({self.ptr[v]} + {tile.format(self.stride[v])})")
        return lines

    def elements(self) -> dict[str, _Code]:
        return {
            f: _Code(self.elem[f], elem=True, vec=t == "vec") for f, t in self.problem.elems.items()
        }

    def summary(
        self, p: ir.Pass, finished: dict[str, _Code], target: dict[str, str], names: _Names
    ) -> list[str]:
        """The block bindings of ``p``: a state field into ``target[field]``, a temporary into a
        new name from ``names``."""
        env = {**finished, **self.elements()}
        lines = []
        for f, e in p.block:
            c = _expr(e, env)
            ident = target[f] if f in target else names.new(f"t_{f}")
            lines.append(f"{ident} = {_state_value(c)}")
            env[f] = _Code(ident, vec=c.vec)
        return lines

    def fold(self, p: ir.Pass, reads: set[str], finished: dict[str, _Code]) -> list[str]:
        fields = p.fields
        state = {f: self.state[p.name, f] for f in fields}
        # the first block's temporaries are kernel names, taken before the loop's names are
        # drawn: a loop name equal to one of them would make Triton carry it through the loop
        first = self.summary(p, finished, state, self.kernel_names)
        loop = self.kernel_names.child()
        R = {f: loop.new(f"R_{f}") for f in fields}
        L = {f: loop.new(f"L_{f}") for f in fields}
        merged = {f: loop.new(f) for f in fields}
        env = dict(finished)
        for f in fields:
            vec = self.prog.types[f"{p.name}.{f}"] == "vec"
            env[f"L.{f}"], env[f"R.{f}"] = _Code(L[f], vec=vec), _Code(R[f], vec=vec)
        merge = []
        for f, e in p.merge:
            c = _expr(e, env)
            ident = merged[f] if f in merged else loop.new(f)
            merge.append(f"{ident} = {_state_value(c)}")
            env[f] = _Code(ident, vec=c.vec)
        body = [
            *self.load(reads, first=False),
            *self.summary(p, finished, R, loop),
            f"{', '.join(L[f] for f in fields)} = {', '.join(state[f] for f in fields)}",
            *merge,
            f"{', '.join(state[f] for f in fields)} = {', '.join(merged[f] for f in fields)}",
        ]
        return [
            f"# pass {_comment(p.name)} (state {_comment(' '.join(fields))}): the summary of the "
            "first block is the state",
            *self.load(reads, first=True),
            *first,
            "# every later block: its summary R, merged into the state so far L",
            "for start in range(BLOCK, N, BLOCK):",
            *_indent(body),
        ]

    def output(self, finished: dict[str, _Code], reads: set[str]) -> list[str]:
        if self.prog.output.kind == "elementwise":
            name, ident, _ = self.outputs[0]
            c = _expr(self.prog.output.exprs[0][1], {**finished, **self.elements()})
            if c.vec:
                raise ValueError("an elementwise output of one vector per element has no layout")
            body = [
                *self.load(reads, first=False),
                f"{ident} = {c.text}",
                f"tl.store({self.row_ptr[name]} + offs, {ident}, mask=mask)",
            ]
            return [
                f"# output: {_comment(name)} for every element, one more pass over the row",
                "for start in range(0, N, BLOCK):",
                *_indent(body),
            ]
        lines = ["# outputs, one per row"]
        exprs = dict(self.prog.output.exprs)
        for name, _, vec in self.outputs:
            c = _expr(exprs[name], finished)
            if c.vec and not vec:
                raise ValueError(f"output {name} is a vector in the program, a scalar here")
            where = f" + row * {self.stride[name]} + dims" if vec else " + row"
            lines.append(f"tl.store({self.ptr[name]}{where}, {c.text})")
        return lines

    # ------------------------------------------------------------------ launcher

    def launcher(self) -> list[str]:
        outs = self.outputs
        if self.layout == "rows":
            first = self.elem[self.fields[0]]
            elementwise = self.prog.output.kind == "elementwise"
            body = [
                f"{self.elem[f]} = inputs[{f!r}].to(torch.float32).contiguous()"
                for f in self.fields
            ]
            body += [f"rows, n = {first}.shape"]
            body += [
                f"assert {self.elem[f]}.shape == {first}.shape, 'the fields have one shape'"
                for f in self.fields[1:]
            ]
            body += ["if n == 0:", "    raise ValueError('a row needs one element or more')"]
            shape = "(rows, n)" if elementwise else "rows"
            body += [
                f"{ident} = torch.empty({shape}, device={first}.device, dtype=torch.float32)"
                for _, ident, _ in outs
            ]
            args = [self.elem[f] for f in self.fields] + [ident for _, ident, _ in outs] + ["n"]
            args += [f"{self.elem[f]}.stride(0)" for f in self.fields]
            args += [f"{ident}.stride(0)" for o, ident, _ in outs if o in self.stride]
            args += ["BLOCK=BLOCK", "num_warps=NUM_WARPS", "enable_fp_fusion=False"]
            ins = ", ".join(f"{f} (rows, N)" for f in self.fields)
            got = ", ".join(f"{o} {'(rows, N)' if elementwise else '(rows,)'}" for o, _, _ in outs)
            doc = f"{ins} float32 -> {got} float32; one program instance per row."
        else:
            v = self.elem[self.value]
            body = [
                "q = inputs['q'].to(torch.float32).contiguous()",
                "k = inputs['k'].to(torch.float32).contiguous()",
                f"{v} = inputs['v'].to(torch.float32).contiguous()",
                "rows, d = q.shape",
                "n = k.shape[0]",
                "if n == 0 or d < 2 or d & (d - 1):",
                "    raise ValueError('one key or more, and D a power of two')",
            ]
            body += [
                f"{ident} = torch.empty({'(rows, d)' if vec else 'rows'}, device=q.device, "
                "dtype=torch.float32)"
                for _, ident, vec in outs
            ]
            args = ["q", "k", v] + [ident for _, ident, _ in outs] + ["n"]
            args += ["q.stride(0)", "k.stride(0)", f"{v}.stride(0)"]
            args += [f"{ident}.stride(0)" for _, ident, vec in outs if vec]
            args += ["math.sqrt(d)", "BLOCK=BLOCK", "D=d", "num_warps=NUM_WARPS"]
            args += ["enable_fp_fusion=False"]
            got = ", ".join(f"{o} {'(rows, D)' if vec else '(rows,)'}" for o, _, vec in outs)
            doc = (
                f"q (rows, D), k (N, D), v (N, D) float32 -> {got} float32: every query attends "
                "to the N keys; one program instance per query."
            )
        body += _call_lines(f"{self.fn}[(rows,)]", args)
        body += ["return {" + ", ".join(f"{o!r}: {ident}" for o, ident, _ in outs) + "}"]
        return [
            "def run(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:",
            *_indent(_docstring(doc)),
            *_indent(body),
        ]


def _indent(lines: list[str]) -> list[str]:
    return ["    " + line if line else "" for line in lines]


def _call_lines(head: str, args: list[str], tail: str = "") -> list[str]:
    """``head(args)tail`` on one line, or with the arguments on lines of their own."""
    one = f"{head}({', '.join(args)}){tail}"
    if len(one) <= 92:
        return [one]
    lines = [""]
    for a in args:
        if lines[-1] and len(lines[-1]) + len(a) + 2 > 88:
            lines.append("")
        lines[-1] += f"{' ' if lines[-1] else ''}{a},"
    return [f"{head}(", *_indent(lines), f"){tail}"]


def _docstring(text: str) -> list[str]:
    lines = textwrap.wrap(text, width=88)
    if len(lines) == 1:
        return [f'"""{text}"""']
    return [f'"""{lines[0]}', *lines[1:-1], f'{lines[-1]}"""']
