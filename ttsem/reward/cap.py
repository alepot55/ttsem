"""Which answers launch a grid of cdiv(n, BLOCK) over a block constexpr, and the same answers with
that constexpr capped (read only: the answers are parsed with `ast`, never imported or run; the
standard library only).

For an answer's code (a TritonBench answer's operator part, before the separator, see
`operator_part`) `analyse` matches every launch `kernel[grid](...)` of a `@triton.jit` function
defined in it against the accepted patterns below. An answer qualifies when one of its launches
does. `rewrite` then replaces, at each qualifying launch, the block constexpr's value E by
`min(E, C)` and every divisor of the grid's cdiv that stands for it the same way, so the grid
covers what it covered with C as the block. No line is added or removed: the judge's line numbers
stay the answer's.

The accepted patterns (a launch qualifies when one axis of its grid is one of the cdiv forms below
over a divisor that is the value of one of the kernel's `tl.constexpr` parameters at that launch):

- cdiv forms of (X, D): `triton.cdiv(X, D)` (any callee named `cdiv`, `ceil_div`, `ceildiv`,
  `div_up`), `(X + D - 1) // D`, `(X + (D - 1)) // D`, `(X - 1) // D + 1`, `math.ceil(X / D)`,
  `-(-X // D)`, any of them under `int(...)`, and a product of such terms (a 2-D tiling flattened
  on one axis, as in grouped matmuls);
- grids: a tuple or list (inline, or a name bound to one), each element inline or a name bound to a
  cdiv form in the launching function (its last binding before the launch); a callable (an inline
  lambda, a name bound to a lambda, or a `def`) whose returned tuple has a cdiv form;
- divisors: `meta['P']` (`meta.get('P')`) in a callable grid, P the constexpr (pattern `meta`); or
  an expression equal to the constexpr's argument at the launch once names bound to constants are
  replaced by them (pattern `value`, a tuple grid or a callable grid reading a closure variable);
- where P's value comes from at the launch: a keyword or positional argument (wrapped there), the
  kernel's `@triton.autotune` configs (each config's value wrapped), a `@triton.heuristics` entry
  (its lambda's body wrapped) or the parameter's default (wrapped).

Not accepted: grids that are plain sizes (`(n_rows,)`, one instance per row), divisors that are not
the constexpr's value (`triton.cdiv(n, 128)` with no constexpr 128), kernels not defined in the
answer, starred arguments, and anything that does not parse.
"""

from __future__ import annotations

import ast
import functools
import re
from collections.abc import Callable
from dataclasses import dataclass, field

SEP = "#" * 146
CDIV_NAME = re.compile(r"^(cdiv|ceil_?div|div_?up|ceildiv)$", re.I)


def operator_part(text: str) -> str:
    return text.split(SEP)[0]


def dotted(node: ast.AST) -> str:
    """`a.b.c` for a Name/Attribute chain, '' otherwise."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return f"{base}.{node.attr}" if base else ""
    return ""


def last_name(node: ast.AST) -> str:
    name = dotted(node)
    return name.rsplit(".", 1)[-1] if name else ""


def is_constexpr(annotation: ast.AST | None) -> bool:
    return annotation is not None and last_name(annotation) == "constexpr"


def same(a: ast.AST, b: ast.AST) -> bool:
    return ast.dump(a) == ast.dump(b)


def is_one(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == 1


@dataclass
class Kernel:
    name: str
    params: list[str]
    constexpr: set[str]
    defaults: dict[str, ast.AST]
    configs: dict[str, list[ast.AST]]  # autotune: P -> the value nodes of every config
    heuristics: dict[str, ast.AST]  # P -> the lambda's body


@dataclass
class Launch:
    line: int
    kernel: str
    grid_kind: str  # tuple | callable
    pattern: str  # meta | value
    forms: list[str]
    params: list[str]  # the constexprs the grid divides by
    sources: list[str]  # where each value comes from: arg | autotune | heuristics | default
    values: list[str]  # the value expression of each, as text
    spans: list[tuple[int, int, int, int]] = field(default_factory=list)


def kernels_of(tree: ast.Module) -> dict[str, Kernel]:
    found: dict[str, Kernel] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        decos = node.decorator_list
        names = [last_name(d.func if isinstance(d, ast.Call) else d) for d in decos]
        if "jit" not in names:
            continue
        args = node.args
        params = [a.arg for a in args.posonlyargs + args.args]
        constexpr = {a.arg for a in args.posonlyargs + args.args if is_constexpr(a.annotation)}
        defaults: dict[str, ast.AST] = {}
        positional = args.posonlyargs + args.args
        for a, d in zip(
            positional[len(positional) - len(args.defaults) :], args.defaults, strict=True
        ):
            defaults[a.arg] = d
        configs: dict[str, list[ast.AST]] = {}
        heuristics: dict[str, ast.AST] = {}
        for d in decos:
            if not isinstance(d, ast.Call):
                continue
            kind = last_name(d.func)
            if kind == "autotune":
                cfg_nodes: list[ast.AST] = []
                for kw in d.keywords:
                    if kw.arg == "configs":
                        cfg_nodes.append(kw.value)
                if d.args:
                    cfg_nodes.append(d.args[0])
                for cfg_list in config_sources(cfg_nodes, tree):
                    for call in ast.walk(cfg_list):
                        if isinstance(call, ast.Call) and last_name(call.func) == "Config":
                            first = call.args[0] if call.args else None
                            for kw in call.keywords:
                                if kw.arg in ("kwargs", "meta"):
                                    first = kw.value
                            if isinstance(first, ast.Dict):
                                for k, v in zip(first.keys, first.values, strict=True):
                                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                                        configs.setdefault(k.value, []).append(v)
            elif kind == "heuristics" and d.args and isinstance(d.args[0], ast.Dict):
                for k, v in zip(d.args[0].keys, d.args[0].values, strict=True):
                    if isinstance(k, ast.Constant) and isinstance(v, ast.Lambda):
                        heuristics[str(k.value)] = v.body
        found[node.name] = Kernel(node.name, params, constexpr, defaults, configs, heuristics)
    return found


def config_sources(nodes: list[ast.AST], tree: ast.Module) -> list[ast.AST]:
    """The expressions an autotune's configs come from: the argument itself, and where it is a
    module-level name or a call of a module-level function, that name's value or that function's
    body, followed through the functions it calls (three levels)."""
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    values = {
        n.targets[0].id: n.value
        for n in tree.body
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
    }
    out: list[ast.AST] = []
    todo = [(n, 0) for n in nodes]
    seen: set[int] = set()
    while todo:
        node, depth = todo.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        out.append(node)
        if depth >= 3:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id in values:
                todo.append((values[sub.id], depth + 1))
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in defs:
                todo.append((defs[sub.func.id], depth + 1))
    return out


class Scope:
    """The bindings a launch can see: its enclosing function's simple assignments and nested
    defs, then the module's."""

    def __init__(self, chain: list[ast.AST]) -> None:
        self.chain = chain  # innermost first; the module last

    def binding(self, name: str, before: int) -> ast.AST | None:
        """The value of `name`'s last simple binding before line `before` in the innermost scope
        that binds it (at module level, any line), a def of that name, or None."""
        for scope in self.chain:
            module = scope is self.chain[-1]
            best: tuple[int, ast.AST] | None = None
            for node in nodes_in_scope(scope):
                line = getattr(node, "lineno", 0)
                if not module and line >= before:
                    continue
                value: ast.AST | None = None
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    value = node
                elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target = node.targets[0]
                    if isinstance(target, ast.Name) and target.id == name:
                        value = node.value
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    if node.target.id == name and node.value is not None:
                        value = node.value
                if value is not None and (best is None or line >= best[0]):
                    best = (line, value)
            if best is not None:
                return best[1]
            if isinstance(scope, ast.FunctionDef):
                if name in {a.arg for a in scope.args.args + scope.args.kwonlyargs}:
                    return None  # a parameter of the launching function: not resolved further
        return None


def nodes_in_scope(scope: ast.AST) -> list[ast.AST]:
    """The nodes of `scope` that are not inside a nested function (only the scope's own code)."""
    out: list[ast.AST] = []
    todo = list(ast.iter_child_nodes(scope))
    while todo:
        node = todo.pop()
        out.append(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        todo.extend(ast.iter_child_nodes(node))
    return out


def canonical(node: ast.AST, scope: Scope, before: int, depth: int = 0) -> ast.AST:
    """`node` with a name bound to a constant (directly or through names) replaced by it."""
    if isinstance(node, ast.Name) and depth < 4:
        bound = scope.binding(node.id, before)
        if isinstance(bound, ast.Constant):
            return bound
        if isinstance(bound, ast.Name):
            got = canonical(bound, scope, before, depth + 1)
            if isinstance(got, ast.Constant):
                return got
    return node


def resolve(node: ast.AST, scope: Scope, before: int, depth: int = 0) -> ast.AST:
    """A name followed to its binding (a value expression or a def), up to four times."""
    while isinstance(node, ast.Name) and depth < 4:
        bound = scope.binding(node.id, before)
        if bound is None:
            return node
        node, depth = bound, depth + 1
    return node


def cdiv_terms(
    node: ast.AST, follow: Callable[[ast.AST], ast.AST] = lambda n: n
) -> list[tuple[str, ast.AST, list[ast.AST]]] | None:
    """The cdiv terms of an axis expression: [(form, X, [divisor nodes])], or None if it is not
    one of the accepted forms (or a product of them). `follow` resolves a name to its binding."""
    node = follow(node)
    if isinstance(node, ast.Call) and last_name(node.func) == "int" and len(node.args) == 1:
        return cdiv_terms(node.args[0], follow)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left, right = cdiv_terms(node.left, follow), cdiv_terms(node.right, follow)
        if left is not None and right is not None:
            return left + right
        return None
    one = cdiv_one(node)
    return [one] if one is not None else None


def cdiv_one(node: ast.AST) -> tuple[str, ast.AST, list[ast.AST]] | None:
    if isinstance(node, ast.Call) and len(node.args) == 2 and CDIV_NAME.match(last_name(node.func)):
        return ("cdiv", node.args[0], [node.args[1]])
    if isinstance(node, ast.Call) and last_name(node.func) == "ceil" and len(node.args) == 1:
        inner = node.args[0]
        if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Div):
            return ("ceil", inner.left, [inner.right])
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv):
        d, top = node.right, node.left
        if isinstance(top, ast.BinOp):
            # (X + D - 1) // D
            if isinstance(top.op, ast.Sub) and is_one(top.right):
                inner = top.left
                if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Add):
                    if same(inner.right, d):
                        return ("add_sub", inner.left, [inner.right, d])
            # (X + (D - 1)) // D
            if isinstance(top.op, ast.Add) and isinstance(top.right, ast.BinOp):
                sub = top.right
                if isinstance(sub.op, ast.Sub) and is_one(sub.right) and same(sub.left, d):
                    return ("add_sub", top.left, [sub.left, d])
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add) and is_one(node.right):
        # (X - 1) // D + 1
        fd = node.left
        if isinstance(fd, ast.BinOp) and isinstance(fd.op, ast.FloorDiv):
            top = fd.left
            if isinstance(top, ast.BinOp) and isinstance(top.op, ast.Sub) and is_one(top.right):
                return ("sub_add", top.left, [fd.right])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        # -(-X // D)
        fd = node.operand
        if isinstance(fd, ast.BinOp) and isinstance(fd.op, ast.FloorDiv):
            if isinstance(fd.left, ast.UnaryOp) and isinstance(fd.left.op, ast.USub):
                return ("neg", fd.left.operand, [fd.right])
    return None


def meta_key(node: ast.AST, meta: str) -> str | None:
    """P for `meta['P']` or `meta.get('P')`, `meta` the callable grid's parameter."""
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        if node.value.id == meta and isinstance(node.slice, ast.Constant):
            return str(node.slice.value)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "get" and isinstance(node.func.value, ast.Name):
            if node.func.value.id == meta and node.args and isinstance(node.args[0], ast.Constant):
                return str(node.args[0].value)
    return None


def span(node: ast.AST) -> tuple[int, int, int, int]:
    line, col = int(getattr(node, "lineno", 0)), int(getattr(node, "col_offset", 0))
    end_line = getattr(node, "end_lineno", None) or line
    return (line, col, int(end_line), int(getattr(node, "end_col_offset", None) or 0))


def grid_axes(
    grid: ast.AST, scope: Scope, line: int
) -> tuple[str, str | None, list[ast.AST]] | None:
    """(kind, the callable's meta parameter or None, the axis expressions) of a launch's grid."""
    node = resolve(grid, scope, line)
    meta: str | None = None
    kind = "tuple"
    if isinstance(node, ast.Lambda):
        kind = "callable"
        meta = node.args.args[0].arg if node.args.args else None
        node = node.body
    elif isinstance(node, ast.FunctionDef):
        kind = "callable"
        meta = node.args.args[0].arg if node.args.args else None
        returns = [n for n in ast.walk(node) if isinstance(n, ast.Return) and n.value is not None]
        if len(returns) != 1:
            return None
        node = returns[0].value  # type: ignore[assignment]
    if isinstance(node, (ast.Tuple, ast.List)):
        return kind, meta, list(node.elts)
    return kind, meta, [node]  # a bare axis count (`kernel[n_blocks]`-style tuples of one)


def value_of(kernel: Kernel, param: str, call: ast.Call) -> tuple[str, list[ast.AST]] | None:
    """Where the constexpr's value comes from at this launch, and the nodes to wrap."""
    for kw in call.keywords:
        if kw.arg == param:
            return "arg", [kw.value]
    if param in kernel.params:
        index = kernel.params.index(param)
        before = call.args[: index + 1]
        if index < len(call.args) and not any(isinstance(a, ast.Starred) for a in before):
            return "arg", [call.args[index]]
    if any(kw.arg is None for kw in call.keywords) or any(
        isinstance(a, ast.Starred) for a in call.args
    ):
        return None  # **kwargs or *args may carry it
    if param in kernel.configs:
        return "autotune", kernel.configs[param]
    if param in kernel.heuristics:
        return "heuristics", [kernel.heuristics[param]]
    if param in kernel.defaults:
        return "default", [kernel.defaults[param]]
    return None


def analyse(source: str) -> tuple[list[Launch], str]:
    """The qualifying launches of an answer's code, and why none qualifies when none does."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return [], "does not parse"
    kernels = kernels_of(tree)
    if not kernels:
        return [], "no @triton.jit kernel defined"
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    launches: list[Launch] = []
    seen_launch = False
    for call in ast.walk(tree):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Subscript)):
            continue
        kname = last_name(call.func.value)
        if kname not in kernels:
            continue
        seen_launch = True
        kernel = kernels[kname]
        chain: list[ast.AST] = []
        up: ast.AST | None = call
        while up is not None:
            up = parents.get(id(up))
            if isinstance(up, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chain.append(up)
        chain.append(tree)
        scope = Scope(chain)
        axes = grid_axes(call.func.slice, scope, call.lineno)
        if axes is None:
            continue
        kind, meta, elements = axes
        params: list[str] = []
        forms: list[str] = []
        sources: list[str] = []
        values: list[str] = []
        spans: list[tuple[int, int, int, int]] = []
        pattern = ""
        resolver = functools.partial(resolve, scope=scope, before=call.lineno)
        for element in elements:
            terms = cdiv_terms(element, resolver)
            if terms is None:
                continue
            for form, _x, divisors in terms:
                divisor = divisors[-1]
                chosen: list[str] = []
                wrap_divisor = False
                key = meta_key(divisor, meta) if meta is not None else None
                if key is not None:
                    chosen = [key]
                    pattern = pattern or "meta"
                else:
                    passed = {}
                    for p in sorted(kernel.constexpr):
                        got = value_of(kernel, p, call)
                        if got is not None and got[0] == "arg":
                            passed[p] = got[1][0]
                    # the same expression first (`cdiv(n, BLOCK)` and `BLOCK_SIZE=BLOCK`), else the
                    # same constant once names bound to constants are replaced (every constexpr
                    # given that constant is capped with it, so the grid and the kernel agree)
                    chosen = [p for p, v in passed.items() if same(v, divisor)]
                    if not chosen:
                        want = ast.dump(canonical(divisor, scope, call.lineno))
                        chosen = [
                            p
                            for p, v in passed.items()
                            if ast.dump(canonical(v, scope, call.lineno)) == want
                        ]
                    if not chosen:
                        continue
                    wrap_divisor = True
                    pattern = "value" if pattern in ("", "value") else "mixed"
                used = False
                for param in chosen:
                    if param not in kernel.constexpr:
                        continue
                    got = value_of(kernel, param, call)
                    if got is None:
                        continue
                    source_kind, nodes = got
                    used = True
                    params.append(param)
                    forms.append(form)
                    sources.append(source_kind)
                    values.append(" | ".join(ast.unparse(n) for n in nodes))
                    spans += [span(n) for n in nodes]
                if used and wrap_divisor:
                    spans += [span(d) for d in divisors]
        if params:
            launches.append(
                Launch(call.lineno, kname, kind, pattern, forms, params, sources, values, spans)
            )
    if launches:
        return launches, ""
    if not seen_launch:
        return [], "no launch of a kernel defined here"
    return [], "no cdiv grid over a constexpr"


def shadows_min(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "min" and isinstance(node.ctx, ast.Store):
            return True
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == "min":
            return True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if any((a.asname or a.name) == "min" for a in node.names):
                return True
        if isinstance(node, ast.arg) and node.arg == "min":
            return True
    return False


def rewrite(text: str, spans: list[tuple[int, int, int, int]], cap: int, use_min: bool) -> str:
    """`text` with each span E replaced by `min(E, cap)` (or a conditional where `min` is
    rebound), on the same lines. Offsets are those of `ast`: UTF-8 bytes."""
    lines = text.encode("utf-8").split(b"\n")
    starts = [0]
    for line in lines[:-1]:
        starts.append(starts[-1] + len(line) + 1)
    data = text.encode("utf-8")
    edits: list[tuple[int, int, bytes]] = []  # (offset, order, text): starts before ends at ties
    for sl, sc, el, ec in sorted(set(spans)):
        a, b = starts[sl - 1] + sc, starts[el - 1] + ec
        if use_min:
            edits += [(a, 0, b"min("), (b, 1, f", {cap})".encode())]
        else:
            expr = data[a:b]
            edits += [(a, 0, b"(" + expr + b" if "), (b, 1, f" <= {cap} else {cap})".encode())]
    for offset, _order, insert in sorted(edits, key=lambda e: (e[0], -e[1]), reverse=True):
        data = data[:offset] + insert + data[offset:]
    out = data.decode("utf-8")
    if out.count("\n") != text.count("\n"):
        raise AssertionError("a rewrite changed the number of lines")
    return out
