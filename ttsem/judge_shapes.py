"""Smaller inputs for a KernelBench task, by rewriting its module-level sizes (``ttsem.judge``).

A KernelBench task sizes its inputs with module-level constants (`batch_size = 16`,
`dim = 16384`, `M = 256 * 8`, `input_shape = (32768,)`) that `get_inputs()` and
`get_init_inputs()` read when they are called. The shapes of the current release are GPU-sized
(a 2048 x 1048576 matrix), far beyond what an interpreter executes in minutes, so the judge can
divide every such constant by one power of two, chosen so the largest input, init input or
reference output holds at most `budget` elements. Only the constants themselves are rewritten, in
place, in the source text (same line numbers): a size computed from them
(`sum_tensor_shape = (out_channels, 1, 1, 1)`) is recomputed from the new values. An answer that
keeps its own copy of the task's constants (with the same value) gets the same rewrite, so a GPU
run of the two rewritten sources checks the very same shapes.

What scaling keeps and what it changes: a size that is not a multiple of 16 stays so (with the same
residue), so a missing mask on a ragged edge is still exercised; the block sizes an answer chose
are untouched, so at the smaller shape fewer program instances run, loops over a dimension run
fewer iterations, and a path the answer takes only at its real shape (a split-K branch, an unmasked
main loop) may not run at all. An out-of-bounds access that only the real shape reaches is missed;
one the smaller shape reaches is real at the smaller shape. An answer that hard-codes the real
shape (a block equal to the dimension, a constant in place of `x.shape[1]`) is right at the real
shape and may be wrong at the smaller one: such a verdict says the answer is not general, not that
it fails KernelBench's own check. Constants below `MIN_SIZE` (axes, kernel sizes, strides, heads,
small channel counts) are left alone, and none goes below `FLOOR`.
"""

from __future__ import annotations

import ast
import contextlib
from typing import Any

MIN_SIZE = 32
FLOOR = 16
BUDGET = 1 << 17
Size = int | tuple[int, ...] | list[int]


def _constant(node: ast.expr) -> Any:
    """The value of an expression made of literals and arithmetic only, else None."""
    if any(isinstance(n, (ast.Name, ast.Call, ast.Attribute)) for n in ast.walk(node)):
        return None
    try:
        # no name, call or attribute reaches here: literals and arithmetic only
        return eval(compile(ast.Expression(node), "<size>", "eval"), {"__builtins__": {}})
    except Exception:  # noqa: BLE001
        return None


def _is_size(value: Any) -> bool:
    if type(value) is int:
        return value >= MIN_SIZE
    if isinstance(value, (tuple, list)) and value and all(type(v) is int for v in value):
        return any(v >= MIN_SIZE for v in value)
    return False


def _assignments(source: str) -> list[tuple[str, ast.expr]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if isinstance(target, ast.Name):
            found.append((target.id, value))
        elif (  # `height, width = 256, 256`
            isinstance(target, ast.Tuple)
            and isinstance(value, ast.Tuple)
            and len(target.elts) == len(value.elts)
            and all(isinstance(t, ast.Name) for t in target.elts)
        ):
            names = [t.id for t in target.elts if isinstance(t, ast.Name)]
            found += list(zip(names, value.elts, strict=True))
    return found


SIZE_CALLS = {"randn", "rand", "zeros", "ones", "empty"}  # every positional argument is a size


def _literal_sites(source: str) -> list[ast.Constant]:
    """The integer literals that are sizes but have no module-level name: in `get_inputs` and
    `get_init_inputs` (`m = 2048`, `torch.randn(32, 4096)`, a shape tuple) and among the default
    arguments of `Model.__init__` / `ModelNew.__init__` (`in_features: int = 4096`). The bounds of
    `torch.randint` and any other scalar stay as they are."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    sites: dict[int, ast.Constant] = {}

    def take(node: ast.AST | None) -> None:
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value >= MIN_SIZE:
            sites[id(node)] = node

    for top in tree.body:
        if isinstance(top, ast.FunctionDef) and top.name in ("get_inputs", "get_init_inputs"):
            for node in ast.walk(top):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    take(node.value)
                elif isinstance(node, (ast.Tuple, ast.List)):
                    for elt in node.elts:
                        take(elt)
                elif isinstance(node, ast.Call):
                    callee = node.func
                    name = callee.attr if isinstance(callee, ast.Attribute) else ""
                    name = callee.id if isinstance(callee, ast.Name) else name
                    if name in SIZE_CALLS:
                        for arg in node.args:
                            take(arg)
        elif isinstance(top, ast.ClassDef) and top.name in ("Model", "ModelNew"):
            for item in top.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    for default in [*item.args.defaults, *item.args.kw_defaults]:
                        take(default)
    return list(sites.values())


def roots(source: str) -> dict[str, Size]:
    """The sizes of a source: module-level constants by name (the last assignment of each name
    wins), and the literal sizes of `_literal_sites` by value, under the key `#<value>`: one
    literal value is one size wherever it is written, so `torch.randn(32, 4096)` in `get_inputs`
    and `in_features: int = 4096` in `Model.__init__` stay equal."""
    found: dict[str, Size] = {}
    for name, node in _assignments(source):
        value = _constant(node)
        if _is_size(value):
            found[name] = value
        else:
            found.pop(name, None)
    for site in _literal_sites(source):
        assert isinstance(site.value, int)  # `_literal_sites` keeps integers only
        found[f"#{site.value}"] = site.value
    return found


def shrink_int(value: int, factor: int) -> int:
    if value < MIN_SIZE:
        return value
    new = max(FLOOR, value // factor)
    if value % 16 and not new % 16:  # a ragged size stays ragged, with the same residue
        new += value % 16
    return new


def shrink(value: Size, factor: int) -> Size:
    if isinstance(value, int):
        return shrink_int(value, factor)
    return type(value)(shrink_int(v, factor) for v in value)


def rewrite(source: str, values: dict[str, Size], only_if: dict[str, Size] | None = None) -> str:
    """The source with the constant right-hand side of each named size replaced, in place (a
    multi-line one keeps its line count). With `only_if`, a name is rewritten only where its
    constant equals the value given there (an answer's own copy of the task's sizes)."""
    if not values:
        return source
    lines = source.splitlines(keepends=True)
    edits: list[tuple[ast.expr, str]] = []
    for name, node in _assignments(source):
        if name not in values:
            continue
        value = _constant(node)
        if not _is_size(value) or (only_if is not None and only_if.get(name) != value):
            continue
        edits.append((node, repr(values[name])))
    for site in _literal_sites(source):
        assert isinstance(site.value, int)
        key = f"#{site.value}"
        if key in values and (only_if is None or key in only_if):
            edits.append((site, repr(values[key])))
    for node, text in sorted(edits, key=lambda e: (e[0].lineno, e[0].col_offset), reverse=True):
        assert node.end_lineno is not None and node.end_col_offset is not None
        first, last = node.lineno - 1, node.end_lineno - 1
        head = lines[first].encode()[: node.col_offset].decode(errors="replace")
        tail = lines[last].encode()[node.end_col_offset :].decode(errors="replace")
        padding = "\n" * (last - first)
        lines[first : last + 1] = [head + text + padding + tail]
    return "".join(lines)


def _meta_mode() -> Any:
    import torch
    from torch.overrides import TorchFunctionMode

    factories = {
        "randn", "rand", "randint", "randperm", "zeros", "ones", "empty", "full", "arange",
        "linspace", "eye", "tensor", "normal", "rand_like", "randn_like", "randint_like",
        "zeros_like", "ones_like", "empty_like", "full_like", "empty_strided",
    }  # fmt: skip

    class Meta(TorchFunctionMode):
        """Every tensor the task makes is a meta tensor: shapes without storage."""

        def __torch_function__(
            self, func: Any, types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            kwargs = dict(kwargs or {})
            name = getattr(func, "__name__", "")
            if name in factories:
                kwargs["device"] = "meta"
            if name in ("cuda", "cpu") and args and isinstance(args[0], torch.Tensor):
                return args[0]
            if name == "to" and args and isinstance(args[0], torch.Tensor):
                kwargs.pop("device", None)
                args = tuple(a for a in args if not isinstance(a, (str, torch.device)))
            return func(*args, **kwargs)

    return Meta()


def footprint(source: str) -> int:
    """Elements of the largest input, init input or reference output of this task source,
    computed on meta tensors (nothing is allocated). Raises when the inputs cannot be made."""
    import torch

    namespace: dict[str, Any] = {}
    exec(source, namespace)  # noqa: S102  the task, inside the sandbox
    largest = 0

    def count(items: Any) -> None:
        nonlocal largest
        for item in items if isinstance(items, (list, tuple)) else [items]:
            if isinstance(item, torch.Tensor):
                largest = max(largest, item.numel())

    with _meta_mode():
        init = namespace["get_init_inputs"]()
        inputs = namespace["get_inputs"]()
        count(init)
        count(inputs)
        # an op without a meta kernel: the inputs decide alone
        with contextlib.suppress(Exception), torch.no_grad():
            count(namespace["Model"](*init)(*inputs))
    return largest


def parameters(source: str) -> int:
    """Elements of the reference model's parameters and buffers (on meta tensors): what the
    judge's full-shape pass would also have to hold, next to the inputs."""
    import torch

    namespace: dict[str, Any] = {}
    exec(source, namespace)  # noqa: S102
    with _meta_mode():
        model = namespace["Model"](*namespace["get_init_inputs"]())
    tensors = [*model.parameters(), *model.buffers()] if isinstance(model, torch.nn.Module) else []
    return sum(t.numel() for t in tensors)


def scaled(source: str, budget: int = BUDGET) -> tuple[dict[str, Size], dict[str, Any]]:
    """(the new sizes, a note on how they were chosen). No new sizes when the task fits."""
    base = roots(source)
    note: dict[str, Any] = {"budget": budget}
    try:
        note["footprint_full"] = footprint(source)
    except Exception as exc:  # noqa: BLE001
        note["footprint_error"] = f"{type(exc).__name__}: {exc}"[:200]
    with contextlib.suppress(Exception):  # not needed for the scaling
        note["params_full"] = parameters(source)
    if not base or note.get("footprint_full", budget + 1) <= budget:
        note["factor"] = 1
        return {}, note
    factor, values = 1, dict(base)
    while factor < 1 << 20:
        factor *= 2
        before, values = values, {name: shrink(value, factor) for name, value in base.items()}
        if values == before:  # every size is at its floor
            break
        if "footprint_full" not in note:  # sizes unknown: bring every size down to 512
            if max(max(v) if isinstance(v, (tuple, list)) else v for v in values.values()) <= 512:
                break
            continue
        try:
            note["footprint"] = footprint(rewrite(source, values))
        except Exception as exc:  # noqa: BLE001
            note["footprint_error"] = f"{type(exc).__name__}: {exc}"[:200]
            break
        if note["footprint"] <= budget:
            break
    note["factor"] = factor
    return values, note
