"""Level 2: the level-1 interpreter with the layouts read back on.

`LayoutInterp` runs exactly what `Interp` runs, op for op and value for value,
and then asks `layout_checks.check` whether each op's layouts have the property
its lowering relies on. Level 1 does not move: this is a subclass, the values
stay numpy arrays, and `ops.py` never learns that layouts exist.

An encoding the port cannot build is recorded in `layout_gaps` and the op is
skipped. That is a gap in coverage, never a silent pass: the two numbers a run
reports are the violations it raised and the encodings it could not read.
"""

from __future__ import annotations

from collections import Counter

from ttsem import mlir
from ttsem.interp import Interp
from ttsem.ir_types import Module, Op, Type
from ttsem.layout_checks import LayoutOf, LayoutViolation, check, has_check
from ttsem.layouts import EncodingParseError, drop_pipelining_dims, parse_encoding, to_linear_layout
from ttsem.linear_layout import LayoutError, LinearLayout
from ttsem.memory import Memory
from ttsem.values import Value

NUM_WARPS = "ttg.num-warps"
THREADS_PER_WARP = "ttg.threads-per-warp"


def module_attrs(text: str) -> dict[str, object]:
    """The attributes of the top-level `builtin.module`, which `mlir.parse` drops."""
    parser = mlir.Parser(mlir.tokenize(text), text)
    parser.parse_leading_aliases()
    return parser.parse_op().attrs


def launch_config(text: str) -> tuple[int | None, int | None]:
    """`(num_warps, threads_per_warp)` from the module attributes of a generic dump."""
    attrs = module_attrs(text)
    num_warps = attrs.get(NUM_WARPS)
    threads = attrs.get(THREADS_PER_WARP)
    return (
        int(num_warps) if isinstance(num_warps, int) else None,
        int(threads) if isinstance(threads, int) else None,
    )


class LayoutInterp(Interp):
    """`Interp` plus the level-2 checks, driven by the module's launch config."""

    def __init__(
        self,
        module: Module,
        memory: Memory,
        num_programs: tuple[int, int, int] = (1, 1, 1),
        num_warps: int | None = None,
        threads_per_warp: int | None = None,
    ) -> None:
        super().__init__(module, memory, num_programs)
        self.num_warps = num_warps
        self.threads_per_warp = threads_per_warp
        self.layout_gaps: dict[str, str] = {}
        self.checked: Counter[str] = Counter()
        self._layouts: dict[tuple[str, tuple[int, ...]], LinearLayout | None] = {}
        self._results: list[Value] = []

    # ---------------------------------------------------------------- layouts

    def layout_of(self, ty: Type | None) -> LinearLayout | None:
        """The linear layout of a type, built once per (encoding, shape) pair."""
        if ty is None or not ty.encoding or ty.shape is None:
            return None
        key = (ty.encoding, tuple(ty.shape))
        if key in self._layouts:
            return self._layouts[key]
        self._layouts[key] = self._build(ty.encoding, tuple(ty.shape))
        return self._layouts[key]

    def _build(self, encoding: str, shape: tuple[int, ...]) -> LinearLayout | None:
        try:
            parsed = parse_encoding(encoding)
            return to_linear_layout(
                parsed,
                drop_pipelining_dims(shape, parsed),
                self.num_warps,
                self.threads_per_warp,
            )
        except (EncodingParseError, LayoutError) as exc:
            self.layout_gaps.setdefault(encoding, str(exc))
            return None

    # ---------------------------------------------------------------- ops

    def bind_results(self, op: Op, results: list[Value]) -> None:
        super().bind_results(op, results)
        self._results = list(results)

    def eval_op(self, op: Op) -> None:
        args = [self.value(name) for name in op.operands]
        super().eval_op(op)
        self.check_op(op, args, self._results)

    def check_op(self, op: Op, args: list[Value], results: list[Value]) -> None:
        """Run the level-2 checks for one op, noting it as covered."""
        try:
            check(op, args, results, self.layout_of)
        except LayoutViolation as violation:
            violation.add_note(f"while checking {(op.text or op.name)[:300]}")
            raise
        if has_check(op.name):
            self.checked[op.name] += 1


def walk_ops(module: Module):
    """Every op of the module, regions included, in program order."""
    stack: list[Op] = list(reversed(module.ops))
    while stack:
        op = stack.pop()
        yield op
        nested = [inner for region in op.regions for block in region.blocks for inner in block.ops]
        stack.extend(reversed(nested))


def check_module(
    module: Module,
    num_warps: int | None = None,
    threads_per_warp: int | None = None,
) -> tuple[list[LayoutViolation], dict[str, str], Counter[str]]:
    """Run only the layout checks over a module, without executing anything.

    Every check implemented today reads types, not values, so a whole dump can
    be checked without a `Memory` and without the level-1 handlers. Returns the
    violations, the encodings that could not be built, and the op names covered.
    """
    interp = LayoutInterp(module, Memory(), num_warps=num_warps, threads_per_warp=threads_per_warp)
    violations: list[LayoutViolation] = []
    layout_of: LayoutOf = interp.layout_of
    for op in walk_ops(module):
        try:
            check(op, [], [], layout_of)
        except LayoutViolation as violation:
            violations.append(violation)
        else:
            if has_check(op.name):
                interp.checked[op.name] += 1
    return violations, interp.layout_gaps, interp.checked
