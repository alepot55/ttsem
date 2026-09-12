"""Level-2 checks: the layout properties that make the level-1 array right.

Level 1 says what a kernel computes; these checks say that each thread holds the
part of it the IR claims. `check(op, args, results, layout_of)` runs the checks
that apply to one op and raises `LayoutViolation` on the first failure.
`layout_of` maps a `Type` to its `LinearLayout`, or to None when the encoding is
outside the port (which the caller records as a coverage gap, never a pass).

The four checks implemented here are the ones whose failures level 1 cannot see:

1. `ttg.convert_layout` moves elements between (register, lane, warp)
   coordinates without changing the array, so both layouts must describe the
   same shape and each must be a bijection on its non-broadcast part.
2. `tt.gather` marked `efficient_layout` must satisfy the warp-locality
   predicate the lowering asserts (`GatherLoweringHelper::isWarpLocal`).
3. a shared layout must map offsets to logical indices one to one, otherwise a
   `ttg.local_store` followed by a `ttg.local_load` is not the identity.
4. `tt.reduce` and `tt.expand_dims` relate a tensor and its slice along an axis,
   and the encodings must say so.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ttsem.ir_types import Op, Type
from ttsem.layouts import Slice, parse_encoding
from ttsem.linear_layout import LayoutError, LinearLayout

LayoutOf = Callable[[Type], "LinearLayout | None"]

K_LANE = "lane"
K_WARP = "warp"
K_BLOCK = "block"

# Above this many offsets the shared round trip is checked algebraically instead
# of by enumerating the permutation.
MAX_ENUMERATED_OFFSETS = 1 << 16


class LayoutViolation(Exception):
    """A layout does not have the property the op's lowering relies on."""

    def __init__(self, op_name: str, reason: str) -> None:
        super().__init__(f"{op_name}: {reason}")
        self.op_name = op_name
        self.reason = reason


def check(op: Op, args: Sequence[object], results: Sequence[object], layout_of: LayoutOf) -> None:
    """Run every level-2 check that applies to `op`."""
    checker = _CHECKS.get(op.name)
    if checker is not None:
        checker(op, args, results, layout_of)


def has_check(op_name: str) -> bool:
    """Whether this op name carries a level-2 check, so coverage is measurable."""
    return op_name in _CHECKS


def checked_op_names() -> list[str]:
    """Every op name that carries a level-2 check."""
    return sorted(_CHECKS)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def drop_broadcast(layout: LinearLayout) -> LinearLayout:
    """The layout with every basis that moves nothing removed, in every in-dim."""
    for dim in layout.in_dim_names():
        layout = layout.remove_zero_bases_along_dim(dim)
    return layout


def is_bijective_ignoring_broadcast(layout: LinearLayout) -> bool:
    """True when the layout covers the tensor and no two kept bases collide.

    This is the condition `LinearEncodingAttr::verify` states as "after removing
    broadcast bases the layout must be bijective".
    """
    reduced = drop_broadcast(layout)
    injective = all(mask == 0 for mask in reduced.free_variable_masks().values())
    return injective and reduced.is_surjective()


def _shape_of(layout: LinearLayout) -> dict[str, int]:
    return dict(layout.out_dims)


# --------------------------------------------------------------------------
# 1. convert_layout
# --------------------------------------------------------------------------


def _check_convert_layout(
    op: Op, args: Sequence[object], results: Sequence[object], layout_of: LayoutOf
) -> None:
    src = layout_of(op.operand_types[0]) if op.operand_types else None
    dst = layout_of(op.result_types[0]) if op.result_types else None
    if src is None or dst is None:
        return
    if _shape_of(src) != _shape_of(dst):
        raise LayoutViolation(
            op.name, f"logical shapes differ: {_shape_of(src)} against {_shape_of(dst)}"
        )
    for name, layout in (("source", src), ("destination", dst)):
        if not is_bijective_ignoring_broadcast(layout):
            raise LayoutViolation(
                op.name,
                f"the {name} layout is not a bijection on its non-broadcast part, so some "
                f"element is held twice or by nobody",
            )
    try:
        src.invert_and_compose(dst)
    except LayoutError as exc:
        raise LayoutViolation(
            op.name, f"no change of basis takes the source to the destination: {exc}"
        ) from exc


# --------------------------------------------------------------------------
# 2. gather warp locality
# --------------------------------------------------------------------------


def is_gather_warp_local(src: LinearLayout, idx: LinearLayout, axis: int, rank: int) -> str | None:
    """Port of `GatherLoweringHelper::isWarpLocal`; returns the failing reason.

    The gather can be done inside a warp when every column along the gather axis
    of both tensors is owned by one warp, and the same lanes own the column in
    the source and in the index tensor.
    """
    gather_dim = f"dim{axis}"
    other = [f"dim{d}" for d in range(rank) if d != axis]
    for name, layout in (("source", src), ("index", idx)):
        if not layout.sublayout_is_zero([K_BLOCK, K_WARP], [gather_dim]):
            return f"the {name} layout moves {gather_dim} with the warp, so a column spans warps"
    if src.sublayout([K_BLOCK, K_WARP], other) != idx.sublayout([K_BLOCK, K_WARP], other):
        return "source and index put the same column in different warps"
    if src.sublayout([K_LANE], other) != idx.sublayout([K_LANE], other):
        return "source and index put the same column in different lanes"
    return None


def _check_gather(
    op: Op, args: Sequence[object], results: Sequence[object], layout_of: LayoutOf
) -> None:
    if not op.attrs.get("efficient_layout"):
        return
    if len(op.operand_types) < 2:
        return
    src = layout_of(op.operand_types[0])
    idx = layout_of(op.operand_types[1])
    if src is None or idx is None:
        return
    axis = int(op.attrs.get("axis", 0))
    rank = len(op.operand_types[0].shape or ())
    reason = is_gather_warp_local(src, idx, axis, rank)
    if reason is not None:
        raise LayoutViolation(op.name, f"gather on axis {axis} is not warp local: {reason}")


# --------------------------------------------------------------------------
# 3. shared memory round trip
# --------------------------------------------------------------------------


def shared_offsets_are_a_permutation(shared: LinearLayout) -> bool:
    """True when the shared layout maps offsets onto the elements evenly.

    A `ttg.local_store` writes the element of each logical index at the offset
    the inverse of this map assigns, and a `ttg.local_load` reads it back, so
    the round trip is the identity exactly when every element has an offset and
    no element is starved by another taking its slot, whatever the swizzle does
    inside. For an unpadded layout that is a permutation of the element range;
    an `fp4Padded` layout has a padded offset beside every real one, so the map
    is the same permutation repeated a fixed number of times, and the check is
    that the number is the same for every element.
    """
    total_in = shared.total_in_dim_size()
    total_out = shared.total_out_dim_size()
    if total_in % total_out:
        return False
    if total_in > MAX_ENUMERATED_OFFSETS:
        return is_bijective_ignoring_broadcast(shared)
    counts: dict[int, int] = {}
    for flat_in in range(total_in):
        image = _flat_image(shared, flat_in)
        counts[image] = counts.get(image, 0) + 1
    per_element = total_in // total_out
    return len(counts) == total_out and all(n == per_element for n in counts.values())


def _flat_image(shared: LinearLayout, flat_in: int) -> int:
    """The layout's output at the `flat_in`-th input, flattened minor-to-major."""
    ins, rest = {}, flat_in
    for dim in shared.in_dim_names():
        size = shared.get_in_dim_size(dim)
        ins[dim] = rest % size
        rest //= size
    out, stride = 0, 1
    for value, size in zip(shared.apply(ins).values(), shared.out_dims.values(), strict=True):
        out += value * stride
        stride *= size
    return out


def _check_shared_round_trip(
    op: Op, args: Sequence[object], results: Sequence[object], layout_of: LayoutOf
) -> None:
    position = _MEMDESC_OPERAND[op.name]
    if position >= len(op.operand_types):
        return
    shared = layout_of(op.operand_types[position])
    if shared is None:
        return
    if not shared_offsets_are_a_permutation(shared):
        raise LayoutViolation(
            op.name,
            "the shared layout does not map offsets one to one onto the elements, so a "
            "local_store followed by a local_load is not the identity",
        )


_MEMDESC_OPERAND = {"ttg.local_load": 0, "ttg.local_store": 1}


# --------------------------------------------------------------------------
# 4. slice encodings on reduce and expand_dims
# --------------------------------------------------------------------------


def _encoding(ty: Type | None):
    if ty is None or not ty.encoding or ty.shape is None:
        return None
    try:
        return parse_encoding(ty.encoding)
    except ValueError:
        return None


def _check_reduce(
    op: Op, args: Sequence[object], results: Sequence[object], layout_of: LayoutOf
) -> None:
    axis = int(op.attrs.get("axis", 0))
    for operand_ty, result_ty in zip(op.operand_types, op.result_types, strict=False):
        parent, sliced = _encoding(operand_ty), _encoding(result_ty)
        if parent is None or sliced is None:
            continue
        _expect_slice(op.name, sliced, parent, axis, "the result of a reduce")


def _check_expand_dims(
    op: Op, args: Sequence[object], results: Sequence[object], layout_of: LayoutOf
) -> None:
    axis = int(op.attrs.get("axis", 0))
    if not op.operand_types or not op.result_types:
        return
    sliced, parent = _encoding(op.operand_types[0]), _encoding(op.result_types[0])
    if parent is None or sliced is None:
        return
    _expect_slice(op.name, sliced, parent, axis, "the operand of an expand_dims")


def _expect_slice(op_name: str, sliced, parent, axis: int, what: str) -> None:
    if not isinstance(sliced, Slice):
        raise LayoutViolation(op_name, f"{what} is not a slice encoding: {type(sliced).__name__}")
    if sliced.dim != axis:
        raise LayoutViolation(op_name, f"{what} slices dim {sliced.dim}, not the axis {axis}")
    if sliced.parent != parent:
        raise LayoutViolation(op_name, f"{what} slices a different layout from the other side")


_CHECKS: dict[str, Callable[[Op, Sequence[object], Sequence[object], LayoutOf], None]] = {
    "ttg.convert_layout": _check_convert_layout,
    "tt.gather": _check_gather,
    "ttg.local_load": _check_shared_round_trip,
    "ttg.local_store": _check_shared_round_trip,
    "tt.reduce": _check_reduce,
    "tt.expand_dims": _check_expand_dims,
}
