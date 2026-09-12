"""TritonGPU layout encodings and their linear layouts.

`parse_encoding` reads the printed form of a `#ttg.*` attribute, the text that
`mlir.Type.encoding` keeps verbatim, into a dataclass.  `to_linear_layout` maps
an encoding plus a tensor shape to a `LinearLayout` over the input dims
`register`, `lane`, `warp`, `block` (distributed layouts) or `offset` (shared
layouts) and the output dims `dim0..dimN`.

This is a port of `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp` and the
parts of `lib/Tools/LayoutUtils.cpp` and `lib/Dialect/TritonGPU/IR/Dialect.cpp`
it depends on, restricted to the NVIDIA encodings: blocked, slice, dot_op over
a blocked or an nvidia_mma parent, nvidia_mma v2 and v3, linear,
swizzled_shared and nvmma_shared.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ttsem.linear_layout import LayoutError, LinearLayout, next_pow2

K_REGISTER = "register"
K_LANE = "lane"
K_WARP = "warp"
K_BLOCK = "block"
K_OFFSET = "offset"


# --------------------------------------------------------------------------
# Encodings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Blocked:
    size_per_thread: tuple[int, ...]
    threads_per_warp: tuple[int, ...]
    warps_per_cta: tuple[int, ...]
    order: tuple[int, ...]
    cga: LinearLayout

    @property
    def rank(self) -> int:
        return len(self.order)

    def rep_order(self) -> tuple[int, ...]:
        return self.order


@dataclass(frozen=True)
class NvidiaMma:
    version_major: int
    version_minor: int
    warps_per_cta: tuple[int, ...]
    instr_shape: tuple[int, ...]
    cga: LinearLayout

    @property
    def rank(self) -> int:
        return len(self.warps_per_cta)

    def is_ampere(self) -> bool:
        return self.version_major == 2

    def is_hopper(self) -> bool:
        return self.version_major == 3

    def rep_order(self) -> tuple[int, ...]:
        return get_matrix_order(self.rank, row_major=True)

    def rep_order_for_operand(self, op_idx: int) -> tuple[int, ...]:
        return get_order_for_dot_operand(op_idx, self.rank, k_contig=True)


@dataclass(frozen=True)
class DotOperand:
    op_idx: int
    parent: Encoding
    k_width: int

    @property
    def rank(self) -> int:
        return self.parent.rank

    def rep_order(self) -> tuple[int, ...]:
        if isinstance(self.parent, NvidiaMma):
            return self.parent.rep_order_for_operand(self.op_idx)
        if isinstance(self.parent, Blocked):
            return self.parent.order
        raise LayoutError(f"dot_op parent {type(self.parent).__name__} is not supported")


@dataclass(frozen=True)
class Slice:
    dim: int
    parent: Encoding

    @property
    def rank(self) -> int:
        return self.parent.rank - 1


@dataclass(frozen=True)
class Linear:
    layout: LinearLayout

    @property
    def rank(self) -> int:
        return len(self.layout.out_dims)

    def rep_order(self) -> tuple[int, ...]:
        default = tuple(range(self.rank - 1, -1, -1))
        return order_per_dim(self.layout, K_REGISTER, default)


@dataclass(frozen=True)
class SwizzledShared:
    vec: int
    per_phase: int
    max_phase: int
    order: tuple[int, ...]
    cga: LinearLayout

    @property
    def rank(self) -> int:
        return len(self.order)


@dataclass(frozen=True)
class NvmmaShared:
    swizzling_byte_width: int
    transposed: bool
    element_bit_width: int
    fp4_padded: bool
    cga: LinearLayout

    @property
    def rank(self) -> int:
        return len(self.cga.out_dims)

    def vec(self) -> int:
        return 1 if self.swizzling_byte_width == 0 else 128 // self.element_bit_width

    def per_phase(self) -> int:
        return 1 if self.swizzling_byte_width == 0 else 128 // self.swizzling_byte_width

    def max_phase(self) -> int:
        return 1 if self.swizzling_byte_width == 0 else self.swizzling_byte_width // 16


Encoding = Blocked | NvidiaMma | DotOperand | Slice | Linear | SwizzledShared | NvmmaShared


# --------------------------------------------------------------------------
# CGA layouts
# --------------------------------------------------------------------------


def standard_out_dim_names(rank: int) -> list[str]:
    return [f"dim{i}" for i in range(rank)]


def cga_one_cta(rank: int) -> LinearLayout:
    return LinearLayout({K_BLOCK: []}, standard_out_dim_names(rank))


def cga_from_block_bases(bases: Sequence[Sequence[int]]) -> LinearLayout:
    """The CGA layout of a printed `CGALayout = [[...], ...]` field.

    Sizes are the smallest that fit the bases, and surjectivity is not required:
    a `CTAsPerCGA` larger than its `CTASplitNum` gives broadcast bases.
    """
    rank = len(bases[0])
    sizes = [1] * rank
    for basis in bases:
        for i, value in enumerate(basis):
            sizes[i] = max(sizes[i], next_pow2(value))
    out_dims = list(zip(standard_out_dim_names(rank), sizes, strict=True))
    return LinearLayout({K_BLOCK: [list(b) for b in bases]}, out_dims, False)


def cga_from_split_params(
    ctas_per_cga: Sequence[int], cta_split_num: Sequence[int], cta_order: Sequence[int]
) -> LinearLayout:
    """The CGA layout that `CGAEncodingAttr::fromSplitParams` builds."""
    rank = len(cta_order)
    names = standard_out_dim_names(rank)
    layout = LinearLayout.empty()
    for i in range(rank):
        dim = cta_order[i]
        split = cta_split_num[dim]
        total = ctas_per_cga[dim]
        if total % split:
            raise LayoutError("CTAsPerCGA must be a multiple of CTASplitNum")
        layout = layout * (
            LinearLayout.identity1D(split, K_BLOCK, names[dim])
            * LinearLayout.zeros1D(total // split, K_BLOCK, names[dim])
        )
    return layout.transpose_outs(names)


def bases_per_dim(
    layout: LinearLayout, dim_name: str, rank: int, skip_broadcast: bool = True
) -> tuple[int, ...]:
    """Per-out-dim extent covered by the bases of `dim_name` (`basesPerDimImpl`)."""
    bases = layout.bases[dim_name]
    if not bases:
        return (1,) * rank
    out = [1] * rank
    non_zero_idx = 0
    for basis in bases:
        found = next((i for i, v in enumerate(basis) if v != 0), None)
        if found is not None:
            non_zero_idx = found
            out[non_zero_idx] *= 2
        elif not skip_broadcast:
            out[non_zero_idx] *= 2
    return tuple(out)


def order_per_dim(
    layout: LinearLayout, dim_name: str, default_order: Sequence[int]
) -> tuple[int, ...]:
    """`orderPerDimImpl`: dims in the order the bases of `dim_name` walk them."""
    order: list[int] = []
    for basis in layout.bases[dim_name]:
        found = next((i for i, v in enumerate(basis) if v != 0), None)
        if found is not None and found not in order:
            order.append(found)
    for i in default_order:
        if i not in order:
            order.append(i)
    return tuple(order)


def cta_split_num(cga: LinearLayout) -> tuple[int, ...]:
    return bases_per_dim(cga, K_BLOCK, len(cga.out_dims))


def get_shape_per_cta(split_num: Sequence[int], shape: Sequence[int]) -> list[int]:
    rank = len(shape)
    splits = list(split_num)
    if len(splits) <= rank:
        splits = [1] * (rank - len(splits)) + splits
    else:
        splits = splits[len(splits) - rank :]
    return [shape[i] // min(shape[i], splits[i]) for i in range(rank)]


def cga_of(encoding: Encoding) -> LinearLayout:
    """The CGA layout of an encoding, following `getCGALayout`."""
    if isinstance(encoding, (Blocked, NvidiaMma, SwizzledShared, NvmmaShared)):
        return encoding.cga
    if isinstance(encoding, Slice):
        return remove_standard_dim(cga_of(encoding.parent), encoding.dim)
    if isinstance(encoding, DotOperand):
        parent = cga_of(encoding.parent)
        rank = len(parent.out_dims)
        broadcast = rank - 1 if encoding.op_idx == 0 else rank - 2
        return parent.resize_out_dim(parent.out_dim_names()[broadcast], 1)
    if isinstance(encoding, Linear):
        return LinearLayout(
            {K_BLOCK: encoding.layout.bases[K_BLOCK]},
            list(encoding.layout.out_dims.items()),
            False,
        )
    raise LayoutError(f"no CGA layout for {type(encoding).__name__}")


# --------------------------------------------------------------------------
# Layout utilities (`lib/Tools/LayoutUtils.cpp`)
# --------------------------------------------------------------------------


def get_matrix_order(rank: int, row_major: bool) -> tuple[int, ...]:
    if rank < 2:
        return (0,) * rank
    order = list(range(rank - 1, -1, -1))
    if not row_major:
        order[0], order[1] = order[1], order[0]
    return tuple(order)


def get_order_for_dot_operand(op_idx: int, rank: int, k_contig: bool) -> tuple[int, ...]:
    if op_idx not in (0, 1):
        raise LayoutError("dot operand index must be 0 or 1")
    return get_matrix_order(rank, row_major=bool(op_idx) != k_contig)


def identity_standard_nd(in_dim: str, shape: Sequence[int], order: Sequence[int]) -> LinearLayout:
    names = standard_out_dim_names(len(shape))
    layout = LinearLayout.empty()
    for i in range(len(shape)):
        dim = order[i]
        layout = layout * LinearLayout.identity1D(shape[dim], in_dim, names[dim])
    return layout


def ensure_layout_not_smaller_than(layout: LinearLayout, shape: Mapping[str, int]) -> LinearLayout:
    if not shape:
        return layout
    minor = layout.in_dim_names()[0]
    result = layout
    for out_dim in layout.out_dim_names():
        actual = layout.get_out_dim_size(out_dim)
        desired = shape[out_dim]
        result = result * LinearLayout.identity1D(desired // actual, minor, out_dim)
    return result


def ensure_layout_not_larger_than(
    layout: LinearLayout, shape: Mapping[str, int], broadcast_registers: bool = True
) -> LinearLayout:
    if not shape:
        return layout
    sizes = [min(size, shape[dim]) for dim, size in layout.out_dims.items()]
    bases = {}
    for in_dim, in_bases in layout.bases.items():
        drop = (not broadcast_registers) and in_dim == K_REGISTER
        kept = []
        for basis in in_bases:
            was_zero = all(v == 0 for v in basis)
            clipped = [0 if v >= sizes[i] else v for i, v in enumerate(basis)]
            if not drop or was_zero or any(v != 0 for v in clipped):
                kept.append(clipped)
        bases[in_dim] = kept
    out_dims = list(zip(layout.out_dim_names(), sizes, strict=True))
    return LinearLayout(bases, out_dims, False)


def reshape_layout(layout: LinearLayout, shape: Sequence[int]) -> LinearLayout:
    src = list(reversed(layout.out_dim_names()))
    new_dims = list(reversed(list(zip(standard_out_dim_names(len(shape)), shape, strict=True))))
    return (
        layout.transpose_outs(src)
        .reshape_outs(new_dims)
        .transpose_outs(standard_out_dim_names(len(shape)))
    )


def transpose_linear_layout(layout: LinearLayout, order: Sequence[int]) -> LinearLayout:
    bases = {
        in_dim: [[basis[i] for i in order] for basis in in_bases]
        for in_dim, in_bases in layout.bases.items()
    }
    return LinearLayout(bases, layout.out_dim_names())


def remove_standard_dim(layout: LinearLayout, dim: int) -> LinearLayout:
    rank = len(layout.out_dims)
    dims = layout.out_dim_names()
    if dims != standard_out_dim_names(rank):
        raise LayoutError("remove_standard_dim expects standard out-dim names")
    dims.pop(dim)
    sub = layout.sublayout(layout.in_dim_names(), dims)
    renames = dict(zip(dims, standard_out_dim_names(rank - 1), strict=True))
    return sub.rename_outs(renames)


def combine_cta_cga_with_shape(
    cta_layout: LinearLayout, cga_layout: LinearLayout, shape: Sequence[int]
) -> LinearLayout:
    rank = len(shape)
    names = standard_out_dim_names(rank)
    labeled = dict(zip(names, shape, strict=True))
    cga = ensure_layout_not_larger_than(cga_layout, labeled).transpose_outs(
        cta_layout.out_dim_names()
    )
    cta_shape = {
        dim: max(1, labeled[dim] // cga.get_out_dim_size(dim)) for dim in cta_layout.out_dim_names()
    }
    cta = ensure_layout_not_smaller_than(cta_layout, cta_shape)
    cta = ensure_layout_not_larger_than(cta, cta_shape)
    return (cta * cga).transpose_outs(names)


# --------------------------------------------------------------------------
# Distributed layouts
# --------------------------------------------------------------------------


def blocked_to_linear_layout(enc: Blocked, shape: Sequence[int]) -> LinearLayout:
    cta = (
        identity_standard_nd(K_REGISTER, enc.size_per_thread, enc.order)
        * identity_standard_nd(K_LANE, enc.threads_per_warp, enc.order)
        * identity_standard_nd(K_WARP, enc.warps_per_cta, enc.order)
    )
    combined = combine_cta_cga_with_shape(cta, enc.cga, shape)
    return combined.remove_zero_bases_along_dim(K_REGISTER)


def broadcasted_dot_operand_layout(
    shape: Sequence[int], order: Sequence[int], k_dim: int, in_dim: str
) -> LinearLayout:
    """Lane or warp layout of a dot operand: identity everywhere but along K."""
    names = standard_out_dim_names(len(shape))
    layout = LinearLayout.empty()
    for dim in order:
        if dim == k_dim:
            layout = layout * LinearLayout.zeros1D(shape[dim], in_dim, names[dim])
        else:
            layout = layout * LinearLayout.identity1D(shape[dim], in_dim, names[dim])
    return layout


def nvidia_mma_tile(
    tile_shape: Sequence[int],
    k_width: int,
    order: Sequence[int],
    rep_order: Sequence[int],
) -> LinearLayout:
    """The register and lane layout of one `mma.sync` tile."""
    rank = len(rep_order)
    names = standard_out_dim_names(rank)
    layout = identity_standard_nd(K_REGISTER, [1] * rank, rep_order)
    inner, outer = order[0], order[1]
    m, n = tile_shape[outer], tile_shape[inner]
    if m % 8 or n % (k_width * 4):
        raise LayoutError(f"mma tile {list(tile_shape)} is not divisible for kWidth={k_width}")
    return (
        layout
        * LinearLayout.identity1D(k_width, K_REGISTER, names[inner])
        * LinearLayout.identity1D(4, K_LANE, names[inner])
        * LinearLayout.identity1D(8, K_LANE, names[outer])
        * LinearLayout.identity1D(m // 8, K_REGISTER, names[outer])
        * LinearLayout.identity1D(n // (k_width * 4), K_REGISTER, names[inner])
    )


def nvidia_mma_to_linear_layout(enc: NvidiaMma, shape: Sequence[int]) -> LinearLayout:
    rank = len(shape)
    if rank != enc.rank:
        raise LayoutError("nvidia_mma rank does not match the shape")
    if enc.is_ampere():
        tile_shape = list(enc.instr_shape)
    elif enc.is_hopper():
        tile_shape = [enc.instr_shape[0], enc.instr_shape[1]]
    else:
        raise LayoutError(f"unsupported mma version {enc.version_major}")
    order = get_matrix_order(rank, row_major=True)
    cta = nvidia_mma_tile(tile_shape, 2, order, enc.rep_order())
    warp_order = get_matrix_order(rank, row_major=not enc.is_hopper())
    warps = identity_standard_nd(K_WARP, enc.warps_per_cta, warp_order)
    cta = cta * warps.transpose_outs(cta.out_dim_names())
    return combine_cta_cga_with_shape(cta, enc.cga, shape)


def nvidia_dot_to_linear_layout(enc: DotOperand, shape: Sequence[int]) -> LinearLayout:
    rank = len(shape)
    mma = enc.parent
    assert isinstance(mma, NvidiaMma)
    is_a = enc.op_idx == 0
    tile_shape = [1] * rank
    instr_m = mma.instr_shape[rank - 2]
    # fp64 (m8n8k4) uses a smaller K tile than every other Ampere instruction.
    k_tile_multiplier = 4 if instr_m == 8 else 8
    if is_a:
        tile_shape[rank - 2] = instr_m
        tile_shape[rank - 1] = enc.k_width * k_tile_multiplier
    else:
        if not mma.is_ampere():
            raise LayoutError("Hopper takes the dot rhs through shared memory")
        tile_shape[rank - 2] = enc.k_width * k_tile_multiplier
        tile_shape[rank - 1] = 8
    order = get_order_for_dot_operand(enc.op_idx, rank, k_contig=True)
    cta = nvidia_mma_tile(tile_shape, enc.k_width, order, enc.rep_order())
    k_dim = rank - 1 if is_a else rank - 2
    warp_order = get_matrix_order(rank, row_major=not mma.is_hopper())
    warps = broadcasted_dot_operand_layout(mma.warps_per_cta, warp_order, k_dim, K_WARP)
    cta = cta * warps.transpose_outs(cta.out_dim_names())
    return combine_cta_cga_with_shape(cta, cga_of(enc), shape)


def fma_dot_to_linear_layout(enc: DotOperand, shape: Sequence[int]) -> LinearLayout:
    rank = len(shape)
    blocked = enc.parent
    assert isinstance(blocked, Blocked)
    order = blocked.order
    k_dim = rank - 1 if enc.op_idx == 0 else rank - 2
    thread_size = list(blocked.size_per_thread)
    thread_size[k_dim] = shape[k_dim]
    rep_names = [standard_out_dim_names(rank)[i] for i in blocked.rep_order()]

    registers = identity_standard_nd(K_REGISTER, thread_size, order)
    lanes = broadcasted_dot_operand_layout(blocked.threads_per_warp, order, k_dim, K_LANE)
    warps = broadcasted_dot_operand_layout(blocked.warps_per_cta, order, k_dim, K_WARP)
    cta = (
        registers.transpose_outs(rep_names)
        * lanes.transpose_outs(rep_names)
        * warps.transpose_outs(rep_names)
    )
    return combine_cta_cga_with_shape(cta, cga_of(enc), shape)


def dot_operand_to_linear_layout(enc: DotOperand, shape: Sequence[int]) -> LinearLayout:
    if isinstance(enc.parent, Blocked):
        return fma_dot_to_linear_layout(enc, shape)
    if isinstance(enc.parent, NvidiaMma):
        return nvidia_dot_to_linear_layout(enc, shape)
    raise LayoutError(f"dot_op parent {type(enc.parent).__name__} is not supported")


def slice_to_linear_layout(enc: Slice, shape: Sequence[int]) -> LinearLayout:
    parent_shape = list(shape)
    parent_shape.insert(enc.dim, 1)
    parent = to_linear_layout(enc.parent, parent_shape)
    sliced = remove_standard_dim(parent, enc.dim)
    return sliced.remove_zero_bases_along_dim(K_REGISTER)


def linear_to_linear_layout(enc: Linear, shape: Sequence[int]) -> LinearLayout:
    layout = enc.layout
    canonical = layout.out_dim_names()
    rep_order = enc.rep_order()
    permuted = [canonical[dim] for dim in rep_order]
    named_shape = {canonical[dim]: shape[dim] for dim in rep_order}
    result = layout.transpose_outs(permuted)
    result = ensure_layout_not_smaller_than(result, named_shape)
    result = ensure_layout_not_larger_than(result, named_shape, broadcast_registers=False)
    return result.transpose_outs(canonical)


# --------------------------------------------------------------------------
# Shared layouts
# --------------------------------------------------------------------------


def swizzled_shared_to_linear_layout(enc: SwizzledShared, shape: Sequence[int]) -> LinearLayout:
    per_cta = get_shape_per_cta(cta_split_num(enc.cga), shape)
    rank = len(shape)
    if rank == 1:
        flat = LinearLayout.identity1D(per_cta[0], K_OFFSET, "dim0")
        return combine_cta_cga_with_shape(flat, enc.cga, shape)

    names = standard_out_dim_names(rank)
    col_dim, row_dim = enc.order[0], enc.order[1]
    num_cols, num_rows = per_cta[col_dim], per_cta[row_dim]
    bases: list[list[int]] = []
    col = 1
    while col < num_cols:
        bases.append([0, col])
        col *= 2
    row = 1
    while row < num_rows:
        phase = (row // enc.per_phase) % enc.max_phase
        bases.append([row, (enc.vec * phase) % num_cols])
        row *= 2
    cta = LinearLayout({K_OFFSET: bases}, [names[row_dim], names[col_dim]])
    for i in range(2, rank):
        dim = enc.order[i]
        cta = cta * LinearLayout.identity1D(per_cta[dim], K_OFFSET, names[dim])
    return combine_cta_cga_with_shape(cta, enc.cga, shape)


def core_matrix_linear_layout(enc: NvmmaShared, disable_swizzle: bool) -> LinearLayout:
    """The layout of the single core matrix that tiles an nvmma_shared buffer."""
    tile_rows = 8
    tile_cols = 8 * max(16, enc.swizzling_byte_width) // enc.element_bit_width
    vec, per_phase, max_phase = enc.vec(), enc.per_phase(), enc.max_phase()
    bases: list[list[int]] = []
    col = 1
    while col < tile_cols:
        # An fp4-padded tile stores 8 real and 8 padded offsets per group of 16;
        # the padded ones alias the real coordinates.
        packed = col // 16 * 8 + col % 8 if enc.fp4_padded else col
        bases.append([0, packed])
        col *= 2
    row = 1
    while row < tile_rows:
        if disable_swizzle:
            bases.append([row, 0])
        else:
            padded = vec * ((row // per_phase) % max_phase)
            bases.append([row, padded // 16 * 8 + padded % 8 if enc.fp4_padded else padded])
        row *= 2
    return LinearLayout({K_OFFSET: bases}, standard_out_dim_names(2))


def tma_block_shape_tiled(
    shape_per_cta: Sequence[int],
    element_bit_width: int,
    swizzle_bytes: int,
    fp4_padded: bool,
    transposed: bool,
    packed_size: bool,
) -> list[int]:
    """`getTMABlockShapeTiled`: the block shape a TMA copy actually moves."""
    block = list(shape_per_cta)
    contig = 0 if transposed else len(block) - 1
    if fp4_padded:
        block[contig] *= 2
    block = [min(size, 256) for size in block]
    if swizzle_bytes != 0:
        contig_size = (8 * swizzle_bytes) // element_bit_width
        if block[contig] < contig_size:
            raise LayoutError(
                f"block shape {block[contig]} along dim {contig} is too small for "
                f"a {swizzle_bytes}-byte swizzle (needs {contig_size})"
            )
        block[contig] = contig_size
    if fp4_padded and packed_size:
        block[contig] //= 2
    return block


def _rotate_left(values: Sequence[int]) -> list[int]:
    return list(values[1:]) + [values[0]]


def _nvmma_unswizzled(
    enc: NvmmaShared, shape: Sequence[int], tma_shape: Sequence[int], per_cta: Sequence[int]
) -> LinearLayout:
    rank = len(shape)
    names = standard_out_dim_names(rank)
    block = _rotate_left(tma_shape) if enc.transposed else list(tma_shape)
    layout = LinearLayout.identity1D(block[rank - 1], K_OFFSET, names[rank - 1])
    for i in range(rank - 2, -1, -1):
        layout = layout * LinearLayout.identity1D(block[i], K_OFFSET, names[i])
    if enc.transposed:
        layout = transpose_linear_layout(layout, [rank - 1] + list(range(rank - 1)))
    layout = ensure_layout_not_smaller_than(layout, dict(zip(names, per_cta, strict=True)))
    return combine_cta_cga_with_shape(layout, enc.cga, shape)


def nvmma_shared_to_linear_layout(
    enc: NvmmaShared, shape: Sequence[int], disable_swizzle: bool = False
) -> LinearLayout:
    per_cta = get_shape_per_cta(cta_split_num(enc.cga), shape)
    tma_shape = tma_block_shape_tiled(
        per_cta,
        enc.element_bit_width,
        enc.swizzling_byte_width,
        enc.fp4_padded,
        enc.transposed,
        packed_size=True,
    )
    if enc.swizzling_byte_width == 0:
        return _nvmma_unswizzled(enc, shape, tma_shape, per_cta)

    rank = len(shape)
    if rank < 2:
        raise LayoutError("a swizzled nvmma_shared layout needs rank >= 2")
    collapsed = [1, tma_shape[-1]]
    for size in tma_shape[:-1]:
        collapsed[0] *= size
    if enc.transposed:
        collapsed = [collapsed[1], collapsed[0]]

    tile = core_matrix_linear_layout(enc, disable_swizzle)
    two_d = standard_out_dim_names(2)
    packing = 2 if enc.fp4_padded else 1
    if collapsed[1] * packing < tile.get_out_dim_size(two_d[1]) or collapsed[
        0
    ] < tile.get_out_dim_size(two_d[0]):
        raise LayoutError(f"collapsed shape {collapsed} is smaller than one core matrix")

    layout = ensure_layout_not_smaller_than(tile, dict(zip(two_d, collapsed, strict=True)))
    block = _rotate_left(tma_shape) if enc.transposed else list(tma_shape)
    if layout.total_out_dim_size() != _product(block):
        raise LayoutError("nvmma_shared layout does not tile the TMA block shape")
    layout = reshape_layout(layout, block)
    if enc.transposed:
        layout = transpose_linear_layout(layout, [rank - 1] + list(range(rank - 1)))
    layout = ensure_layout_not_smaller_than(
        layout, dict(zip(standard_out_dim_names(rank), per_cta, strict=True))
    )
    return combine_cta_cga_with_shape(layout, enc.cga, shape)


def _product(values: Sequence[int]) -> int:
    total = 1
    for value in values:
        total *= value
    return total


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def drop_pipelining_dims(shape: Sequence[int], encoding: Encoding) -> list[int]:
    """The layout-ranked suffix of a memdesc shape, as `dropPipeliningDim` does.

    A multi-buffered shared allocation carries its stage count as leading
    dimensions that the shared encoding does not describe, so
    `!ttg.memdesc<2x16x64xbf16, #ttg.nvmma_shared<...>>` lays out `16x64`.
    """
    rank = encoding.rank
    if rank > len(shape):
        raise LayoutError(f"encoding rank {rank} exceeds the shape rank {len(shape)}")
    return list(shape[len(shape) - rank :])


def to_linear_layout(
    encoding: Encoding,
    shape: Sequence[int],
    num_warps: int | None = None,
    threads_per_warp: int | None = None,
) -> LinearLayout:
    """The linear layout of `encoding` over a tensor of `shape`.

    The shape must have the encoding's rank; for a memdesc whose leading
    dimensions are pipelining stages, pass `drop_pipelining_dims(shape, enc)`.

    `num_warps` and `threads_per_warp` come from the module attributes
    `ttg.num-warps` and `ttg.threads-per-warp`; when given they are checked
    against the layout, which is the cheapest way to catch an encoding that was
    parsed against the wrong module.
    """
    if len(shape) != encoding.rank:
        raise LayoutError(
            f"{type(encoding).__name__} has rank {encoding.rank} but the shape is "
            f"{list(shape)}; drop_pipelining_dims trims a memdesc shape"
        )
    layout = _dispatch(encoding, shape)
    if threads_per_warp is not None and layout.has_in_dim(K_LANE):
        if layout.get_in_dim_size(K_LANE) != threads_per_warp:
            raise LayoutError(
                f"layout has {layout.get_in_dim_size(K_LANE)} lanes, module says {threads_per_warp}"
            )
    if num_warps is not None and layout.has_in_dim(K_WARP):
        if layout.get_in_dim_size(K_WARP) != num_warps:
            raise LayoutError(
                f"layout has {layout.get_in_dim_size(K_WARP)} warps, module says {num_warps}"
            )
    return layout


def _dispatch(encoding: Encoding, shape: Sequence[int]) -> LinearLayout:
    if isinstance(encoding, Blocked):
        return blocked_to_linear_layout(encoding, shape)
    if isinstance(encoding, NvidiaMma):
        return nvidia_mma_to_linear_layout(encoding, shape)
    if isinstance(encoding, DotOperand):
        return dot_operand_to_linear_layout(encoding, shape)
    if isinstance(encoding, Slice):
        return slice_to_linear_layout(encoding, shape)
    if isinstance(encoding, Linear):
        return linear_to_linear_layout(encoding, shape)
    if isinstance(encoding, SwizzledShared):
        return swizzled_shared_to_linear_layout(encoding, shape)
    if isinstance(encoding, NvmmaShared):
        return nvmma_shared_to_linear_layout(encoding, shape)
    raise LayoutError(f"unsupported encoding {type(encoding).__name__}")


# --------------------------------------------------------------------------
# Parsing the printed attribute
# --------------------------------------------------------------------------


class EncodingParseError(ValueError):
    """The encoding text is not a layout this module knows how to read."""


def parse_encoding(text: str) -> Encoding:
    """Parse the printed form of a `#ttg.*` layout attribute.

    A memdesc keeps its memory space and its `mutable` flag in the same field
    (`mlir.Type.encoding` of a memdesc is everything after the element type), so
    a trailing comma-separated list of plain attributes and flags is accepted
    and ignored: only the layout is returned.
    """
    value, pos = _parse_value(text, 0)
    pos = _skip_memdesc_tail(text, pos)
    if pos != len(text):
        raise EncodingParseError(f"trailing text in encoding: {text[pos:]!r}")
    if not isinstance(value, tuple) or len(value) != 2:
        raise EncodingParseError(f"not an attribute: {text!r}")
    return _build_encoding(*value)


def _skip_memdesc_tail(text: str, pos: int) -> int:
    pos = _skip_ws(text, pos)
    while pos < len(text) and text[pos] == ",":
        pos = _skip_ws(text, pos + 1)
        if pos < len(text) and text[pos] == "#":
            pos += 1
        _, pos = _parse_word(text, pos)
        pos = _skip_ws(text, pos)
    return pos


def _skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _expect(text: str, pos: int, char: str) -> int:
    pos = _skip_ws(text, pos)
    if pos >= len(text) or text[pos] != char:
        raise EncodingParseError(f"expected {char!r} at offset {pos} of {text!r}")
    return pos + 1


def _parse_value(text: str, pos: int) -> tuple[object, int]:
    pos = _skip_ws(text, pos)
    if pos >= len(text):
        raise EncodingParseError(f"unexpected end of {text!r}")
    char = text[pos]
    if char == "#":
        return _parse_attribute(text, pos)
    if char == "[":
        return _parse_list(text, pos)
    if char == "-" or char.isdigit():
        return _parse_int(text, pos)
    word, pos = _parse_word(text, pos)
    if word in ("true", "false"):
        return word == "true", pos
    return word, pos


def _parse_int(text: str, pos: int) -> tuple[int, int]:
    start = pos
    if text[pos] == "-":
        pos += 1
    while pos < len(text) and text[pos].isdigit():
        pos += 1
    return int(text[start:pos]), pos


def _parse_word(text: str, pos: int) -> tuple[str, int]:
    start = pos = _skip_ws(text, pos)
    while pos < len(text) and (text[pos].isalnum() or text[pos] in "_.-"):
        pos += 1
    if pos == start:
        raise EncodingParseError(f"expected a word at offset {pos} of {text!r}")
    return text[start:pos], pos


def _parse_list(text: str, pos: int) -> tuple[list[object], int]:
    pos = _expect(text, pos, "[")
    items: list[object] = []
    pos = _skip_ws(text, pos)
    if pos < len(text) and text[pos] == "]":
        return items, pos + 1
    while True:
        value, pos = _parse_value(text, pos)
        items.append(value)
        pos = _skip_ws(text, pos)
        if pos < len(text) and text[pos] == ",":
            pos += 1
            continue
        return items, _expect(text, pos, "]")


def _parse_attribute(text: str, pos: int) -> tuple[tuple[str, dict[str, object]], int]:
    pos = _expect(text, pos, "#")
    mnemonic, pos = _parse_word(text, pos)
    pos = _expect(text, pos, "<")
    pos = _expect(text, pos, "{")
    fields: dict[str, object] = {}
    pos = _skip_ws(text, pos)
    if pos < len(text) and text[pos] == "}":
        pos += 1
    else:
        while True:
            key, pos = _parse_word(text, pos)
            pos = _expect(text, pos, "=")
            fields[key], pos = _parse_value(text, pos)
            pos = _skip_ws(text, pos)
            if pos < len(text) and text[pos] == ",":
                pos += 1
                continue
            pos = _expect(text, pos, "}")
            break
    return (mnemonic, fields), _expect(text, pos, ">")


def _ints(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(isinstance(v, int) for v in value):
        raise EncodingParseError(f"expected a list of integers, got {value!r}")
    return tuple(value)  # type: ignore[arg-type]


def _bases(value: object) -> list[list[int]]:
    if not isinstance(value, list):
        raise EncodingParseError(f"expected a list of bases, got {value!r}")
    return [list(_ints(item)) for item in value]


def _nested(value: object) -> Encoding:
    if not isinstance(value, tuple) or len(value) != 2:
        raise EncodingParseError(f"expected a nested attribute, got {value!r}")
    return _build_encoding(*value)


def _cga_field(fields: Mapping[str, object], rank: int) -> LinearLayout:
    if "CGALayout" in fields:
        return cga_from_block_bases(_bases(fields["CGALayout"]))
    return cga_one_cta(rank)


def _build_encoding(mnemonic: str, fields: Mapping[str, object]) -> Encoding:
    builder = _BUILDERS.get(mnemonic)
    if builder is None:
        raise EncodingParseError(f"unsupported layout attribute #{mnemonic}")
    return builder(fields)


def _build_blocked(fields: Mapping[str, object]) -> Blocked:
    order = _ints(fields["order"])
    return Blocked(
        size_per_thread=_ints(fields["sizePerThread"]),
        threads_per_warp=_ints(fields["threadsPerWarp"]),
        warps_per_cta=_ints(fields["warpsPerCTA"]),
        order=order,
        cga=_cga_field(fields, len(order)),
    )


def _build_mma(fields: Mapping[str, object]) -> NvidiaMma:
    warps = _ints(fields["warpsPerCTA"])
    return NvidiaMma(
        version_major=int(fields["versionMajor"]),  # type: ignore[arg-type]
        version_minor=int(fields["versionMinor"]),  # type: ignore[arg-type]
        warps_per_cta=warps,
        instr_shape=_ints(fields["instrShape"]),
        cga=_cga_field(fields, len(warps)),
    )


def _build_dot_op(fields: Mapping[str, object]) -> DotOperand:
    return DotOperand(
        op_idx=int(fields["opIdx"]),  # type: ignore[arg-type]
        parent=_nested(fields["parent"]),
        k_width=int(fields.get("kWidth", 0)),  # type: ignore[arg-type]
    )


def _build_slice(fields: Mapping[str, object]) -> Slice:
    return Slice(dim=int(fields["dim"]), parent=_nested(fields["parent"]))  # type: ignore[arg-type]


def _build_linear(fields: Mapping[str, object]) -> Linear:
    bases = {name: _bases(fields[name]) for name in (K_REGISTER, K_LANE, K_WARP, K_BLOCK)}
    rank = next((len(bs[0]) for bs in bases.values() if bs), 0)
    if rank == 0:
        raise EncodingParseError("a linear layout with no bases has no rank")
    return Linear(LinearLayout(bases, standard_out_dim_names(rank)))


def _build_swizzled_shared(fields: Mapping[str, object]) -> SwizzledShared:
    order = _ints(fields["order"])
    return SwizzledShared(
        vec=int(fields["vec"]),  # type: ignore[arg-type]
        per_phase=int(fields["perPhase"]),  # type: ignore[arg-type]
        max_phase=int(fields["maxPhase"]),  # type: ignore[arg-type]
        order=order,
        cga=_cga_field(fields, len(order)),
    )


def _build_nvmma_shared(fields: Mapping[str, object]) -> NvmmaShared:
    rank = int(fields.get("rank", 2))  # type: ignore[arg-type]
    return NvmmaShared(
        swizzling_byte_width=int(fields["swizzlingByteWidth"]),  # type: ignore[arg-type]
        transposed=bool(fields["transposed"]),
        element_bit_width=int(fields["elementBitWidth"]),  # type: ignore[arg-type]
        fp4_padded=bool(fields.get("fp4Padded", False)),
        cga=_cga_field(fields, rank),
    )


_BUILDERS = {
    "ttg.blocked": _build_blocked,
    "ttg.nvidia_mma": _build_mma,
    "ttg.dot_op": _build_dot_op,
    "ttg.slice": _build_slice,
    "ttg.linear": _build_linear,
    "ttg.swizzled_shared": _build_swizzled_shared,
    "ttg.nvmma_shared": _build_nvmma_shared,
}
