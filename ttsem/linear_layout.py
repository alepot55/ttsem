"""Linear layouts over GF(2), ported from Triton's `lib/Tools/LinearLayout.cpp`.

A linear layout is a function from a hardware location to a logical tensor
index.  Its input dimensions are named (`register`, `lane`, `warp`, `block` for
distributed tensors, `offset` for shared memory) and so are its outputs
(`dim0`, `dim1`, ...).  The function is linear over GF(2): it is fully given by
its value at every power-of-two input, and every other value is the xor of
those bases.

Conventions follow the C++ exactly:

- dimensions are ordered minor-to-major, and the order only matters for reshape;
- the bit order inside a dimension is little-endian, so `bases[d][i]` is the
  value of the layout at `d = 2**i`;
- out-dim sizes are powers of two, and a layout inferring its own sizes must be
  surjective.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

Basis = list[int]
BasesT = dict[str, list[Basis]]
OutDimsT = dict[str, int]


class LayoutError(ValueError):
    """A layout invariant was violated, or an operation is undefined."""


def next_pow2(value: int) -> int:
    """Smallest power of two strictly greater than `value` (LLVM NextPowerOf2)."""
    result = 1
    while result <= value:
        result *= 2
    return result


def log2_exact(value: int) -> int:
    if value <= 0 or value & (value - 1):
        raise LayoutError(f"{value} is not a positive power of 2")
    return value.bit_length() - 1


def rref_gf2(rows: list[int], num_cols: int) -> list[int]:
    """Reduced row echelon form of a GF(2) matrix whose rows are bitmasks.

    Column `c` is bit `c` of a row, so the pivot of a row is its lowest set bit,
    exactly as `f2reduce::inplace_rref_strided` leaves it.
    """
    out = list(rows)
    pivot = 0
    for col in range(num_cols):
        sel = next((r for r in range(pivot, len(out)) if (out[r] >> col) & 1), None)
        if sel is None:
            continue
        out[pivot], out[sel] = out[sel], out[pivot]
        for r in range(len(out)):
            if r != pivot and (out[r] >> col) & 1:
                out[r] ^= out[pivot]
        pivot += 1
    return out


def supremum(x: Sequence[str], y: Sequence[str]) -> list[str]:
    """Least common supersequence of two ordered lists, biased towards `x`.

    Raises when no supersequence respects both orders, e.g. `[a, b]` and
    `[b, a]`.
    """
    result: list[str] = []
    seen: set[str] = set()

    def push(name: str) -> None:
        if name not in seen:
            seen.add(name)
            result.append(name)

    pos_x = {name: i for i, name in enumerate(x)}
    pos_y = {name: i for i, name in enumerate(y)}
    inf = len(x) + len(y) + 1
    i = j = 0
    while i < len(x) or j < len(y):
        while i < len(x) and x[i] in seen:
            i += 1
        while j < len(y) and y[j] in seen:
            j += 1
        if i >= len(x) and j >= len(y):
            break
        if i < len(x) and j < len(y) and x[i] == y[j]:
            if pos_y[x[i]] < j:
                raise LayoutError("supremum does not exist")
            push(x[i])
            i, j = i + 1, j + 1
            continue
        cand_x = pos_y[x[i]] if i < len(x) and pos_y.get(x[i], -1) >= j else inf
        cand_y = pos_x[y[j]] if j < len(y) and pos_x.get(y[j], -1) >= i else inf
        if i < len(x) and cand_x == inf:
            push(x[i])
            i += 1
        elif j < len(y) and cand_y == inf:
            push(y[j])
            j += 1
        elif cand_x <= cand_y:
            push(x[i])
            i += 1
        else:
            push(y[j])
            j += 1
    return result


class LinearLayout:
    """A GF(2)-linear map from named input dims to named output dims."""

    __slots__ = ("bases", "out_dims", "rank")

    bases: BasesT
    out_dims: OutDimsT
    rank: int

    def __init__(
        self,
        bases: Mapping[str, Sequence[Sequence[int]]]
        | Iterable[tuple[str, Sequence[Sequence[int]]]],
        out_dims: Sequence[str] | Sequence[tuple[str, int]],
        require_surjective: bool = True,
    ) -> None:
        self.bases = {name: [list(b) for b in bs] for name, bs in dict(bases).items()}
        out_list = list(out_dims)
        if out_list and isinstance(out_list[0], str):
            self.out_dims = self._infer_out_dims([str(n) for n in out_list])
            require_surjective = True
        else:
            self.out_dims = {str(name): int(size) for name, size in out_list}  # type: ignore[misc]
        self.rank = 0
        self._check_invariants(require_surjective)

    def _infer_out_dims(self, names: Sequence[str]) -> OutDimsT:
        sizes = {name: 1 for name in names}
        for in_bases in self.bases.values():
            for basis in in_bases:
                for i, value in enumerate(basis):
                    sizes[names[i]] = max(sizes[names[i]], next_pow2(value))
        return sizes

    def _check_invariants(self, require_surjective: bool) -> None:
        names = list(self.out_dims)
        for name, size in self.out_dims.items():
            if size <= 0 or size & (size - 1):
                raise LayoutError(f"out-dim {name!r} size {size} is not a power of 2")
        for in_dim, in_bases in self.bases.items():
            for basis in in_bases:
                if any(b < 0 for b in basis):
                    raise LayoutError(f"negative basis for in-dim {in_dim!r}")
                if len(basis) != len(names):
                    raise LayoutError(
                        f"basis for in-dim {in_dim!r} has {len(basis)} entries, "
                        f"expected {len(names)}"
                    )
                for i, value in enumerate(basis):
                    if value >= self.out_dims[names[i]]:
                        raise LayoutError(
                            f"basis {value} for {in_dim!r} -> {names[i]!r} is not "
                            f"smaller than the out-dim size {self.out_dims[names[i]]}"
                        )
        self.rank = self._matrix_rank()
        if require_surjective and not self.is_surjective():
            raise LayoutError(f"layout is not surjective:{self}")

    # -- construction ------------------------------------------------------

    @staticmethod
    def empty() -> LinearLayout:
        return LinearLayout({}, [], require_surjective=True)

    @staticmethod
    def strided1D(size: int, stride: int, in_dim: str, out_dim: str) -> LinearLayout:
        if size == 0:
            return LinearLayout.empty()
        bases = [[i * stride] for i in _powers_below(size)]
        return LinearLayout({in_dim: bases}, [(out_dim, stride * size)], stride == 1)

    @staticmethod
    def identity1D(size: int, in_dim: str, out_dim: str) -> LinearLayout:
        return LinearLayout.strided1D(size, 1, in_dim, out_dim)

    @staticmethod
    def zeros1D(size: int, in_dim: str, out_dim: str, out_dim_size: int = 1) -> LinearLayout:
        if size == 0:
            return LinearLayout.empty()
        bases = [[0] for _ in _powers_below(size)]
        return LinearLayout({in_dim: bases}, [(out_dim, out_dim_size)], out_dim_size == 1)

    # -- queries -----------------------------------------------------------

    def in_dim_names(self) -> list[str]:
        return list(self.bases)

    def out_dim_names(self) -> list[str]:
        return list(self.out_dims)

    def has_in_dim(self, dim: str) -> bool:
        return dim in self.bases

    def has_out_dim(self, dim: str) -> bool:
        return dim in self.out_dims

    def get_out_dim_index(self, dim: str) -> int:
        return list(self.out_dims).index(dim)

    def get_in_dim_size_log2(self, dim: str) -> int:
        return len(self.bases[dim])

    def get_in_dim_size(self, dim: str) -> int:
        return 1 << self.get_in_dim_size_log2(dim)

    def get_out_dim_size(self, dim: str) -> int:
        return self.out_dims[dim]

    def get_out_dim_size_log2(self, dim: str) -> int:
        return log2_exact(self.out_dims[dim])

    def total_in_dim_size_log2(self) -> int:
        return sum(len(b) for b in self.bases.values())

    def total_out_dim_size_log2(self) -> int:
        return sum(log2_exact(s) for s in self.out_dims.values())

    def total_in_dim_size(self) -> int:
        return 1 << self.total_in_dim_size_log2()

    def total_out_dim_size(self) -> int:
        return 1 << self.total_out_dim_size_log2()

    def get_basis(self, in_dim: str, pos: int, out_dim: str) -> int:
        """`L(0, ..., in_dim = 2**pos, ..., 0)` projected on `out_dim`."""
        return self.bases[in_dim][pos][self.get_out_dim_index(out_dim)]

    def is_surjective(self) -> bool:
        return self.rank == self.total_out_dim_size_log2()

    def is_invertible(self) -> bool:
        return self.is_surjective() and self.total_in_dim_size() == self.total_out_dim_size()

    # -- matrix form -------------------------------------------------------

    def matrix(self) -> list[int]:
        """Rows are out-bits, columns (bit positions) are in-bits."""
        num_rows = self.total_out_dim_size_log2()
        rows = [0] * num_rows
        r = 0
        for out_dim in self.out_dims:
            idx = self.get_out_dim_index(out_dim)
            width = self.get_out_dim_size_log2(out_dim)
            c = 0
            for in_bases in self.bases.values():
                for basis in in_bases:
                    value = basis[idx]
                    for j in range(width):
                        rows[r + j] |= ((value >> j) & 1) << c
                    c += 1
            r += width
        return rows

    def _matrix_rank(self) -> int:
        num_cols = self.total_in_dim_size_log2()
        reduced = rref_gf2(self.matrix(), num_cols)
        return sum(1 for row in reduced if row != 0)

    # -- evaluation --------------------------------------------------------

    def apply(self, ins: Mapping[str, int]) -> dict[str, int]:
        if set(ins) != set(self.bases):
            raise LayoutError(f"apply expects in-dims {self.in_dim_names()}, got {list(ins)}")
        out: dict[str, int] = {}
        for out_dim in self.out_dims:
            idx = self.get_out_dim_index(out_dim)
            acc = 0
            for in_dim, value in ins.items():
                for i, basis in enumerate(self.bases[in_dim]):
                    if value & (1 << i):
                        acc ^= basis[idx]
            out[out_dim] = acc
        return out

    # -- structural transforms --------------------------------------------

    def transpose_ins(self, new_in_dims: Sequence[str]) -> LinearLayout:
        if set(new_in_dims) != set(self.bases):
            raise LayoutError("transpose_ins must be a permutation of the in-dims")
        bases = {dim: self.bases[dim] for dim in new_in_dims}
        return LinearLayout(bases, list(self.out_dims.items()), self.is_surjective())

    def transpose_outs(self, new_out_dims: Sequence[str]) -> LinearLayout:
        if set(new_out_dims) != set(self.out_dims):
            raise LayoutError("transpose_outs must be a permutation of the out-dims")
        perm = [self.get_out_dim_index(dim) for dim in new_out_dims]
        bases = {
            in_dim: [[basis[i] for i in perm] for basis in in_bases]
            for in_dim, in_bases in self.bases.items()
        }
        sizes = [(dim, self.out_dims[dim]) for dim in new_out_dims]
        return LinearLayout(bases, sizes, self.is_surjective())

    def reshape_outs(self, new_out_dims: Sequence[tuple[str, int]]) -> LinearLayout:
        shifts = []
        acc = 0
        for out_dim in self.out_dims:
            shifts.append(acc)
            acc += self.get_out_dim_size_log2(out_dim)
        bases: BasesT = {}
        for in_dim, in_bases in self.bases.items():
            dim_bases = []
            for basis in in_bases:
                flat = sum(value << shifts[i] for i, value in enumerate(basis))
                multi = []
                for _, size in new_out_dims:
                    multi.append(flat % size)
                    flat //= size
                dim_bases.append(multi)
            bases[in_dim] = dim_bases
        return LinearLayout(bases, list(new_out_dims), self.is_surjective())

    def resize_out_dim(self, out_dim: str, new_size: int) -> LinearLayout:
        """Shrink an out-dim, zeroing every basis that no longer fits."""
        if new_size > self.get_out_dim_size(out_dim):
            raise LayoutError("resize_out_dim may only shrink an out-dim")
        idx = self.get_out_dim_index(out_dim)
        bases = {
            in_dim: [
                [0 if i == idx and v >= new_size else v for i, v in enumerate(basis)]
                for basis in in_bases
            ]
            for in_dim, in_bases in self.bases.items()
        }
        sizes = [(d, new_size if d == out_dim else s) for d, s in self.out_dims.items()]
        return LinearLayout(bases, sizes, False)

    def rename_outs(self, renames: Mapping[str, str]) -> LinearLayout:
        sizes = [(renames.get(dim, dim), size) for dim, size in self.out_dims.items()]
        return LinearLayout(self.bases, sizes, self.is_surjective())

    def sublayout(self, in_dims: Sequence[str], out_dims: Sequence[str]) -> LinearLayout:
        in_set, out_set = set(in_dims), set(out_dims)
        if not in_set <= set(self.bases) or not out_set <= set(self.out_dims):
            raise LayoutError("sublayout dims must be a subset of the layout's dims")
        keep = [i for i, dim in enumerate(self.out_dims) if dim in out_set]
        bases = {
            in_dim: [[basis[i] for i in keep] for basis in in_bases]
            for in_dim, in_bases in self.bases.items()
            if in_dim in in_set
        }
        sizes = [(dim, size) for dim, size in self.out_dims.items() if dim in out_set]
        return LinearLayout(bases, sizes, False)

    def sublayout_is_zero(self, in_dims: Sequence[str], out_dims: Sequence[str]) -> bool:
        return all(output_basis_mask(self, in_dims, dim) == 0 for dim in out_dims)

    def remove_zero_bases_along_dim(self, strip_dim: str) -> LinearLayout:
        nonzero = input_basis_mask(self, strip_dim, self.out_dim_names())
        bases: BasesT = {}
        for in_dim, in_bases in self.bases.items():
            if in_dim != strip_dim:
                bases[in_dim] = [list(b) for b in in_bases]
                continue
            bases[in_dim] = [list(b) for i, b in enumerate(in_bases) if nonzero & (1 << i)]
        return LinearLayout(bases, list(self.out_dims.items()), self.is_surjective())

    # -- algebra -----------------------------------------------------------

    def __mul__(self, outer: LinearLayout) -> LinearLayout:
        inner = self
        in_dims = supremum(inner.in_dim_names(), outer.in_dim_names())
        out_dims = supremum(inner.out_dim_names(), outer.out_dim_names())

        in_log2 = {dim: 0 for dim in in_dims}
        out_log2 = {dim: 0 for dim in out_dims}
        for layout in (inner, outer):
            for dim in layout.in_dim_names():
                in_log2[dim] += layout.get_in_dim_size_log2(dim)
            for dim in layout.out_dim_names():
                out_log2[dim] += layout.get_out_dim_size_log2(dim)

        bases: BasesT = {}
        for in_dim, width in in_log2.items():
            dim_bases = [[0] * len(out_log2) for _ in range(width)]
            for out_idx, out_dim in enumerate(out_log2):
                if inner.has_in_dim(in_dim) and inner.has_out_dim(out_dim):
                    for i in range(inner.get_in_dim_size_log2(in_dim)):
                        dim_bases[i][out_idx] = inner.get_basis(in_dim, i, out_dim)
                if outer.has_in_dim(in_dim) and outer.has_out_dim(out_dim):
                    offset = inner.get_in_dim_size_log2(in_dim) if inner.has_in_dim(in_dim) else 0
                    shift = (
                        inner.get_out_dim_size_log2(out_dim) if inner.has_out_dim(out_dim) else 0
                    )
                    for i in range(outer.get_in_dim_size_log2(in_dim)):
                        dim_bases[offset + i][out_idx] = (
                            outer.get_basis(in_dim, i, out_dim) << shift
                        )
            bases[in_dim] = dim_bases

        sizes = [(dim, 1 << width) for dim, width in out_log2.items()]
        return LinearLayout(bases, sizes, inner.is_surjective() and outer.is_surjective())

    def compose(self, outer: LinearLayout) -> LinearLayout:
        if set(self.out_dims) != set(outer.bases):
            raise LayoutError("compose requires this layout's out-dims to be outer's in-dims")
        for dim in self.out_dims:
            if self.get_out_dim_size(dim) > outer.get_in_dim_size(dim):
                raise LayoutError(f"compose: out-dim {dim!r} is larger than outer's in-dim")
        bases: BasesT = {}
        for in_dim, in_bases in self.bases.items():
            dim_bases = []
            for basis in in_bases:
                point = dict(zip(self.out_dims, basis, strict=True))
                dim_bases.append(list(outer.apply(point).values()))
            bases[in_dim] = dim_bases
        surjective = (
            self.is_surjective()
            and outer.is_surjective()
            and all(self.get_out_dim_size(d) == outer.get_in_dim_size(d) for d in self.out_dims)
        )
        return LinearLayout(bases, list(outer.out_dims.items()), surjective)

    def invert_and_compose(self, outer: LinearLayout) -> LinearLayout:
        """`C = B.invert_and_compose(A)` satisfies `A(C(x)) == B(x)`."""
        out_dims = self.out_dim_names()
        if set(out_dims) != set(outer.out_dims):
            raise LayoutError("invert_and_compose requires matching out-dims")
        b = self
        a = outer.transpose_outs(out_dims)
        for dim in out_dims:
            if a.get_out_dim_size(dim) < b.get_out_dim_size(dim):
                raise LayoutError(f"invert_and_compose: outer is too small in {dim!r}")

        identity_dims = [
            dim
            for dim in a.in_dim_names()
            if b.has_in_dim(dim)
            and a.sublayout([dim], out_dims).equal_ignoring_out_dim_sizes(
                b.sublayout([dim], out_dims)
            )
        ]
        a_rest = [d for d in a.in_dim_names() if d not in identity_dims]
        b_rest = [d for d in b.in_dim_names() if d not in identity_dims]
        solution = lstsq(a.sublayout(a_rest, out_dims), b.sublayout(b_rest, out_dims))
        if solution is None:
            raise LayoutError("outer layout does not cover this layout's image")
        for dim in identity_dims:
            solution = solution * LinearLayout.identity1D(a.get_in_dim_size(dim), dim, dim)
        return solution.transpose_ins(b.in_dim_names()).transpose_outs(a.in_dim_names())

    def pseudoinvert(self) -> LinearLayout:
        identity = LinearLayout.empty()
        for out_dim in self.out_dims:
            identity = identity * LinearLayout.identity1D(
                self.get_out_dim_size(out_dim), out_dim, out_dim
            )
        return identity.invert_and_compose(self)

    def invert(self) -> LinearLayout:
        if not self.is_invertible():
            raise LayoutError("layout must be surjective and square to be invertible")
        return self.pseudoinvert()

    def free_variable_masks(self) -> dict[str, int]:
        reduced = rref_gf2(self.matrix(), self.total_in_dim_size_log2())
        basic = {(row & -row).bit_length() - 1 for row in reduced if row != 0}
        masks: dict[str, int] = {}
        c = 0
        for dim, in_bases in self.bases.items():
            mask = 0
            for i in range(len(in_bases)):
                if c not in basic:
                    mask |= 1 << i
                c += 1
            masks[dim] = mask
        return masks

    def num_consecutive_in_out(self) -> int:
        if not self.bases or not self.out_dims:
            return 1
        in_dim = self.in_dim_names()[0]
        out_dim = self.out_dim_names()[0]
        limit = min(self.get_in_dim_size(in_dim), self.get_out_dim_size(out_dim))
        return maximal_identity_prefix(self, in_dim, out_dim, limit)

    # -- comparison and printing -------------------------------------------

    def equal_ignoring_out_dim_sizes(self, other: LinearLayout) -> bool:
        return self.out_dim_names() == other.out_dim_names() and self.bases == other.bases

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LinearLayout):
            return NotImplemented
        return self.equal_ignoring_out_dim_sizes(other) and list(self.out_dims.values()) == list(
            other.out_dims.values()
        )

    def __hash__(self) -> int:
        flat = tuple((d, tuple(tuple(b) for b in bs)) for d, bs in self.bases.items())
        return hash((flat, tuple(self.out_dims.items())))

    def __str__(self) -> str:
        out = "[" + ", ".join(f"{d} (size {s})" for d, s in self.out_dims.items()) + "]"
        if not self.bases:
            return (
                "\n(empty layout)" if not self.out_dims else f"\n(empty layout with out-dims {out})"
            )
        lines = ["\n"]
        for in_dim, in_bases in self.bases.items():
            if not in_bases:
                lines.append(f" - {in_dim} is a size 1 dimension\n")
                continue
            body = "\n   ".join(
                f"{in_dim}={1 << i} -> ({', '.join(str(v) for v in basis)})"
                for i, basis in enumerate(in_bases)
            )
            lines.append(f" - {body}\n")
        lines.append(f"where out dims are: {out}")
        return "".join(lines)

    __repr__ = __str__


def _powers_below(size: int) -> list[int]:
    powers = []
    i = 1
    while i < size:
        powers.append(i)
        i *= 2
    return powers


def output_basis_mask(layout: LinearLayout, in_dims: Sequence[str], out_dim: str) -> int:
    """Or of every basis value that the given in-dims write into `out_dim`."""
    idx = layout.get_out_dim_index(out_dim)
    mask = 0
    for in_dim in in_dims:
        for basis in layout.bases[in_dim]:
            mask |= basis[idx]
    return mask


def input_basis_mask(layout: LinearLayout, in_dim: str, out_dims: Sequence[str]) -> int:
    """Bitmask of the bases of `in_dim` that are non-zero on any of `out_dims`."""
    indices = [layout.get_out_dim_index(dim) for dim in out_dims]
    mask = 0
    for i, basis in enumerate(layout.bases[in_dim]):
        if any(basis[idx] != 0 for idx in indices):
            mask |= 1 << i
    return mask


def _concat_matrices(a: LinearLayout, b: LinearLayout) -> list[int]:
    num_cols_a = a.total_in_dim_size_log2()
    num_cols_b = b.total_in_dim_size_log2()
    concat = a.matrix()
    b_mat = b.matrix()
    row_a = row_b = 0
    for out_dim, size in a.out_dims.items():
        for r in range(log2_exact(size)):
            if r < b.get_out_dim_size_log2(out_dim):
                if num_cols_b:
                    concat[row_a] |= b_mat[row_b] << num_cols_a
                row_b += 1
            row_a += 1
    return concat


def lstsq(a: LinearLayout, b: LinearLayout) -> LinearLayout | None:
    """Solve `A X = B` over GF(2), free variables set to zero.

    Returns None when the image of B is not contained in the image of A, or the
    out-dims are incompatible.
    """
    if set(a.out_dims) != set(b.out_dims):
        return None
    if any(b.get_out_dim_size(d) > a.get_out_dim_size(d) for d in a.out_dims):
        return None

    ordered_b = b.transpose_outs(a.out_dim_names())
    num_rows = a.total_out_dim_size_log2()
    num_cols_a = a.total_in_dim_size_log2()
    num_cols_b = ordered_b.total_in_dim_size_log2()
    combined = rref_gf2(_concat_matrices(a, ordered_b), num_cols_a + num_cols_b)

    pivot_of_col = [-1] * num_cols_a
    for r in range(num_rows):
        row = combined[r]
        if row == 0:
            continue
        col = (row & -row).bit_length() - 1
        if col >= num_cols_a:
            return None
        pivot_of_col[col] = r

    solution = [0] * num_cols_a
    for col in range(num_cols_a):
        row = pivot_of_col[col]
        if row != -1 and num_cols_b:
            solution[col] = combined[row] >> num_cols_a

    bases: BasesT = {}
    b_column = 0
    for b_in_dim in ordered_b.in_dim_names():
        dim_bases = []
        for _ in range(ordered_b.get_in_dim_size_log2(b_in_dim)):
            basis = []
            a_column = 0
            for a_in_dim in a.in_dim_names():
                value = 0
                for a_bit in range(a.get_in_dim_size_log2(a_in_dim)):
                    if solution[a_column] & (1 << b_column):
                        value |= 1 << a_bit
                    a_column += 1
                basis.append(value)
            dim_bases.append(basis)
            b_column += 1
        bases[b_in_dim] = dim_bases

    sizes = [(dim, a.get_in_dim_size(dim)) for dim in a.in_dim_names()]
    return LinearLayout(bases, sizes, False)


def divide_left(a: LinearLayout, b: LinearLayout) -> LinearLayout | None:
    """Compute `C` with `A == B * C`, or None when no such `C` exists."""
    if not set(b.bases) <= set(a.bases) or not set(b.out_dims) <= set(a.out_dims):
        return None
    c_out_sizes: dict[str, int] = {}
    for out_dim in a.out_dims:
        width_b = b.get_out_dim_size_log2(out_dim) if b.has_out_dim(out_dim) else 0
        width_c = a.get_out_dim_size_log2(out_dim) - width_b
        if width_c < 0:
            return None
        c_out_sizes[out_dim] = 1 << width_c

    bases: BasesT = {}
    for in_dim in a.in_dim_names():
        width_b = b.get_in_dim_size_log2(in_dim) if b.has_in_dim(in_dim) else 0
        width_a = a.get_in_dim_size_log2(in_dim)
        if width_a < width_b:
            return None
        for i in range(width_b):
            for out_dim in a.out_dims:
                expected = b.get_basis(in_dim, i, out_dim) if b.has_out_dim(out_dim) else 0
                if a.get_basis(in_dim, i, out_dim) != expected:
                    return None
        dim_bases = []
        for i in range(width_b, width_a):
            basis = []
            for out_dim in c_out_sizes:
                shift = b.get_out_dim_size_log2(out_dim) if b.has_out_dim(out_dim) else 0
                value = a.get_basis(in_dim, i, out_dim)
                if value & ((1 << shift) - 1):
                    return None
                basis.append(value >> shift)
            dim_bases.append(basis)
        bases[in_dim] = dim_bases

    surjective = a.is_surjective() and b.is_surjective()
    return LinearLayout(bases, list(c_out_sizes.items()), surjective)


def maximal_identity_prefix(layout: LinearLayout, in_dim: str, out_dim: str, max_size: int) -> int:
    """Largest `V <= max_size` such that the layout starts with an identity of size V."""
    result = 1
    size = 2
    while size <= max_size:
        if divide_left(layout, LinearLayout.identity1D(size, in_dim, out_dim)) is None:
            break
        result = size
        size *= 2
    return result
