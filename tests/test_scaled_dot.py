"""`tt.dot_scaled` and `ttg.fp4_to_fp`, against a reference written from the formats.

`reference_scaled_dot` below decodes every element with scalar Python arithmetic straight from
the definitions of `e2m1`, `e8m0` and the fp8 kinds, and multiplies with three nested loops. It
shares no code with `ops.py`: the point is that two independent readings of the same
specification agree, not that the implementation reproduces itself.

The values are chosen so that the compute type (`bf16`, or `f16` when a format is `fp16`)
holds every product exactly, which every microscaling format satisfies by construction: a
scale is a power of two and `e2m1`, `e4m3` and `e5m2` have at most four significand bits
against bf16's eight. `test_compute_type_rounds_the_operands` is the one case that leaves that
regime on purpose.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import one, op, tensor, ty

from ttsem import mlir
from ttsem.values import Unsupported, from_float

E4M3 = "f8E4M3FN"
E5M2 = "f8E5M2"
E2M1, E4M3_FMT, E5M2_FMT, BF16_FMT, FP16_FMT = 4, 0, 1, 5, 6
FORMAT_NAMES = {E4M3_FMT: E4M3, E5M2_FMT: E5M2, BF16_FMT: "bf16", FP16_FMT: "f16"}
# what the operand's own MLIR type is: packed `e2m1` has none of its own and travels as bytes
STORAGE_NAMES = {E2M1: "i8", **FORMAT_NAMES}


# --------------------------------------------------------------------------- the reference


def ref_e2m1(nibble: int) -> float:
    """One `e2m1` nibble: a sign bit, two exponent bits with bias 1, one mantissa bit."""
    sign = -1.0 if nibble & 0x8 else 1.0
    exponent, mantissa = (nibble >> 1) & 0x3, nibble & 0x1
    if exponent == 0:
        return sign * 0.5 * mantissa
    return sign * (2.0 ** (exponent - 1)) * (1.0 + 0.5 * mantissa)


def ref_fp8(byte: int, name: str) -> float:
    """One `e4m3` (no infinity, 0x7F is NaN) or `e5m2` (IEEE-shaped) byte."""
    ebits, mbits, bias = (4, 3, 7) if name == E4M3 else (5, 2, 15)
    top = (1 << ebits) - 1
    sign = -1.0 if byte >> 7 else 1.0
    exponent, mantissa = (byte >> mbits) & top, byte & ((1 << mbits) - 1)
    if exponent == top and name == E5M2:
        return sign * float("inf") if mantissa == 0 else float("nan")
    if exponent == top and mantissa == (1 << mbits) - 1:
        return float("nan")
    if exponent == 0:
        return sign * (mantissa / (1 << mbits)) * 2.0 ** (1 - bias)
    return sign * (1.0 + mantissa / (1 << mbits)) * 2.0 ** (exponent - bias)


def ref_e8m0(byte: int, compute: str = "bf16") -> float:
    """One `e8m0` scale byte: the exponent field of an f32, so 127 is 1.0 and 255 is NaN.

    Byte 0 follows `DecomposeScaledBlocked::scaleTo16` since triton#11624: `2**-127` (a bf16
    subnormal) when the dot computes in bf16, the zero that the shift produces when it
    computes in f16.
    """
    if byte == 0xFF:
        return float("nan")
    if byte == 0:
        return 2.0**-127 if compute == "bf16" else 0.0
    return 2.0 ** (byte - 127)


def ref_decode(raw: np.ndarray, fmt: int) -> np.ndarray:
    """A stored operand as a float64 matrix of the same shape, without unpacking."""
    flat = np.asarray(raw).reshape(-1)
    if fmt in (E4M3_FMT, E5M2_FMT):
        out = [ref_fp8(int(np.uint8(v)), FORMAT_NAMES[fmt]) for v in flat]
    elif fmt == BF16_FMT:
        out = [float(np.frombuffer(np.uint32(int(v) << 16).tobytes(), np.float32)[0]) for v in flat]
    else:
        out = [float(np.float16(v)) for v in flat]
    return np.array(out, dtype=np.float64).reshape(np.shape(raw))


def ref_unpack(raw: np.ndarray, axis: int) -> np.ndarray:
    """Two nibbles per byte along `axis`, low nibble first, one element at a time."""
    raw = np.asarray(raw)
    shape = list(raw.shape)
    shape[axis] *= 2
    out = np.zeros(shape, dtype=np.float64)
    for index in np.ndindex(*shape):
        source = list(index)
        source[axis] = index[axis] // 2
        byte = int(np.uint8(raw[tuple(source)]))
        out[index] = ref_e2m1((byte >> 4) if index[axis] % 2 else (byte & 0xF))
    return out


def ref_operand(
    raw: np.ndarray,
    fmt: int,
    scale: np.ndarray | None,
    op_idx: int,
    k_pack: bool,
    compute: str = "bf16",
) -> np.ndarray:
    """One decoded and scaled operand, `[M, K]` for the lhs and `[K, N]` for the rhs."""
    k_dim = 1 if op_idx == 0 else 0
    if fmt == E2M1:
        values = ref_unpack(raw, k_dim if k_pack else 1 - k_dim)
    else:
        values = ref_decode(raw, fmt)
    if scale is None:
        return values
    scales = np.asarray(scale)
    out = np.zeros_like(values)
    rows, cols = values.shape
    for i in range(rows):
        for j in range(cols):
            # the scale is indexed [M, K / group] for the lhs and [N, K / group] for the rhs
            along_k, other = (j, i) if op_idx == 0 else (i, j)
            group = values.shape[k_dim] // scales.shape[1]
            byte = int(np.uint8(scales[other, along_k // group]))
            out[i, j] = values[i, j] * ref_e8m0(byte, compute)
    return out


def reference_scaled_dot(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    fmt_a: int,
    fmt_b: int,
    a_scale: np.ndarray | None = None,
    b_scale: np.ndarray | None = None,
    lhs_k_pack: bool = True,
    rhs_k_pack: bool = True,
) -> np.ndarray:
    compute = "f16" if FP16_FMT in (fmt_a, fmt_b) else "bf16"
    lhs = ref_operand(a, fmt_a, a_scale, 0, lhs_k_pack, compute)
    rhs = ref_operand(b, fmt_b, b_scale, 1, rhs_k_pack, compute)
    out = np.array(c, dtype=np.float64)
    for i in range(lhs.shape[0]):
        for j in range(rhs.shape[1]):
            for k in range(lhs.shape[1]):
                out[i, j] += lhs[i, k] * rhs[k, j]
    return out


# --------------------------------------------------------------------------- the op builders


def scaled_dot_op(
    a_shape: tuple[int, int],
    a_elem: str,
    b_shape: tuple[int, int],
    b_elem: str,
    out_shape: tuple[int, int],
    *,
    a_scale: tuple[int, int] | None = None,
    b_scale: tuple[int, int] | None = None,
    scale_elem: str = "i8",
    **attrs: object,
):
    """A `tt.dot_scaled` whose operand list holds only the scales that are present."""
    types = [tensor(a_shape, a_elem), tensor(b_shape, b_elem), tensor(out_shape, "f32")]
    sizes = [1, 1, 1, int(a_scale is not None), int(b_scale is not None)]
    for shape in (a_scale, b_scale):
        if shape is not None:
            types.append(tensor(shape, scale_elem))
    full = {"operandSegmentSizes": sizes, "fastMath": False, **attrs}
    return op("tt.dot_scaled", types, tensor(out_shape, "f32"), attrs=full)


def run_scaled_dot(a, b, c, fmt_a, fmt_b, a_scale=None, b_scale=None, **attrs) -> np.ndarray:
    """Evaluate the op on the same arguments the reference takes."""
    target = scaled_dot_op(
        a.shape,
        STORAGE_NAMES[fmt_a],
        b.shape,
        STORAGE_NAMES[fmt_b],
        c.shape,
        a_scale=None if a_scale is None else a_scale.shape,
        b_scale=None if b_scale is None else b_scale.shape,
        a_elem_type=fmt_a,
        b_elem_type=fmt_b,
        **attrs,
    )
    args = [a, b, c] + [s for s in (a_scale, b_scale) if s is not None]
    return one(target, args)


def bytes_of(*values: int, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """A stored `i8` tensor from raw byte values (the storage of an `i8` type is signed)."""
    out = np.array(values, dtype=np.uint8).astype(np.int8)
    return out if shape is None else out.reshape(shape)


# --------------------------------------------------------------------------- ttg.fp4_to_fp


@pytest.mark.parametrize("elem", ["bf16", "f16"])
def test_fp4_to_fp_unpacks_the_low_nibble_first(elem: str) -> None:
    src = bytes_of(0x21, 0x76, shape=(1, 2))
    got = one(
        op("ttg.fp4_to_fp", [tensor((1, 2), "i8")], tensor((1, 4), elem), attrs={"axis": 1}),
        [src],
    )
    want = ref_unpack(src, 1)
    assert np.array_equal(got, from_float(want.astype(np.float32), ty(elem)))
    # spelled out: 0x21 is the pair (1, 2) and 0x76 the pair (6, 7)
    assert np.array_equal(want, np.array([[0.5, 1.0, 4.0, 6.0]]))


def test_fp4_to_fp_along_axis_0() -> None:
    src = bytes_of(0x21, 0x76, 0x8F, 0x00, shape=(2, 2))
    got = one(
        op("ttg.fp4_to_fp", [tensor((2, 2), "i8")], tensor((4, 2), "bf16"), attrs={"axis": 0}),
        [src],
    )
    want = ref_unpack(src, 0)
    assert np.array_equal(got, from_float(want.astype(np.float32), ty("bf16")))
    assert np.array_equal(want[:, 0], np.array([0.5, 1.0, -6.0, -0.0]))


def test_fp4_to_fp_keeps_the_sign_of_a_negative_zero() -> None:
    src = bytes_of(0x08, shape=(1, 1))
    got = one(
        op("ttg.fp4_to_fp", [tensor((1, 1), "i8")], tensor((1, 2), "bf16"), attrs={"axis": 1}),
        [src],
    )
    assert np.array_equal(got, np.array([[0x8000, 0x0000]], dtype=np.uint16))


def test_fp4_to_fp_rejects_an_axis_out_of_range() -> None:
    target = op("ttg.fp4_to_fp", [tensor((1, 2), "i8")], tensor((1, 4), "bf16"), attrs={"axis": 3})
    with pytest.raises(Unsupported):
        one(target, [bytes_of(0, 0, shape=(1, 2))])


# --------------------------------------------------------------------------- tt.dot_scaled


def test_scaled_dot_fp4_times_fp8_with_one_scale_group() -> None:
    """K = 8, so one `e8m0` scale covers the whole reduction of each row."""
    a = bytes_of(0x21, 0x43, 0x65, 0x07, 0x12, 0x34, 0x56, 0x70, shape=(2, 4))
    b = np.arange(1, 17, dtype=np.uint8).reshape(8, 2)
    scale = bytes_of(128, 126, shape=(2, 1))  # 2.0 and 0.5
    c = np.zeros((2, 2), np.float32)
    got = run_scaled_dot(a, b, c, E2M1, E4M3_FMT, a_scale=scale)
    want = reference_scaled_dot(a, b, c, E2M1, E4M3_FMT, a_scale=scale)
    assert np.array_equal(got, want.astype(np.float32))


def test_scaled_dot_applies_one_scale_per_group_of_32() -> None:
    """Two groups along K, so the second 32 elements of the row take the second scale."""
    rng = np.random.default_rng(7)
    a = rng.integers(0, 256, size=(2, 32), dtype=np.uint8).astype(np.int8)  # 64 nibbles
    b = np.tile(bytes_of(56), (64, 3))  # e4m3 56 is 1.0
    scale = bytes_of(127, 129, 126, 127, shape=(2, 2))  # 1, 4, 0.5, 1
    c = np.zeros((2, 3), np.float32)
    got = run_scaled_dot(a, b, c, E2M1, E4M3_FMT, a_scale=scale)
    want = reference_scaled_dot(a, b, c, E2M1, E4M3_FMT, a_scale=scale)
    assert np.array_equal(got, want.astype(np.float32))
    # the row is the sum of the first group plus four times the second, times nothing else
    values = ref_unpack(a, 1)
    assert got[0, 0] == np.float32(values[0, :32].sum() + 4.0 * values[0, 32:].sum())


def test_scaled_dot_scale_factor_of_16() -> None:
    """The group is the ratio of the shapes, so a scale of `K / 16` entries also works."""
    a = np.tile(bytes_of(56), (1, 32))  # 32 e4m3 ones
    b = np.tile(bytes_of(56), (32, 1))
    scale = bytes_of(127, 128, shape=(1, 2))  # 1.0 over the first 16, 2.0 over the second
    c = np.zeros((1, 1), np.float32)
    got = run_scaled_dot(a, b, c, E4M3_FMT, E4M3_FMT, a_scale=scale)
    assert got[0, 0] == np.float32(16.0 + 32.0)
    assert got[0, 0] == reference_scaled_dot(a, b, c, E4M3_FMT, E4M3_FMT, a_scale=scale)[0, 0]


def test_scaled_dot_nan_scale_poisons_its_whole_group() -> None:
    a = np.tile(bytes_of(56), (2, 32))
    b = np.tile(bytes_of(56), (32, 2))
    scale = bytes_of(0xFF, 127, shape=(2, 1))
    c = np.zeros((2, 2), np.float32)
    got = run_scaled_dot(a, b, c, E4M3_FMT, E4M3_FMT, a_scale=scale)
    assert np.all(np.isnan(got[0])) and np.array_equal(got[1], np.full(2, 32.0, np.float32))
    want = reference_scaled_dot(a, b, c, E4M3_FMT, E4M3_FMT, a_scale=scale)
    assert np.array_equal(np.isnan(got), np.isnan(want))


def test_scaled_dot_fast_math_leaves_the_nan_scale_as_an_infinity() -> None:
    """`fastMath` skips `maskNan`, so 0xFF stays the infinity the shifted byte encodes."""
    a = np.tile(bytes_of(56), (1, 4))
    b = np.tile(bytes_of(56), (4, 1))
    scale = bytes_of(0xFF, shape=(1, 1))
    c = np.zeros((1, 1), np.float32)
    got = run_scaled_dot(a, b, c, E4M3_FMT, E4M3_FMT, a_scale=scale, fastMath=True)
    assert np.isposinf(got[0, 0])


def test_scaled_dot_a_zero_scale_byte_is_two_to_the_minus_127_in_the_bf16_path() -> None:
    """Two e4m3 operands compute in bf16, where byte 0 is the minimum scale since triton#11624
    (3.8.0 read it as zero: that difference is the E8M0 item of triton#11735)."""
    a = np.tile(bytes_of(56), (1, 4))
    b = np.tile(bytes_of(56), (4, 1))
    got = run_scaled_dot(
        a, b, np.zeros((1, 1), np.float32), E4M3_FMT, E4M3_FMT, a_scale=bytes_of(0, shape=(1, 1))
    )
    assert got[0, 0] == np.float32(4 * 2.0**-127)


def test_scaled_dot_bf16_times_fp4_without_c() -> None:
    values = np.array([[1.0, 2.0, -3.0, 0.5, 8.0, 1.5, -1.0, 4.0]], np.float32)
    a = from_float(values, ty("bf16"))
    b = bytes_of(0x21, 0x43, 0x65, 0x07, shape=(4, 1))
    scale = bytes_of(129, shape=(1, 1))  # 4.0 over the whole K = 8 of the rhs
    c = np.zeros((1, 1), np.float32)
    got = run_scaled_dot(a, b, c, BF16_FMT, E2M1, b_scale=scale)
    raw_a = np.asarray(a, dtype=np.uint16).astype(np.int64)
    want = reference_scaled_dot(raw_a, b, c, BF16_FMT, E2M1, b_scale=scale)
    assert got[0, 0] == np.float32(want[0, 0])


def test_scaled_dot_bf16_times_fp4_with_c() -> None:
    values = np.array([[1.0, 2.0, -3.0, 0.5, 8.0, 1.5, -1.0, 4.0]], np.float32)
    a = from_float(values, ty("bf16"))
    b = bytes_of(0x21, 0x43, 0x65, 0x07, shape=(4, 1))
    scale = bytes_of(129, shape=(1, 1))
    zero, c = np.zeros((1, 1), np.float32), np.array([[2.5]], np.float32)
    without = run_scaled_dot(a, b, zero, BF16_FMT, E2M1, b_scale=scale)
    with_c = run_scaled_dot(a, b, c, BF16_FMT, E2M1, b_scale=scale)
    assert with_c[0, 0] == without[0, 0] + np.float32(2.5)
    raw_a = np.asarray(a, dtype=np.uint16).astype(np.int64)
    want = reference_scaled_dot(raw_a, b, c, BF16_FMT, E2M1, b_scale=scale)
    assert with_c[0, 0] == np.float32(want[0, 0])


def test_scaled_dot_without_any_scale_is_a_plain_upcast_dot() -> None:
    a = bytes_of(0x21, 0x43, 0x65, 0x07, shape=(1, 4))
    b = np.tile(bytes_of(56), (8, 1))
    c = np.zeros((1, 1), np.float32)
    got = run_scaled_dot(a, b, c, E2M1, E4M3_FMT)
    want = reference_scaled_dot(a, b, c, E2M1, E4M3_FMT)
    assert got[0, 0] == np.float32(want[0, 0]) == np.float32(ref_unpack(a, 1).sum())


def test_scaled_dot_lhs_packed_along_m() -> None:
    """`lhs_k_pack = false`: the nibbles of the lhs run along M, so M doubles and K does not."""
    a = bytes_of(0x21, 0x43, shape=(1, 2))  # unpacks to 2 rows of 2 along M
    b = np.tile(bytes_of(56), (2, 3))
    c = np.zeros((2, 3), np.float32)
    got = run_scaled_dot(a, b, c, E2M1, E4M3_FMT, lhs_k_pack=False)
    want = reference_scaled_dot(a, b, c, E2M1, E4M3_FMT, lhs_k_pack=False)
    assert np.array_equal(got, want.astype(np.float32))
    # the low nibbles are the first row along M and the high ones the second
    assert np.array_equal(got[:, 0], np.array([0.5 + 1.5, 1.0 + 2.0], np.float32))


def test_scaled_dot_rhs_packed_along_n() -> None:
    """`rhs_k_pack = false`: the nibbles of the rhs run along N, so N doubles."""
    a = np.tile(bytes_of(56), (1, 2))
    b = bytes_of(0x21, 0x43, shape=(2, 1))
    c = np.zeros((1, 2), np.float32)
    got = run_scaled_dot(a, b, c, E4M3_FMT, E2M1, rhs_k_pack=False)
    want = reference_scaled_dot(a, b, c, E4M3_FMT, E2M1, rhs_k_pack=False)
    assert np.array_equal(got, want.astype(np.float32))
    assert np.array_equal(got[0], np.array([0.5 + 1.5, 1.0 + 2.0], np.float32))


def test_scaled_dot_rhs_scale_is_given_transposed() -> None:
    """The rhs scale is `[N, K / group]`, not `[K / group, N]`: the columns scale apart."""
    a = np.tile(bytes_of(56), (1, 8))
    b = np.tile(bytes_of(56), (8, 2))
    scale = bytes_of(128, 126, shape=(2, 1))  # column 0 by 2.0, column 1 by 0.5
    c = np.zeros((1, 2), np.float32)
    got = run_scaled_dot(a, b, c, E4M3_FMT, E4M3_FMT, b_scale=scale)
    assert np.array_equal(got, np.array([[16.0, 4.0]], np.float32))
    want = reference_scaled_dot(a, b, c, E4M3_FMT, E4M3_FMT, b_scale=scale)
    assert np.array_equal(got, want.astype(np.float32))


def test_scaled_dot_reads_an_fp8_operand_stored_as_i8() -> None:
    """`a_elem_type` is the authority when the operand travels as bytes, as it may."""
    a = np.tile(bytes_of(56), (1, 4))
    b = np.tile(bytes_of(56), (4, 1))
    types = [tensor((1, 4), "i8"), tensor((4, 1), E4M3), tensor((1, 1), "f32")]
    target = op(
        "tt.dot_scaled",
        types,
        tensor((1, 1), "f32"),
        attrs={
            "operandSegmentSizes": [1, 1, 1, 0, 0],
            "fastMath": False,
            "a_elem_type": E4M3_FMT,
            "b_elem_type": E4M3_FMT,
        },
    )
    assert one(target, [a, b, np.zeros((1, 1), np.float32)])[0, 0] == np.float32(4.0)


def test_scaled_dot_compute_type_is_f16_when_a_format_is_fp16() -> None:
    """A bf16 operand beside an fp16 one is rounded into f16, where 2**16 overflows."""
    a = from_float(np.array([[2.0**16]], np.float32), ty("bf16"))
    b = np.array([[1.0]], np.float16)
    got = run_scaled_dot(a, b, np.zeros((1, 1), np.float32), BF16_FMT, FP16_FMT)
    assert np.isposinf(got[0, 0])


def test_compute_type_rounds_the_operands() -> None:
    """The same operand pair under a bf16 compute type stays finite: 2**16 fits in bf16."""
    a = from_float(np.array([[2.0**16]], np.float32), ty("bf16"))
    b = from_float(np.array([[1.0]], np.float32), ty("bf16"))
    got = run_scaled_dot(a, b, np.zeros((1, 1), np.float32), BF16_FMT, BF16_FMT)
    assert got[0, 0] == np.float32(2.0**16)


def test_scaled_dot_rejects_a_format_with_no_frontend() -> None:
    """`e2m3` (2) and `e3m2` (3) are in the enum and nothing in Triton produces them."""
    target = scaled_dot_op((1, 2), E4M3, (2, 1), E4M3, (1, 1), a_elem_type=2, b_elem_type=E4M3_FMT)
    args = [np.tile(bytes_of(56), (1, 2)), np.tile(bytes_of(56), (2, 1))]
    with pytest.raises(Unsupported):
        one(target, [*args, np.zeros((1, 1), np.float32)])


def test_scaled_dot_rejects_a_scale_that_does_not_divide_k() -> None:
    a = np.tile(bytes_of(56), (1, 6))
    b = np.tile(bytes_of(56), (6, 1))
    with pytest.raises(Unsupported):
        run_scaled_dot(
            a,
            b,
            np.zeros((1, 1), np.float32),
            E4M3_FMT,
            E4M3_FMT,
            a_scale=bytes_of(127, 127, 127, 127, shape=(1, 4)),
        )


# --------------------------------------------------------------------- through the parser

GENERIC = """
"builtin.module"() ({
  "tt.func"() <{function_type = () -> (), sym_name = "k"}> ({
    %0 = "arith.constant"() <{value = dense<0> : tensor<2x4xi8>}> : () -> tensor<2x4xi8>
    %1 = "arith.constant"() <{value = dense<0> : tensor<8x1xi8>}> : () -> tensor<8x1xi8>
    %2 = "arith.constant"() <{value = dense<0.0> : tensor<2x2xf32>}> : () -> tensor<2x2xf32>
    %3 = "arith.constant"() <{value = dense<127> : tensor<2x1xi8>}> : () -> tensor<2x1xi8>
    %4 = "tt.dot_scaled"(%0, %1, %2, %3) <{a_elem_type = 4 : i32, b_elem_type = 4 : i32, \
fastMath = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>, rhs_k_pack = false}> \
: (tensor<2x4xi8>, tensor<8x1xi8>, tensor<2x2xf32>, tensor<2x1xi8>) -> tensor<2x2xf32>
    "tt.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
"""


def test_generic_form_carries_the_attributes_the_op_reads() -> None:
    """The printed form is the source of truth: the enum is a number and a true pack is gone.

    `ScaleDotElemType` is a plain `I32EnumAttr`, so `a_elem_type = 4 : i32` and not the
    keyword `e2m1`; `lhs_k_pack` is a `DefaultValuedAttr` and is elided when it is true, so
    the absent one has to read as true. The rhs here packs along N, and N doubles to 4.
    """
    target = next(
        o
        for o in mlir.parse(GENERIC).funcs["k"].regions[0].blocks[0].ops
        if o.name == "tt.dot_scaled"
    )
    assert target.attrs["a_elem_type"] == E2M1 and target.attrs["b_elem_type"] == E2M1
    assert "lhs_k_pack" not in target.attrs and target.attrs["rhs_k_pack"] is False
    assert target.attrs["operandSegmentSizes"] == [1, 1, 1, 1, 0]

    a = bytes_of(0x21, 0x43, 0x65, 0x07, 0x12, 0x34, 0x56, 0x70, shape=(2, 4))
    b = bytes_of(*range(0x10, 0x18), shape=(8, 1))
    scale, c = bytes_of(128, 126, shape=(2, 1)), np.zeros((2, 2), np.float32)
    got = one(target, [a, b, c, scale])
    want = reference_scaled_dot(a, b, c, E2M1, E2M1, a_scale=scale, rhs_k_pack=False)
    assert np.array_equal(got, want.astype(np.float32))


def test_scaled_dot_e8m0_byte_zero_is_the_minimum_scale_in_the_bf16_path() -> None:
    """`test_scaled_dot_minimum_scale`: a bf16 lhs of 2**112, an e4m3 rhs of 256 under scale
    byte 0, which is 2**-127 since triton#11624, so the 32 products of 2**-7 sum to 0.25."""
    a = np.full((1, 32), 0x7780, dtype=np.uint16)  # bf16 2**112
    b = np.full((32, 1), 0x78, dtype=np.uint8).astype(np.int8)  # e4m3 256.0
    scale = bytes_of(0, shape=(1, 1))
    c = np.zeros((1, 1), np.float32)
    got = run_scaled_dot(a, b, c, BF16_FMT, E4M3_FMT, b_scale=scale)
    assert got[0, 0] == np.float32(0.25)
    want = reference_scaled_dot(a, b, c, BF16_FMT, E4M3_FMT, b_scale=scale)
    assert np.array_equal(got, want.astype(np.float32))


def test_scaled_dot_e8m0_byte_zero_is_zero_in_the_f16_path() -> None:
    """The same scale byte under an f16 lhs goes through an f32 exponent field, so it is 0."""
    a = np.ones((1, 32), dtype=np.float16)
    b = np.full((32, 1), 0x78, dtype=np.uint8).astype(np.int8)
    scale = bytes_of(0, shape=(1, 1))
    c = np.zeros((1, 1), np.float32)
    got = run_scaled_dot(a, b, c, FP16_FMT, E4M3_FMT, b_scale=scale)
    assert got[0, 0] == np.float32(0.0)
    want = reference_scaled_dot(a, b, c, FP16_FMT, E4M3_FMT, b_scale=scale)
    assert np.array_equal(got, want.astype(np.float32))
