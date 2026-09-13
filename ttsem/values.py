"""Values of the semantics: what an SSA name is bound to, and how a `Type` becomes numpy.

A value is one of
  * `np.ndarray` for tensors and for scalars (a scalar is a 0-d array),
  * `Descriptor` for `!tt.tensordesc`,
  * `MemDesc` for `!ttg.memdesc`.

Element encodings follow `triton.runtime.interpreter`: `i1` is `np.bool_`, `bf16` and the fp8
kinds are the raw bit patterns in `np.uint16` / `np.uint8`, `f16`/`f32`/`f64` are the native
numpy floats, and a pointer is an `np.int64` byte address. Integers are stored *signed* at
their width, because MLIR integer types are signless and the operation, not the type, carries
the signedness (`arith.divsi` vs `arith.divui`); see `SEMANTICS.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ttsem.ir_types import Type


class Unsupported(Exception):
    """An op the semantics does not model. Carries the op name and, if known, its text."""

    def __init__(self, name: str, detail: str = "") -> None:
        super().__init__(f"{name}{': ' + detail if detail else ''}")
        self.name = name
        self.detail = detail


class Poison(Exception):
    """An operation whose result MLIR leaves undefined (`sdiv` by zero, `INT_MIN / -1`)."""


@dataclass
class Descriptor:
    """A `!tt.tensordesc`: a base address plus the shape, element strides and block shape of
    the global tensor it addresses. Loads clip to `shape` and fill with `pad`."""

    base: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    block_shape: tuple[int, ...]
    elem: Type
    pad: str = "zero"
    # The IR's integers are signless; a host descriptor over a `uint32` tensor and a
    # tensormap with an unsigned data type say so here, and `min`/`max` reductions listen.
    unsigned: bool = False


@dataclass
class MemDesc:
    """A `!ttg.memdesc`: shared memory as a plain numpy buffer.

    `data` is a view; views made by `memdesc_subview` / `memdesc_index` alias the allocation,
    so a `local_store` through one is visible through the others, as on the device.
    """

    data: np.ndarray
    elem: Type
    attrs: dict[str, object] = field(default_factory=dict)

    def with_data(self, data: np.ndarray) -> MemDesc:
        return MemDesc(data, self.elem, self.attrs)


Value = np.ndarray | Descriptor | MemDesc


# name -> (exponent bits, mantissa bits, exponent bias, family)
# family: "ieee" has infinities and NaNs at the all-ones exponent; "fn" has no infinity and
# reserves the all-ones significand as NaN; "fnuz" has no infinity, no negative zero, and
# 0x80 alone is NaN.
FLOAT_FORMATS: dict[str, tuple[int, int, int, str]] = {
    "f16": (5, 10, 15, "ieee"),
    "bf16": (8, 7, 127, "ieee"),
    "f32": (8, 23, 127, "ieee"),
    "f64": (11, 52, 1023, "ieee"),
    "f8E5M2": (5, 2, 15, "ieee"),
    "f8E4M3": (4, 3, 7, "fn"),
    "f8E4M3FN": (4, 3, 7, "fn"),
    "f8E4M3B15": (4, 3, 15, "ieee"),
    "f8E4M3B11FNUZ": (4, 3, 11, "fnuz"),
    "f8E4M3FNUZ": (4, 3, 8, "fnuz"),
    "f8E5M2FNUZ": (5, 2, 16, "fnuz"),
}
FP8_NAMES = tuple(n for n in FLOAT_FORMATS if n.startswith("f8"))
NATIVE_FLOATS = {"f16": np.float16, "f32": np.float32, "f64": np.float64}
INT_WIDTHS = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}
UINT_WIDTHS = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}


def is_float(ty: Type) -> bool:
    return ty.scalar.name in FLOAT_FORMATS


def is_bool(ty: Type) -> bool:
    s = ty.scalar
    return s.kind == "int" and s.width == 1


def is_token(ty: Type) -> bool:
    """A type that carries ordering and no data: `!ttg.async.token` and its relatives.

    The module the pipeliner produces has `!ttg.async.token` block arguments and `ub.poison`
    of that type, so a value of one has to exist. Level 1 has already run every asynchronous
    copy synchronously and in program order, so nothing reads a token: it is an opaque byte.
    """
    return ty.scalar.kind == "other" and ty.scalar.name.rstrip(">").endswith("token")


def to_numpy(ty: Type) -> np.dtype:
    """The numpy dtype an SSA value of this type is stored in."""
    s = ty.scalar
    if is_token(s):
        return np.dtype(np.int8)
    if s.kind == "ptr":
        return np.dtype(np.int64)
    if s.kind == "index":
        return np.dtype(np.int64)
    if s.kind == "int":
        if s.width == 1:
            return np.dtype(np.bool_)
        if s.width not in INT_WIDTHS:
            raise Unsupported("type", f"integer width {s.width}")
        return np.dtype(INT_WIDTHS[s.width])
    if s.name in NATIVE_FLOATS:
        return np.dtype(NATIVE_FLOATS[s.name])
    if s.name == "bf16":
        return np.dtype(np.uint16)
    if s.name in FP8_NAMES:
        return np.dtype(np.uint8)
    raise Unsupported("type", s.name or s.kind)


def int_dtype(ty: Type) -> np.dtype:
    """The signed integer dtype of the same storage width, for bit manipulation."""
    bits = to_numpy(ty).itemsize * 8
    return np.dtype(INT_WIDTHS[bits])


def uint_dtype(ty: Type) -> np.dtype:
    bits = to_numpy(ty).itemsize * 8
    return np.dtype(UINT_WIDTHS[bits])


def viewable(x: np.ndarray) -> np.ndarray:
    """A contiguous array with the *same shape*, ready for `.view`.

    `np.ascontiguousarray` promotes a 0-d array to 1-d, and a scalar here is a 0-d array: a
    bf16 add of two scalars must not come back with shape `(1,)`, because the store that
    consumes it broadcasts against a 0-d address. A 0-d array is always contiguous, so this
    returns it unchanged.
    """
    x = np.asarray(x)
    return x if x.flags["C_CONTIGUOUS"] else x.copy()


def to_bits(x: np.ndarray, ty: Type) -> np.ndarray:
    """Reinterpret a value as unsigned integers of the same width."""
    return viewable(x).view(uint_dtype(ty))


def from_bits(bits: np.ndarray, ty: Type) -> np.ndarray:
    """Reinterpret unsigned integers of the storage width as a value of `ty`."""
    return viewable(np.asarray(bits).astype(uint_dtype(ty), copy=False)).view(to_numpy(ty))


def float_bits(bits: int, ty: Type) -> np.ndarray:
    """A 0-d value of `ty` holding the raw bit pattern `bits` (what `0x7FC00000` prints)."""
    udtype = uint_dtype(ty)
    masked = int(bits) & ((1 << (udtype.itemsize * 8)) - 1)
    return from_bits(np.array(masked, dtype=udtype), ty)


def _fp8_table(name: str) -> np.ndarray:
    """The 256 float32 values of an 8-bit float kind, indexed by the byte."""
    ebits, mbits, bias, family = FLOAT_FORMATS[name]
    codes = np.arange(256, dtype=np.uint32)
    sign = np.where(codes >> 7 != 0, -1.0, 1.0).astype(np.float64)
    exp = ((codes >> mbits) & ((1 << ebits) - 1)).astype(np.int64)
    man = (codes & ((1 << mbits) - 1)).astype(np.float64)
    normal = sign * np.ldexp(1.0 + man / (1 << mbits), exp - bias)
    sub = sign * np.ldexp(man / (1 << mbits), 1 - bias)
    out = np.where(exp == 0, sub, normal)
    top = (1 << ebits) - 1
    if family == "ieee":
        out = np.where((exp == top) & (man == 0), sign * np.inf, out)
        out = np.where((exp == top) & (man != 0), np.nan, out)
    elif family == "fn":
        out = np.where((exp == top) & (man == (1 << mbits) - 1), np.nan, out)
    else:  # fnuz: 0x80 is the only NaN and there is no negative zero
        out = np.where(codes == 0x80, np.nan, out)
    return out.astype(np.float32)


def _fp8_encoder(table: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """The sorted finite values of a kind, their codes, and the NaN and infinity codes."""
    codes = np.nonzero(np.isfinite(table))[0].astype(np.uint8)
    order = np.argsort(table[codes], kind="stable")
    nan = int(np.nonzero(np.isnan(table))[0][0])
    pos = np.nonzero(np.isposinf(table))[0]
    neg = np.nonzero(np.isneginf(table))[0]
    pos_inf = int(pos[0]) if len(pos) else nan
    neg_inf = int(neg[0]) if len(neg) else nan
    return table[codes][order], codes[order], nan, pos_inf, neg_inf


_FP8_TABLES: dict[str, np.ndarray] = {n: _fp8_table(n) for n in FP8_NAMES}
_FP8_ENCODERS = {n: _fp8_encoder(t) for n, t in _FP8_TABLES.items()}


def _fp8_encode(x: np.ndarray, name: str) -> np.ndarray:
    """Round float32 to the nearest value of an 8-bit kind, ties to the even code.

    A finite input larger than the largest finite value saturates; only a real infinity
    becomes an infinity, and only for the kinds that have one.
    """
    vals, codes, nan_code, pos_inf, neg_inf = _FP8_ENCODERS[name]
    flat = np.asarray(x, dtype=np.float32).reshape(-1)
    hi = np.clip(np.searchsorted(vals, flat), 1, len(vals) - 1)
    lo = hi - 1
    dlo, dhi = np.abs(flat - vals[lo]), np.abs(flat - vals[hi])
    take_hi = (dhi < dlo) | ((dhi == dlo) & (codes[hi] % 2 == 0))
    out = np.where(take_hi, codes[hi], codes[lo]).astype(np.uint8)
    out = np.where(flat <= vals[0], codes[0], out)
    out = np.where(flat >= vals[-1], codes[-1], out)
    out = np.where(np.isposinf(flat), np.uint8(pos_inf), out)
    out = np.where(np.isneginf(flat), np.uint8(neg_inf), out)
    # The two zero codes are the same value, so the nearest-value search cannot tell them
    # apart and would give a tiny positive the code its stable order happens to put first.
    # The result keeps the sign of the input, as the hardware does (test_typeconvert_downcast
    # on sm_120 against the 3.8.0 wheel: our +0 where the device had -0, and the reverse).
    is_zero = _FP8_TABLES[name][out] == 0
    has_negative_zero = FLOAT_FORMATS[name][3] != "fnuz"
    zero_code = np.where(np.signbit(flat) & has_negative_zero, np.uint8(0x80), np.uint8(0))
    out = np.where(is_zero, zero_code, out)
    return np.where(np.isnan(flat), np.uint8(nan_code), out).astype(np.uint8).reshape(np.shape(x))


# The 16 values of an `e2m1` nibble, indexed by the nibble itself: exponent bits 2:1 with bias
# 1, one mantissa bit, one sign bit, so the magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4, 6. This is
# `positive_e2m1_lut` of `triton.runtime.interpreter` mirrored onto the sign bit, and the
# negative zero at index 8 is kept: `_e2m1_to_f32` builds it the same way, with `np.where`.
E2M1_VALUES: np.ndarray = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def unpack_fp4(packed: np.ndarray, axis: int) -> np.ndarray:
    """Decode two `e2m1` values per byte along `axis` to float32, the low nibble first.

    `ttg.fp4_to_fp` says it: "the lower 4 bits of the i8s represent the first fp4 element, and
    the upper 4 bits the second"; `_e2m1_to_f32` of the shipped interpreter writes the low
    nibbles at the even positions of the last axis. The length of `axis` doubles.
    """
    data = np.moveaxis(viewable(np.asarray(packed)).view(np.uint8), axis, -1)
    nibbles = np.empty(data.shape[:-1] + (data.shape[-1] * 2,), dtype=np.uint8)
    nibbles[..., 0::2] = data & np.uint8(0x0F)
    nibbles[..., 1::2] = data >> np.uint8(4)
    return np.moveaxis(E2M1_VALUES[nibbles], -1, axis)


def e8m0_to_float(scale: np.ndarray) -> np.ndarray:
    """An `e8m0` scale byte as a float32 power of two: the byte *is* an f32 exponent field.

    Byte `b` becomes the float32 whose bit pattern is `b << 23`, so 127 is 1.0 and 0 is +0.0
    rather than the `2**-127` of the microscaling specification: that is the f16 path of
    `DecomposeScaledBlocked::scaleTo16`; the bf16 path reads 0 as `2**-127` since triton#11624,
    and `ops._scale_factors` applies that rule. 255 comes out as an infinity here, and the caller
    turns it into a NaN, which is where the device (`DecomposeScaledBlocked::maskNan` selects a
    NaN) and the shipped interpreter (which stops at the shift) disagree.
    """
    bits = viewable(np.asarray(scale)).view(np.uint8).astype(np.uint32) << np.uint32(23)
    return bits.view(np.float32)


def to_float(x: np.ndarray, ty: Type) -> np.ndarray:
    """Decode a stored float value into a numpy float array it can be computed in.

    `f16`, `f32` and `f64` are already numpy floats and are returned unchanged; `bf16` and the
    fp8 kinds widen to `float32`.
    """
    name = ty.scalar.name
    if name in NATIVE_FLOATS:
        return x
    if name == "bf16":
        # a typed shift: with a Python int, numpy 1.x turns a 0-d uint32 into an int64 scalar
        return np.asarray(to_bits(x, ty).astype(np.uint32) << np.uint32(16)).view(np.float32)
    if name in FP8_NAMES:
        return _FP8_TABLES[name][to_bits(x, ty)]
    raise Unsupported("type", f"not a float: {name}")


def from_float(x: np.ndarray, ty: Type) -> np.ndarray:
    """Round a numpy float array back into the storage encoding of `ty` (RTNE)."""
    name = ty.scalar.name
    if name in NATIVE_FLOATS:
        return np.asarray(x).astype(NATIVE_FLOATS[name])
    if name == "bf16":
        f32 = np.asarray(x, dtype=np.float32)
        u = viewable(f32).view(np.uint32)
        bf = ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16).astype(np.uint16)
        quiet = (u >> 16).astype(np.uint16) | np.uint16(0x0040)
        return np.where(np.isnan(f32), quiet, bf).astype(np.uint16)
    if name in FP8_NAMES:
        return _fp8_encode(np.asarray(x, dtype=np.float32), name)
    raise Unsupported("type", f"not a float: {name}")


def max_finite(ty: Type) -> float:
    """The largest finite value of a float type, as a Python float."""
    name = ty.scalar.name
    if name in NATIVE_FLOATS:
        return float(np.finfo(NATIVE_FLOATS[name]).max)
    if name == "bf16":
        return float(np.array([0x7F7F0000], dtype=np.uint32).view(np.float32)[0])
    if name in FP8_NAMES:
        return float(_FP8_ENCODERS[name][0][-1])
    raise Unsupported("type", f"not a float: {name}")


def from_float_rtz(x: np.ndarray, ty: Type) -> np.ndarray:
    """Round a numpy float array into the storage encoding of `ty` toward zero.

    Round to nearest first, then step one representable value toward zero wherever the
    nearest one overshot in magnitude: for a native dtype with `nextafter`, for `bf16` and
    the fp8 kinds by taking one off the magnitude bits of the code (their codes order by
    magnitude within a sign). An infinity or NaN passes through as nearest gives it.
    """
    name = ty.scalar.name
    x = np.asarray(x)
    nearest = from_float(x, ty)
    back = to_float(nearest, ty).astype(np.float64)
    overshot = np.isfinite(x) & np.isfinite(back) & (np.abs(back) > np.abs(x))
    if name in NATIVE_FLOATS:
        dtype = NATIVE_FLOATS[name]
        stepped = np.nextafter(nearest.astype(dtype), np.zeros((), dtype=dtype))
        return np.where(overshot, stepped, nearest).astype(dtype)
    codes = np.asarray(nearest)
    sign_bit = np.array(0x8000 if name == "bf16" else 0x80, dtype=codes.dtype)
    magnitude = codes & ~sign_bit
    stepped = (codes & sign_bit) | np.where(magnitude > 0, magnitude - 1, magnitude)
    return np.where(overshot, stepped, codes).astype(codes.dtype)


def cast_to(x: np.ndarray, ty: Type) -> np.ndarray:
    """Store an already-computed numpy array as a value of `ty`, without reinterpreting bits."""
    if is_float(ty):
        return from_float(x, ty)
    return np.asarray(x).astype(to_numpy(ty))


def splat(scalar: object, ty: Type) -> np.ndarray:
    """A value of `ty` (tensor or scalar) filled with `scalar`, already in storage form."""
    shape = ty.shape if ty.kind == "tensor" and ty.shape is not None else ()
    if is_float(ty) and not isinstance(scalar, (np.ndarray, np.generic)):
        return np.broadcast_to(from_float(np.float64(scalar), ty), shape).copy()
    return np.full(shape, scalar, dtype=to_numpy(ty))
