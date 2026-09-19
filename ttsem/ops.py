"""One handler per op name, in `OPS`.

A handler has the signature `fn(interp, op, args) -> list[Value]`: `args` are the already
evaluated operands, in order, and the returned list is bound to the op's results. Ops that
transfer control (`scf.yield`, `cf.br`, `tt.return`, ...) raise a `Signal` instead, which
`Interp.run_region` catches; that keeps the op table uniform and the block runner small.

Integer types are signless, so it is the op and not the type that says whether a value is
signed: `arith.divsi` reads its operands as signed and `arith.divui` as unsigned, over the same
stored `int32`. Every integer result wraps modulo 2**width. Float ops decode to a numpy float
(`f16`/`f32`/`f64` natively, `bf16` and fp8 to `float32`), compute, and round back.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

from ttsem.ir_types import DenseAttr, FloatBits, Op, Region, Type
from ttsem.values import (
    E2M1_VALUES,
    Aref,
    ArefToken,
    Barrier,
    Descriptor,
    MemDesc,
    Poison,
    Unsupported,
    Value,
    cast_to,
    e8m0_to_float,
    float_bits,
    from_bits,
    from_float,
    from_float_rtz,
    is_bool,
    is_float,
    max_finite,
    to_bits,
    to_float,
    to_numpy,
    uint_dtype,
    unpack_fp4,
    viewable,
)

if TYPE_CHECKING:
    from ttsem.interp import Interp

Handler = Callable[["Interp", Op, list[Value]], list[Value]]
OPS: dict[str, Handler] = {}


class Signal(Exception):
    """Control leaving a region or a block."""


class Yield(Signal):
    def __init__(self, values: list[Value]) -> None:
        super().__init__("yield")
        self.values = values


class Condition(Signal):
    def __init__(self, cond: bool, values: list[Value]) -> None:
        super().__init__("condition")
        self.cond = cond
        self.values = values


class Branch(Signal):
    def __init__(self, label: str, args: list[Value]) -> None:
        super().__init__(f"br {label}")
        self.label = label
        self.args = args


def register(*names: str) -> Callable[[Handler], Handler]:
    def wrap(fn: Handler) -> Handler:
        for name in names:
            OPS[name] = fn
        return fn

    return wrap


def reject(name: str, reason: str) -> None:
    """Register an op that is deliberately out of scope."""

    def fn(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
        raise Unsupported(name, reason)

    OPS[name] = fn


def nop(*names: str) -> None:
    """Register ops that are a no-op once execution is sequential.

    A result of such an op is a token nobody reads, so it is bound to an opaque zero rather
    than to a value of its declared type.
    """

    def fn(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
        return [np.zeros((), dtype=np.int8) for _ in op.result_types]

    for name in names:
        OPS[name] = fn


# --------------------------------------------------------------------------- helpers


def _attr(op: Op, key: str, default: object = None) -> object:
    return op.attrs.get(key, default)


def _int_attr(op: Op, key: str, default: int = 0) -> int:
    v = op.attrs.get(key, default)
    return int(v) if isinstance(v, (int, float, np.generic)) else default


def _int_list(op: Op, key: str) -> list[int]:
    v = op.attrs.get(key)
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    raise Unsupported(op.name, f"attribute {key} is not a list: {v!r}")


def _rty(op: Op, i: int = 0) -> Type:
    return op.result_types[i]


def _segments(op: Op, args: list[Value], n: int) -> list[list[Value]]:
    """Split the operands into the op's `n` declared groups by `operandSegmentSizes`."""
    sizes = op.attrs.get("operandSegmentSizes")
    if not isinstance(sizes, (list, tuple)) or len(sizes) != n:
        raise Unsupported(op.name, f"needs operandSegmentSizes with {n} groups, got {sizes!r}")
    groups, start = [], 0
    for size in sizes:
        groups.append(args[start : start + int(size)])
        start += int(size)
    return groups


def _typed_segments(op: Op, args: list[Value], n: int) -> list[tuple[Value, Type] | None]:
    """The op's `n` declared groups as `(value, type)` pairs, `None` for an empty group.

    For an op whose groups hold one operand at most (`tt.dot_scaled`), where the type of the
    third group is not `operand_types[2]` unless every earlier optional group is present:
    only the operands that exist are printed.
    """
    pairs: list[tuple[Value, Type] | None] = []
    start = 0
    for group in _segments(op, args, n):
        pairs.append((group[0], op.operand_types[start]) if group else None)
        start += len(group)
    return pairs


def _unsigned(x: np.ndarray) -> np.ndarray:
    """The same bits read as unsigned, with the shape kept (see `values.viewable`)."""
    if x.dtype == np.bool_:
        return x.astype(np.uint8)
    return viewable(x).view(np.dtype(f"uint{x.dtype.itemsize * 8}"))


def _scalar(x: Value) -> int:
    return int(np.asarray(x).reshape(-1)[0])


def _out(x: np.ndarray, op: Op, i: int = 0) -> list[Value]:
    return [cast_to(x, _rty(op, i))]


# --------------------------------------------------------------------------- arith


def _constant_element(value: object, ty: Type) -> np.ndarray:
    """One element of a constant, as a 0-d value already in the storage encoding of `ty`.

    A `FloatBits` is a bit pattern MLIR printed in hex (`dense<0x7FC0> : tensor<...xbf16>` is
    a NaN, `0xFF800000 : f32` is minus infinity) and is written straight into the storage,
    which is the only way a NaN payload or a signalling NaN survives.
    """
    if isinstance(value, FloatBits):
        if not is_float(ty):
            raise Unsupported("arith.constant", f"hex float literal for {ty.scalar.name}")
        return float_bits(value.bits, ty)
    if is_float(ty):
        return from_float(np.float64(value), ty)
    return np.asarray(value).astype(to_numpy(ty))


def _flatten(value: object) -> list[object]:
    """A `dense<[[1, 2], [3, 4]]>` payload as one flat list, in row-major order."""
    if not isinstance(value, (list, tuple)):
        return [value]
    out: list[object] = []
    for item in value:
        out.extend(_flatten(item))
    return out


@register("arith.constant")
def _constant(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    ty = _rty(op)
    raw = _attr(op, "value")
    if isinstance(raw, DenseAttr):
        raw = raw.value
    shape = tuple(ty.shape) if ty.kind == "tensor" and ty.shape is not None else ()
    if not isinstance(raw, (list, tuple)):
        return [np.broadcast_to(_constant_element(raw, ty), shape).copy()]
    flat = _flatten(raw)
    if any(isinstance(v, FloatBits) for v in flat):
        out = np.stack([_constant_element(v, ty) for v in flat])
    else:
        dtype = np.float64 if is_float(ty) else np.int64
        out = cast_to(np.array(flat, dtype=dtype), ty)
    return [np.asarray(out, dtype=to_numpy(ty)).reshape(shape)]


# The ops that read their integer operands as unsigned. Only `i1` needs to know: `true` is
# `-1` to a signed op and `1` to an unsigned one, and `np.bool_` is neither.
_UNSIGNED_INT_BINOPS = frozenset(
    {"arith.divui", "arith.remui", "arith.maxui", "arith.minui", "arith.shrui", "arith.ceildivui"}
)


def _int_binop(fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> Handler:
    def handler(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
        a, b = np.asarray(args[0]), np.asarray(args[1])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if a.dtype == np.bool_ or b.dtype == np.bool_:
                # `i1` is a 1-bit integer whose results wrap mod 2, and numpy's bool arithmetic
                # is not that: `np.add` is a logical or (`True + True` would be `True`) and
                # `np.subtract` refuses to run at all. So the operands are widened, the op runs
                # at that width, and `_narrow` keeps the low bit, which is what `arith.trunci`
                # to `i1` means. `test_int1_bin_op_wraparound` and triton-lang/triton#10919.
                signed = op.name not in _UNSIGNED_INT_BINOPS
                return _narrow(np.asarray(fn(_widen(a, signed), _widen(b, signed))), op)
            return _out(fn(a, b), op)

    return handler


def _divide(a: np.ndarray, b: np.ndarray, signed: bool, rem: bool) -> np.ndarray:
    if np.any(b == 0):
        raise Poison("integer division by zero")
    if not signed:
        au, bu = _unsigned(a), _unsigned(b)
        return (au % bu if rem else au // bu).astype(a.dtype)
    lo = np.iinfo(a.dtype).min
    if np.any((a == lo) & (b == -1)):
        raise Poison("signed division overflow (MIN / -1)")
    r = np.fmod(a, b)
    return r if rem else (a - r) // b


def _rounded_divide(a: np.ndarray, b: np.ndarray, signed: bool, up: bool) -> np.ndarray:
    """`arith.floordivsi` (toward minus infinity) and `arith.ceildiv{si,ui}` (toward plus).

    Neither is `divsi`, which truncates toward zero: they differ from it, and from each other,
    on every inexact division with a negative operand. numpy's `//` already floors, so the
    ceiling is the floor plus one whenever the remainder is not zero. The poison cases are
    `divsi`'s: a zero divisor, and `INT_MIN / -1` for the signed forms.
    """
    if np.any(b == 0):
        raise Poison("integer division by zero")
    if signed:
        lo = np.iinfo(a.dtype).min
        if np.any((a == lo) & (b == -1)):
            raise Poison("signed division overflow (MIN / -1)")
        x, y = a, b
    else:
        x, y = _unsigned(a), _unsigned(b)
    quotient = np.floor_divide(x, y)
    if up:
        quotient = quotient + (np.remainder(x, y) != 0)
    return quotient.astype(a.dtype)


def _shift(x: np.ndarray, s: np.ndarray, kind: str) -> np.ndarray:
    """A shift by at least the width gives 0, or the sign for `shrsi`."""
    bits = max(8, x.dtype.itemsize * 8)
    amt = _unsigned(s).astype(np.uint64)
    big = amt >= bits
    n = np.where(big, np.uint64(0), amt)
    if kind == "shru":
        val = np.right_shift(_unsigned(x).astype(np.uint64), n).astype(np.int64)
        return np.where(big, np.int64(0), val)
    small = n.astype(np.int64)
    if kind == "shl":
        return np.where(big, np.int64(0), np.left_shift(x.astype(np.int64), small))
    fill = np.where(x < 0, np.int64(-1), np.int64(0))
    return np.where(big, fill, np.right_shift(x.astype(np.int64), small))


_CMPI = {
    0: lambda a, b: a == b,
    1: lambda a, b: a != b,
    2: lambda a, b: a < b,
    3: lambda a, b: a <= b,
    4: lambda a, b: a > b,
    5: lambda a, b: a >= b,
    6: lambda a, b: _unsigned(a) < _unsigned(b),
    7: lambda a, b: _unsigned(a) <= _unsigned(b),
    8: lambda a, b: _unsigned(a) > _unsigned(b),
    9: lambda a, b: _unsigned(a) >= _unsigned(b),
}


_SIGNED_CMPI = (2, 3, 4, 5)  # slt, sle, sgt, sge


@register("arith.cmpi")
def _cmpi(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    pred = _int_attr(op, "predicate", -1)
    if pred not in _CMPI:
        raise Unsupported(op.name, f"predicate {pred}")
    a, b = np.asarray(args[0]), np.asarray(args[1])
    if pred in _SIGNED_CMPI and (a.dtype == np.bool_ or b.dtype == np.bool_):
        a, b = _widen(a, True), _widen(b, True)  # a signed `i1` `true` is -1, so it is the least
    return _out(_CMPI[pred](a, b), op)


# ordered / unordered float predicates of arith::CmpFPredicate
_CMPF: dict[int, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    1: lambda a, b: a == b,
    2: lambda a, b: a > b,
    3: lambda a, b: a >= b,
    4: lambda a, b: a < b,
    5: lambda a, b: a <= b,
    6: lambda a, b: (a != b) & ~(np.isnan(a) | np.isnan(b)),
}
_UNORDERED = {8: 1, 9: 2, 10: 3, 11: 4, 12: 5, 13: 6}


@register("arith.cmpf")
def _cmpf(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    pred = _int_attr(op, "predicate", -1)
    a, b = (to_float(x, op.operand_types[i]) for i, x in enumerate(args[:2]))
    unordered = np.isnan(a) | np.isnan(b)
    if pred == 0:
        return _out(np.zeros(np.broadcast(a, b).shape, bool), op)
    if pred == 15:
        return _out(np.ones(np.broadcast(a, b).shape, bool), op)
    if pred == 7:
        return _out(~unordered, op)
    if pred == 14:
        return _out(unordered, op)
    if pred in _CMPF:
        return _out(_CMPF[pred](a, b) & ~unordered, op)
    if pred in _UNORDERED:
        return _out(_CMPF[_UNORDERED[pred]](a, b) | unordered, op)
    raise Unsupported(op.name, f"predicate {pred}")


def _float_binop(fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> Handler:
    def handler(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
        a = to_float(args[0], op.operand_types[0])
        b = to_float(args[1], op.operand_types[1])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return [from_float(fn(a, b), _rty(op))]

    return handler


def _float_unop(fn: Callable[[np.ndarray], np.ndarray]) -> Handler:
    def handler(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
        x = to_float(args[0], op.operand_types[0])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return [from_float(fn(x), _rty(op))]

    return handler


for _name, _fn in {
    "arith.addi": np.add,
    "arith.subi": np.subtract,
    "arith.muli": np.multiply,
    "arith.andi": np.bitwise_and,
    "arith.ori": np.bitwise_or,
    "arith.xori": np.bitwise_xor,
    "arith.maxsi": np.maximum,
    "arith.minsi": np.minimum,
}.items():
    OPS[_name] = _int_binop(_fn)

OPS["arith.maxui"] = _int_binop(lambda a, b: np.maximum(_unsigned(a), _unsigned(b)))
OPS["arith.minui"] = _int_binop(lambda a, b: np.minimum(_unsigned(a), _unsigned(b)))
OPS["arith.divsi"] = _int_binop(lambda a, b: _divide(a, b, True, False))
OPS["arith.divui"] = _int_binop(lambda a, b: _divide(a, b, False, False))
OPS["arith.remsi"] = _int_binop(lambda a, b: _divide(a, b, True, True))
OPS["arith.floordivsi"] = _int_binop(lambda a, b: _rounded_divide(a, b, True, False))
OPS["arith.ceildivsi"] = _int_binop(lambda a, b: _rounded_divide(a, b, True, True))
OPS["arith.ceildivui"] = _int_binop(lambda a, b: _rounded_divide(a, b, False, True))
OPS["arith.remui"] = _int_binop(lambda a, b: _divide(a, b, False, True))
OPS["arith.shli"] = _int_binop(lambda a, b: _shift(a, b, "shl"))
OPS["arith.shrsi"] = _int_binop(lambda a, b: _shift(a, b, "shrs"))
OPS["arith.shrui"] = _int_binop(lambda a, b: _shift(a, b, "shru"))

for _name, _fn in {
    "arith.addf": np.add,
    "arith.subf": np.subtract,
    "arith.mulf": np.multiply,
    "arith.divf": np.divide,
    "arith.remf": np.fmod,  # LLVM frem keeps the sign of the dividend, like C fmod
    "tt.precise_divf": np.divide,
    "arith.maxnumf": np.fmax,
    "arith.minnumf": np.fmin,
    "arith.maximumf": np.maximum,
    "arith.minimumf": np.minimum,
}.items():
    OPS[_name] = _float_binop(_fn)

OPS["arith.negf"] = _float_unop(np.negative)


@register("arith.select", "tt.select")
def _select(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    if isinstance(args[1], (MemDesc, Descriptor)) or isinstance(args[2], (MemDesc, Descriptor)):
        # a scalar condition choosing between two shared views, or between two descriptors
        # (a loop-carried descriptor: test_make_tensor_descriptor_loop_carried): one of them
        return [args[1] if bool(np.asarray(args[0]).reshape(-1)[0]) else args[2]]
    cond, a, b = args[0], args[1], args[2]
    return [np.asarray(np.where(cond, a, b), dtype=to_numpy(_rty(op)))]


def _widen(x: np.ndarray, signed: bool) -> np.ndarray:
    """An integer value as int64. `i1` sign-extends to -1 and zero-extends to 1, as in LLVM."""
    if x.dtype == np.bool_:
        return np.where(x, np.int64(-1 if signed else 1), np.int64(0))
    if signed:
        return x.astype(np.int64)
    return _unsigned(x).astype(np.uint64).astype(np.int64)


def _narrow(x: np.ndarray, op: Op) -> list[Value]:
    """Store an int64 into the result type; truncation to `i1` keeps the low bit."""
    if is_bool(_rty(op)):
        return [(x & 1).astype(np.bool_)]
    return _out(x, op)


@register("arith.extsi", "arith.trunci", "arith.index_cast")
def _int_resize(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    return _narrow(_widen(np.asarray(args[0]), True), op)


@register("arith.extui", "arith.index_castui")
def _extui(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    return _narrow(_widen(np.asarray(args[0]), False), op)


@register("arith.sitofp")
def _sitofp(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    return [from_float(_widen(np.asarray(args[0]), True).astype(np.float64), _rty(op))]


@register("arith.uitofp")
def _uitofp(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    return [from_float(_unsigned(np.asarray(args[0])).astype(np.float64), _rty(op))]


@register("arith.fptosi")
def _fptosi(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    return _narrow(to_float(args[0], op.operand_types[0]).astype(np.int64), op)


@register("arith.fptoui")
def _fptoui(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    x = to_float(args[0], op.operand_types[0])
    return [np.asarray(x.astype(np.uint64)).astype(to_numpy(_rty(op)))]


@register("arith.extf", "arith.truncf")
def _float_resize(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    return [from_float(to_float(args[0], op.operand_types[0]), _rty(op))]


@register("arith.bitcast", "tt.bitcast", "tt.ptr_to_int", "tt.int_to_ptr")
def _bitcast(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    src, dst = op.operand_types[0], _rty(op)
    bits = to_bits(np.asarray(args[0]), src)
    return [viewable(bits.astype(uint_dtype(dst), copy=False)).view(to_numpy(dst))]


# --------------------------------------------------------------------------- math

_erf = np.vectorize(math.erf, otypes=[np.float64])

for _name, _fn in {
    "math.exp": np.exp,
    "math.exp2": np.exp2,
    "math.log": np.log,
    "math.log2": np.log2,
    "math.sqrt": np.sqrt,
    "tt.precise_sqrt": np.sqrt,
    "math.rsqrt": lambda x: 1.0 / np.sqrt(x),
    "math.sin": np.sin,
    "math.cos": np.cos,
    "math.tanh": np.tanh,
    "math.erf": _erf,
    "math.floor": np.floor,
    "math.ceil": np.ceil,
    "math.absf": np.abs,
}.items():
    OPS[_name] = _float_unop(_fn)


@register("math.absi")
def _absi(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    with np.errstate(over="ignore"):
        return _out(np.abs(np.asarray(args[0])), op)


# precision in bits, exponent of the smallest subnormal, largest exponent of a finite value
_FMA_FORMATS: dict[str, tuple[int, int, int]] = {
    "f64": (53, -1074, 1023),
    "f32": (24, -149, 127),
    "f16": (11, -24, 15),
    "bf16": (8, -133, 127),
}


def _round_to_format(exact: Fraction, negative: bool, fmt: tuple[int, int, int]) -> float:
    """Round a non-zero exact rational once, to nearest even, into a binary float format.

    `fmt` is (precision in bits, exponent of the smallest subnormal, largest finite exponent).
    The result is returned as a Python float that the format represents exactly, so the caller's
    conversion into the storage type cannot round again: `float(Fraction)` followed by a cast to
    `float32` rounds twice, and `test_math_fma_op_edge_cases` is built on the inputs where the
    two roundings disagree (`4097 * 4097 + 2**-30` sits just above a `float32` midpoint, and
    `2**128 - 2**103 - 1` just below the overflow midpoint).
    """
    p, lsb_min, emax = fmt
    x = abs(exact)
    e = x.numerator.bit_length() - x.denominator.bit_length()
    if x < Fraction(2) ** e:
        e -= 1
    elif x >= Fraction(2) ** (e + 1):
        e += 1
    lsb = max(e - (p - 1), lsb_min)
    scaled = x / Fraction(2) ** lsb
    m, rem = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * rem
    if twice > scaled.denominator or (twice == scaled.denominator and m % 2 == 1):
        m += 1
    if m == 0:
        return -0.0 if negative else 0.0
    if m * Fraction(2) ** lsb >= Fraction(2) ** (emax + 1):
        return -math.inf if negative else math.inf
    value = math.ldexp(float(m), lsb)
    return -value if negative else value


def _fma_scalar(
    a: float, b: float, c: float, fmt: tuple[int, int, int] = _FMA_FORMATS["f64"]
) -> float:
    """`a * b + c` with one rounding over the exact product and sum, as the hardware does.

    `a * b` in `float64` is not that: it rounds twice, and it overflows or underflows where
    the exact product does not (`test_math_fma_op_special_values` has `-max * max + inf`,
    which is `+inf` for a real fma and `nan` for the unfused expression, and
    `-smallest_subnormal * 0.5 + 0.0`, which is `-0.0` and not `+0.0`). `Fraction` is exact,
    and `_round_to_format` rounds it once into the result format, so the single rounding, the
    subnormals and the overflow are all right by construction, in `float32` as in `float64`.
    """
    if math.isnan(a) or math.isnan(b) or math.isnan(c):
        return math.nan
    if math.isinf(a) or math.isinf(b):
        if a == 0.0 or b == 0.0:
            return math.nan
        product = math.inf if (a > 0) == (b > 0) else -math.inf
        return math.nan if math.isinf(c) and (c > 0) != (product > 0) else product
    if math.isinf(c):
        return c  # a finite exact product cannot cancel an infinity
    exact = Fraction(a) * Fraction(b) + Fraction(c)
    if exact == 0:
        # IEEE 754: an exact cancellation is +0; two zeros compose by the rule for addition,
        # so the result is -0 only when the product and the addend are both -0.
        product_negative = math.copysign(1.0, a) != math.copysign(1.0, b)
        addend_negative = math.copysign(1.0, c) < 0
        zeros = a * b == 0.0 and c == 0.0
        return -0.0 if zeros and product_negative and addend_negative else 0.0
    return _round_to_format(exact, exact < 0, fmt)


@register("math.fma")
def _fma(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    ty = _rty(op)
    fmt = _FMA_FORMATS.get(ty.scalar.name)
    if fmt is None:
        raise Unsupported("type", f"fma in {ty.scalar.name}")
    a, b, c = np.broadcast_arrays(
        *(to_float(x, op.operand_types[i]).astype(np.float64) for i, x in enumerate(args[:3]))
    )
    flat = (
        _fma_scalar(float(x), float(y), float(z), fmt)
        for x, y, z in zip(a.ravel(), b.ravel(), c.ravel(), strict=True)
    )
    out = np.fromiter(flat, dtype=np.float64, count=a.size).reshape(a.shape)
    return [from_float(out, ty)]


# --------------------------------------------------------------------------- tt: shapes


@register("tt.make_range")
def _make_range(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    start, end = _int_attr(op, "start"), _int_attr(op, "end")
    return _out(np.arange(start, end, dtype=np.int64), op)


@register("tt.splat")
def _splat_op(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    ty = _rty(op)
    return [np.broadcast_to(np.asarray(args[0]), ty.shape or ()).copy()]


@register("tt.unsplat")
def _unsplat(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """The inverse of `tt.splat`: a one-element tensor read back as a scalar (a 0-d array)."""
    x = np.asarray(args[0])
    if x.size != 1:
        raise Unsupported(op.name, f"source has {x.size} elements, not 1")
    return [x.reshape(())]


@register("tt.broadcast", "tt.expand_dims", "tt.reshape", "ttg.convert_layout")
def _reshape_like(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    ty = _rty(op)
    x = np.asarray(args[0])
    if ty.shape is None or tuple(ty.shape) == x.shape:
        return [x]
    if op.name == "tt.broadcast":
        return [np.broadcast_to(x, ty.shape).copy()]
    return [x.reshape(ty.shape)]


@register("tt.trans")
def _trans(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    return [np.transpose(np.asarray(args[0]), _int_list(op, "order")).copy()]


@register("tt.join")
def _join(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    return [np.stack([np.asarray(args[0]), np.asarray(args[1])], axis=-1)]


@register("tt.split")
def _split(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    x = np.asarray(args[0])
    return [x[..., 0].copy(), x[..., 1].copy()]


@register("tt.cat")
def _cat(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    return [np.concatenate([np.asarray(a) for a in args])]


@register("tt.gather")
def _gather(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    axis = _int_attr(op, "axis")
    idx = np.asarray(args[1]).astype(np.int64)
    return [np.take_along_axis(np.asarray(args[0]), idx, axis=axis)]


@register("tt.histogram")
def _histogram(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """Bin `i` counts the live lanes equal to `i`; every other value is dropped.

    "Each bin has a width of 1 and bins start at 0" (`TT_HistogramOp`), so a value at or above
    the bin count belongs to no bin. `np.histogram`'s last bin is closed on the right and puts
    it in the last one instead, which is what the shipped interpreter inherits and what
    `test_histogram_out_of_range` and `test_histogram_silent_data_corruption` show the device
    not doing: this is a deliberate divergence from `triton/runtime/interpreter.py`.
    """
    ty = _rty(op)
    bins = int((ty.shape or (0,))[0])
    src = np.asarray(args[0])
    live = np.ones(src.shape, bool)
    if len(args) > 1:
        live = live & np.broadcast_to(np.asarray(args[1], bool), src.shape)
    inside = live & (src >= 0) & (src < bins)
    counts = np.bincount(src[inside].astype(np.int64).reshape(-1), minlength=max(bins, 1))
    return _out(counts[:bins], op)


# --------------------------------------------------------------------------- tt: program


@register("tt.get_program_id")
def _program_id(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    return _out(np.int64(interp.program_id[_int_attr(op, "axis")]), op)


@register("tt.get_num_programs")
def _num_programs(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    return _out(np.int64(interp.num_programs[_int_attr(op, "axis")]), op)


# --------------------------------------------------------------------------- tt: memory


def _elem_bytes(ptr_ty: Type) -> int:
    pointee = ptr_ty.scalar.elem
    if pointee is None:
        raise Unsupported("tt.addptr", f"pointer without a pointee: {ptr_ty.name}")
    return max(1, to_numpy(pointee).itemsize)


@register("tt.addptr")
def _addptr(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    step = _elem_bytes(op.operand_types[0])
    with np.errstate(over="ignore"):
        addr = np.asarray(args[0], dtype=np.int64) + step * np.asarray(args[1]).astype(np.int64)
    return [addr]


@register("tt.load")
def _load(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    ty = _rty(op)
    ptrs = np.asarray(args[0], dtype=np.int64)
    mask = np.asarray(args[1], bool) if len(args) > 1 else None
    other = np.asarray(args[2]) if len(args) > 2 else None
    return [interp.memory.load(ptrs, mask, other, to_numpy(ty))]


@register("tt.store")
def _store(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    ptrs = np.asarray(args[0], dtype=np.int64)
    mask = np.asarray(args[2], bool) if len(args) > 2 else None
    interp.memory.store(ptrs, np.asarray(args[1]), mask)
    return []


# mlir::triton::RMWOp, which starts at 1
_RMW_KINDS = {
    1: "and",
    2: "or",
    3: "xor",
    4: "add",
    5: "fadd",
    6: "max",
    7: "min",
    8: "umax",
    9: "umin",
    10: "xchg",
}


def _bit_pattern_codec(ty: Type) -> tuple[Callable[..., np.ndarray], ...] | None:
    """`(decode, encode)` for a float type stored as a bit pattern (bf16, the fp8 kinds).

    `None` for everything whose storage *is* the value: integers, and `f16`/`f32`/`f64`, which
    numpy has native dtypes for.
    """
    if is_float(ty) and to_numpy(ty).kind == "u":
        return (lambda x: to_float(x, ty), lambda x: from_float(x, ty))
    return None


@register("tt.atomic_rmw")
def _atomic_rmw(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    raw = _attr(op, "atomic_rmw_op", _attr(op, "rmw_op"))
    kind = _RMW_KINDS.get(int(raw), "") if isinstance(raw, (int, float)) else str(raw).lower()
    if kind not in _RMW_KINDS.values():
        raise Unsupported(op.name, f"rmw op {raw!r}")
    rty = _rty(op)
    ptrs = np.asarray(args[0], dtype=np.int64)
    mask = np.asarray(args[2], bool) if len(args) > 2 else None
    val = np.asarray(args[1], dtype=to_numpy(rty))
    return [interp.memory.atomic(kind, ptrs, val, mask, codec=_bit_pattern_codec(rty))]


@register("tt.atomic_poll")
def _atomic_poll(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """One relaxed load compared with `expected`, because nothing can change under us.

    The op spins until every element equals `expected` or the shared timeout expires, and the
    docs promise each element is loaded at least once even with a zero timeout. Level 1 runs
    one program at a time, so no other program can write the flag between two iterations of
    the spin and the first load already decides the answer. Without a timeout operand a poll
    that does not match would block forever waiting for a program this schedule will only run
    later, which is a level-1 boundary and not a value to invent: it raises `Unsupported`.
    """
    ptrs = np.asarray(args[0], dtype=np.int64)
    expected = np.asarray(args[1])
    got = interp.memory.load(ptrs, None, None, expected.dtype)
    matched = got == np.broadcast_to(expected, got.shape)
    if len(args) < 3 and not bool(np.all(matched)):
        raise Unsupported(op.name, "a poll with no timeout that would wait for another program")
    return _out(matched, op)


@register("tt.atomic_cas")
def _atomic_cas(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    dtype = to_numpy(_rty(op))
    ptrs = np.asarray(args[0], dtype=np.int64)
    pair = (np.asarray(args[1], dtype=dtype), np.asarray(args[2], dtype=dtype))
    return [interp.memory.atomic("cas", ptrs, pair, None)]


# --------------------------------------------------------------------------- tt: dot


def _to_tf32(x: np.ndarray) -> np.ndarray:
    """Drop the low 13 mantissa bits, as the tensor core does before a tf32 multiply."""
    bits = viewable(np.asarray(x).astype(np.float32)).view(np.uint32)
    return (bits & np.uint32(0xFFFFE000)).view(np.float32)


def _fma_dot(op: Op) -> bool:
    """A dot whose operands carry a `dot_op` layout over a *blocked* parent is lowered to
    scalar FMAs, which multiply f32 in full whatever `inputPrecision` says (the tf32 rounding
    is the tensor core's, and this dot never reaches one): gluon's `dot_fma`, and a `tt.dot`
    the layout assignment left on the FMA path."""
    return any(
        "dot_op<" in (t.encoding or "") and "parent = #ttg.blocked<" in (t.encoding or "")
        for t in op.operand_types[:2]
    )


@register("tt.dot")
def _dot(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    ta, tb, tc = op.operand_types[0], op.operand_types[1], op.operand_types[2]
    rty = _rty(op)
    if not is_float(rty):
        prod = np.matmul(np.asarray(args[0]).astype(np.int64), np.asarray(args[1]).astype(np.int64))
        with np.errstate(over="ignore"):
            return _out(prod + np.asarray(args[2]).astype(np.int64), op)
    acc_dtype = np.float64 if rty.scalar.name == "f64" else np.float32
    a = to_float(args[0], ta).astype(acc_dtype)
    b = to_float(args[1], tb).astype(acc_dtype)
    if _int_attr(op, "inputPrecision", 0) == 0 and ta.scalar.name == "f32" and not _fma_dot(op):
        a, b = _to_tf32(a), _to_tf32(b)
    out = np.matmul(a, b, dtype=acc_dtype) + to_float(args[2], tc).astype(acc_dtype)
    return [from_float(out, rty)]


# ------------------------------------------------------------------- tt: inline PTX fragments

# `tt.elementwise_inline_asm` is PTX, which this semantics does not model in general. What the
# ecosystem actually writes is a handful of single-purpose fragments (`triton_kernels`: mxfp4
# packing, E8M0 scales, tf32 rounding, a fast exp2, a signed max; `test_core`: the tf32
# rounding), and each has a small exact meaning, measured bit for bit on an RTX PRO 6000 on
# 17 Sep 2026 (`ptxprobe.py`, tables in the tests). They are recognised by their normalised
# text; anything else stays out of scope, and the run names the fragment.

_F32_NAN_BITS = np.uint32(0x7FFF_FFFF)  # the canonical NaN these instructions produce


def _norm_asm(asm: object) -> str:
    """The fragment with braces gone and whitespace collapsed: the key of `PTX_FRAGMENTS`."""
    return " ".join(str(asm).replace("{", " ").replace("}", " ").split())


def _f32(x: object) -> np.ndarray:
    """The operand as float32, keeping a 0-d scalar 0-d (`ascontiguousarray` would make it 1-d,
    and a scalar atomic downstream cannot take a one-element vector)."""
    a = np.asarray(x, dtype=np.float32)
    return a if a.ndim == 0 else np.ascontiguousarray(a)


def _xorsign_abs(pick: Any) -> Any:
    """`max/min.NaN.xorsign.abs.f32`: the larger (smaller) magnitude with the XOR of the two
    signs; any NaN in gives the canonical NaN out."""

    def fn(a: object, b: object) -> list[np.ndarray]:
        fa, fb = _f32(a), _f32(b)
        with np.errstate(invalid="ignore"):
            mag = pick(np.abs(fa), np.abs(fb))
        sign = (fa.view(np.uint32) ^ fb.view(np.uint32)) & np.uint32(0x8000_0000)
        out = (np.asarray(mag, np.float32).view(np.uint32) & np.uint32(0x7FFF_FFFF)) | sign
        out = np.where(np.isnan(fa) | np.isnan(fb), _F32_NAN_BITS, out)
        return [out.astype(np.uint32).view(np.float32)]

    return fn


def _ex2_approx_ftz(x: object) -> list[np.ndarray]:
    """`ex2.approx.ftz.f32`: 2**x within about an ulp (the policy's `div` class covers it),
    subnormal inputs read as zero and subnormal results flushed to zero, NaN canonical."""
    fx = _f32(x)
    bits = fx.view(np.uint32)
    flushed = np.where((bits & np.uint32(0x7F80_0000)) == 0, np.float32(0.0), fx)
    with np.errstate(over="ignore", under="ignore"):
        y = np.exp2(flushed.astype(np.float64)).astype(np.float32)
    y = np.where((y.view(np.uint32) & np.uint32(0x7F80_0000)) == 0, np.float32(0.0), y)
    out = np.where(np.isnan(fx), _F32_NAN_BITS, np.asarray(y, np.float32).view(np.uint32))
    return [out.astype(np.uint32).view(np.float32)]


def _cvt_tf32(away: bool) -> Any:
    """`cvt.rn.tf32.f32` / `cvt.rna.tf32.f32`: an f32 pattern with the low 13 bits cleared,
    rounded to nearest even or to nearest away from zero. Both keep infinities. The `rn` form
    turns any NaN into 0x7FFFE000; the `rna` form keeps the NaN's sign and payload, masked."""

    def fn(x: object) -> list[np.ndarray]:
        bits = _f32(x).view(np.uint32).astype(np.uint64)
        special = (bits & np.uint64(0x7F80_0000)) == np.uint64(0x7F80_0000)
        nan = special & ((bits & np.uint64(0x007F_FFFF)) != 0)
        bias = (
            np.uint64(0x1000)
            if away
            else ((bits >> np.uint64(13)) & np.uint64(1)) + np.uint64(0xFFF)
        )
        rounded = (bits + bias) & np.uint64(0xFFFF_E000)
        out = np.where(special, bits & np.uint64(0xFFFF_E000), rounded)
        if not away:
            out = np.where(nan, np.uint64(0x7FFF_E000), out)
        return [out.astype(np.uint32).view(np.float32)]

    return fn


def _e2m1_nibble(x: np.ndarray) -> np.ndarray:
    """`cvt.rn.satfinite.e2m1x2.f32` on one operand: the nearest e2m1 magnitude, ties to the
    even code, anything past 6 (infinity included) saturated to 6, the sign kept (a negative
    zero is 0x8), and a NaN turned into +6 (0x7) whatever its sign."""
    fx = _f32(x)
    grid = E2M1_VALUES[:8].astype(np.float64)
    mag = np.abs(fx).astype(np.float64)
    hi = np.clip(np.searchsorted(grid, mag, side="left"), 0, 7)
    lo = np.clip(hi - 1, 0, 7)
    with np.errstate(invalid="ignore"):
        d_lo, d_hi = mag - grid[lo], grid[hi] - mag
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0))
    code = np.where(pick_hi, hi, lo)
    code = np.where(mag > 6.0, 7, code).astype(np.uint8)
    code = code | np.where(np.signbit(fx), np.uint8(8), np.uint8(0))
    return np.where(np.isnan(fx), np.uint8(7), code).astype(np.uint8)


def _e2m1x2_pack(hi: object, lo: object) -> list[np.ndarray]:
    """The `_downcast_to_mxfp` fragment: two f32 to one byte, the first operand in the high
    nibble, replicated four times in the b32 the kernel reads back as a u8."""
    return [(_e2m1_nibble(hi) << np.uint8(4)) | _e2m1_nibble(lo)]


def _e2m1x2_unpack(x: object) -> list[np.ndarray]:
    """The `_upcast_from_mxfp` fragment: one byte to a b32 holding two f16, the low nibble in
    the low half. Exact: every e2m1 value is an f16."""
    codes = np.asarray(x).astype(np.uint8)
    halves = E2M1_VALUES.astype(np.float16).view(np.uint16).astype(np.uint32)
    return [halves[codes & np.uint8(0xF)] | (halves[codes >> np.uint8(4)] << np.uint32(16))]


def _ue8m0_to_bf16(x: object) -> list[np.ndarray]:
    """`cvt.rn.bf16x2.ue8m0x2`: an E8M0 byte `e` is 2**(e-127), so its bf16 is `e << 7`;
    byte 0 is the subnormal 2**-127 (0x0040) and byte 255 is NaN (0x7FFF)."""
    codes = np.asarray(x).astype(np.uint8).astype(np.uint16)  # the i8 view of a byte, unsigned
    out = codes << np.uint16(7)
    out = np.where(codes == 0, np.uint16(0x0040), out)
    return [np.where(codes == 255, np.uint16(0x7FFF), out).astype(np.uint16)]


# normalised fragment -> (semantics, class of `harness.INEXACT_OPS` or None when exact)
PTX_FRAGMENTS: dict[str, tuple[Any, str | None]] = {
    _norm_asm(k): v
    for k, v in {
        "max.NaN.xorsign.abs.f32 $0, $1, $2;": (_xorsign_abs(np.maximum), None),
        "min.NaN.xorsign.abs.f32 $0, $1, $2;": (_xorsign_abs(np.minimum), None),
        "ex2.approx.ftz.f32 $0, $1;": (_ex2_approx_ftz, "div"),
        "cvt.rn.tf32.f32 $0, $1;": (_cvt_tf32(away=False), None),
        "cvt.rna.tf32.f32 $0, $1;": (_cvt_tf32(away=True), None),
        ".reg .b8 r; cvt.rn.satfinite.e2m1x2.f32 r, $1, $2; mov.b32 $0, {r, r, r, r};": (
            _e2m1x2_pack,
            None,
        ),
        ".reg .b8 in_8; .reg .f16x2 out; cvt.u8.u32 in_8, $1; cvt.rn.f16x2.e2m1x2 out, in_8; "
        "mov.b32 $0, out;": (_e2m1x2_unpack, None),
        "cvt.rn.bf16x2.ue8m0x2 $0, $1;": (_ue8m0_to_bf16, None),
    }.items()
}


# fragments whose result is a packed float format in an integer tensor: the buffer such a
# result is stored to holds pairs of that format, and the comparison decodes it as such
PACKING_FRAGMENTS: dict[str, str] = {
    _norm_asm(
        ".reg .b8 r; cvt.rn.satfinite.e2m1x2.f32 r, $1, $2; mov.b32 $0, {r, r, r, r};"
    ): "e2m1x2",
}


def inline_asm_packs(asm: object) -> str | None:
    """The packed format a known fragment produces in an integer result, or None."""
    return PACKING_FRAGMENTS.get(_norm_asm(asm))


def inline_asm_class(asm: object) -> str | None:
    """The `INEXACT_OPS` class of a known fragment, None for an exact or unknown one."""
    entry = PTX_FRAGMENTS.get(_norm_asm(asm))
    return entry[1] if entry else None


@register("tt.elementwise_inline_asm")
def _inline_asm(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """A known fragment by its text, elementwise over the operands (the packing only says
    how the lowering groups lanes); anything else is out of scope."""
    key = _norm_asm(_attr(op, "asm_string", ""))
    entry = PTX_FRAGMENTS.get(key)
    if entry is None:
        raise Unsupported(op.name, f"the semantics does not model PTX: {key[:100]}")
    results = entry[0](*[np.asarray(a) for a in args])
    # the shape is the operands' (a scalar operand is one value per program here, so no reshape
    # to the declared scalar type: `_p_matmul` folds a flexpoint scale with the scalar form)
    return [
        np.asarray(r).astype(to_numpy(t)) for r, t in zip(results, op.result_types, strict=True)
    ]


# --------------------------------------------------------------------- tt: microscaled dot

# `ScaleDotElemType` of `TritonAttrDefs.td`, by its number: the enum is a plain `I32EnumAttr`,
# so the generic form prints `a_elem_type = 4 : i32` and not the keyword. The value is the
# spelling of the type the format decodes as; `e2m1` has no MLIR type of its own, it travels
# two per byte in an `i8` tensor.
SCALED_ELEM_TYPES: dict[int, str] = {
    0: "f8E4M3FN",  # e4m3
    1: "f8E5M2",  # e5m2
    4: "e2m1",
    5: "bf16",
    6: "f16",
}
_SCALED_TYPES: dict[str, Type] = {
    name: Type("float", width=8 if name.startswith("f8") else 16, name=name)
    for name in ("f8E4M3FN", "f8E5M2", "bf16", "f16")
}


def _scaled_format(op: Op, key: str) -> str:
    code = _int_attr(op, key, -1)
    name = SCALED_ELEM_TYPES.get(code)
    if name is None:
        raise Unsupported(op.name, f"{key} = {code}: e2m3 and e3m2 have no frontend")
    return name


def _k_pack(op: Op, op_idx: int) -> bool:
    """`lhs_k_pack` / `rhs_k_pack`: they default to true and are elided when they are."""
    return bool(_attr(op, "lhs_k_pack" if op_idx == 0 else "rhs_k_pack", True))


def _scale_factors(scale: Value, ty: Type, compute: Type) -> tuple[np.ndarray, np.ndarray]:
    """A scale operand as float32 factors, and the mask of the entries that mean NaN.

    An integer scale is `e8m0` and 0xFF is its NaN; a float scale (the `f8E4M3FN` scales of
    nvfp4) decodes as itself and carries its own NaNs, which is the pair of cases
    `DecomposeScaledBlocked::scaleTo16` and `::maskNan` distinguish. Byte 0 depends on the
    compute type, as `scaleTo16` does since triton#11624: in the `bf16` path the byte becomes
    the bf16 exponent field clamped to the code 0x0040, so 0 reads as the `2**-127` of the
    specification (a bf16 subnormal); in the `f16` path the byte becomes an f32 exponent field,
    so 0 is +0.0 and the truncation to f16 keeps it there.
    """
    if is_float(ty):
        values = np.asarray(to_float(scale, ty), dtype=np.float32)
        return values, np.isnan(values)
    raw = _unsigned(np.asarray(scale))
    if raw.dtype != np.uint8:
        raise Unsupported("tt.dot_scaled", f"scale stored as {ty.scalar.name}, not a byte")
    factors = e8m0_to_float(raw)
    if compute.scalar.name == "bf16":
        factors = np.where(raw == np.uint8(0), np.float32(2.0**-127), factors)
    return factors, raw == np.uint8(0xFF)


def _scaled_operand(
    op: Op,
    value: tuple[Value, Type],
    scale: tuple[Value, Type] | None,
    fmt: str,
    op_idx: int,
    compute: Type,
) -> np.ndarray:
    """One operand of `tt.dot_scaled`, decoded, scaled, and rounded into the compute type.

    The steps are `DecomposeScaledBlocked::scaleArg`: upcast to the compute type, broadcast
    the scale over its group along K and multiply there, then replace a group whose scale is
    NaN with NaN.
    """
    value, vty = value
    rank = len(vty.shape or ())
    if rank < 2:
        raise Unsupported(op.name, "operands must be at least 2-D")
    k_dim = rank - 1 if op_idx == 0 else rank - 2
    non_k_dim = rank - 2 if op_idx == 0 else rank - 1
    if fmt == "e2m1":
        decoded = unpack_fp4(value, k_dim if _k_pack(op, op_idx) else non_k_dim)
    elif is_float(vty):
        decoded = to_float(value, vty)
    else:
        named = _SCALED_TYPES[fmt]
        if to_numpy(named).itemsize != np.asarray(value).dtype.itemsize:
            raise Unsupported(op.name, f"{fmt} operand stored as {vty.scalar.name}")
        decoded = to_float(from_bits(_unsigned(np.asarray(value)), named), named)
    with np.errstate(over="ignore", invalid="ignore"):
        decoded = to_float(from_float(decoded, compute), compute)
        if scale is None:
            return np.asarray(decoded, dtype=np.float32)

        factors, is_nan = _scale_factors(*scale, compute)
        if op_idx == 1:
            # "For some weird reason, we take the scale with shape as if it were coming from
            # the lhs even when it's the rhs": it is [..., N, K / group], so it transposes.
            factors, is_nan = np.swapaxes(factors, -1, -2), np.swapaxes(is_nan, -1, -2)
        group = decoded.shape[k_dim] // factors.shape[k_dim]
        if group * factors.shape[k_dim] != decoded.shape[k_dim]:
            raise Unsupported(op.name, f"scale {factors.shape} does not divide {decoded.shape}")
        factors = np.repeat(factors, group, axis=k_dim)
        is_nan = np.repeat(is_nan, group, axis=k_dim)
        scaled = decoded * to_float(from_float(factors, compute), compute)
        scaled = to_float(from_float(scaled, compute), compute)
    if _attr(op, "fastMath", False):
        return np.asarray(scaled, dtype=np.float32)
    nan = np.float32(np.nan)
    return np.where(np.broadcast_to(is_nan, scaled.shape), nan, scaled).astype(np.float32)


@register("tt.dot_scaled")
def _dot_scaled(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """`d = matmul(scale(a, a_scale), scale(b, b_scale)) + c`.

    Both operands are upcast to one compute type (`f16` when either format is `fp16`, else
    `bf16`, as `DecomposeScaledBlocked::getComputeType` chooses), scaled there, and then
    accumulated in the accumulator's type exactly as `tt.dot` does.
    """
    a, b, c, a_scale, b_scale = _typed_segments(op, args, 5)
    if a is None or b is None or c is None:
        raise Unsupported(op.name, "a, b and c are not optional")
    fmt_a, fmt_b = _scaled_format(op, "a_elem_type"), _scaled_format(op, "b_elem_type")
    compute = _SCALED_TYPES["f16" if "f16" in (fmt_a, fmt_b) else "bf16"]
    rty = _rty(op)
    acc_dtype = np.float64 if rty.scalar.name == "f64" else np.float32
    lhs = _scaled_operand(op, a, a_scale, fmt_a, 0, compute).astype(acc_dtype)
    rhs = _scaled_operand(op, b, b_scale, fmt_b, 1, compute).astype(acc_dtype)
    with np.errstate(invalid="ignore"):
        out = np.matmul(lhs, rhs, dtype=acc_dtype)
    return [from_float(out + to_float(c[0], c[1]).astype(acc_dtype), rty)]


@register("ttg.fp4_to_fp")
def _fp4_to_fp(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """Unpack `e2m1` nibbles along `axis` into the result's float type, low nibble first."""
    axis = _int_attr(op, "axis", 0)
    rank = len(op.operand_types[0].shape or ())
    if not 0 <= axis < rank:
        raise Unsupported(op.name, f"axis {axis} out of range for rank {rank}")
    return _out(unpack_fp4(args[0], axis), op)


# --------------------------------------------------------------------- tt: mapped regions


# libdevice, by the base name of the symbol: CUDA spells the float entry point `__nv_<name>f`
# and the double one `__nv_<name>`, so both are generated from one line here. Every one of
# these is computed in the decoded dtype of the operands, which is `float32` for the `f`
# variants: like `math.*`, they do not model the device's fast approximations, so a result
# that depends on the last bit differs and the harness reports it as `approx`.
def _erfinv(y: np.ndarray) -> np.ndarray:
    """The inverse error function: Giles' approximation, then two Newton steps on `math.erf`."""
    y = np.asarray(y, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = -np.log((1.0 - y) * (1.0 + y))
        small = w < 5.0
        ws, wl = w - 2.5, np.sqrt(np.where(small, 5.0, w)) - 3.0
        ps = 2.81022636e-08
        for c in (3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087, -0.00125372503,
                  -0.00417768164, 0.246640727, 1.50140941):  # fmt: skip
            ps = c + ps * ws
        pl = -0.000200214257
        for c in (0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773, -0.0076224613,
                  0.00943887047, 1.00167406, 2.83297682):  # fmt: skip
            pl = c + pl * wl
        x = np.where(small, ps, pl) * y
        erf = np.vectorize(math.erf, otypes=[np.float64])
        for _ in range(2):
            x = x - (erf(x) - y) / (2.0 / math.sqrt(math.pi) * np.exp(-x * x))
    x = np.where(np.abs(y) == 1.0, np.copysign(np.inf, y), x)
    return np.where(np.abs(y) > 1.0, np.nan, x)


def _lgamma(x: np.ndarray) -> np.ndarray:
    def one(v: float) -> float:
        try:
            return math.lgamma(v)
        except ValueError:  # a pole: zero and the negative integers
            return math.inf

    return np.vectorize(one, otypes=[np.float64])(np.asarray(x, dtype=np.float64))


_LIBDEVICE: dict[str, tuple[int, Callable[..., np.ndarray]]] = {
    "exp": (1, np.exp),
    "exp2": (1, np.exp2),
    "expm1": (1, np.expm1),
    "log": (1, np.log),
    "log2": (1, np.log2),
    "log10": (1, np.log10),
    "log1p": (1, np.log1p),
    "sqrt": (1, np.sqrt),
    "rsqrt": (1, lambda x: 1.0 / np.sqrt(x)),
    "cbrt": (1, np.cbrt),
    "sin": (1, np.sin),
    "cos": (1, np.cos),
    "tan": (1, np.tan),
    "asin": (1, np.arcsin),
    "acos": (1, np.arccos),
    "atan": (1, np.arctan),
    "atan2": (2, np.arctan2),
    "sinh": (1, np.sinh),
    "cosh": (1, np.cosh),
    "tanh": (1, np.tanh),
    "erf": (1, np.vectorize(math.erf, otypes=[np.float64])),
    "erfc": (1, np.vectorize(math.erfc, otypes=[np.float64])),
    "erfinv": (1, _erfinv),
    "lgamma": (1, _lgamma),
    "floor": (1, np.floor),
    "ceil": (1, np.ceil),
    "trunc": (1, np.trunc),
    "rint": (1, np.rint),  # nearest, ties to even, which is the default rounding mode
    "nearbyint": (1, np.rint),
    "round": (1, lambda x: np.trunc(x + np.copysign(0.5, x))),  # nearest, ties away from zero
    "fabs": (1, np.abs),
    "fmin": (2, np.fmin),
    "fmax": (2, np.fmax),
    "fmod": (2, np.fmod),
    "copysign": (2, np.copysign),
    "hypot": (2, np.hypot),
    "pow": (2, np.power),
    "fdiv": (2, np.divide),
    "fma": (3, lambda a, b, c: a * b + c),
}
_EXTERN: dict[str, tuple[int, Callable[..., np.ndarray]]] = {
    f"__nv_{name}{suffix}": entry for name, entry in _LIBDEVICE.items() for suffix in ("", "f")
}


def _ffs(bits: int) -> Callable[[np.ndarray], np.ndarray]:
    """`__nv_ffs`/`__nv_ffsll`: the 1-based position of the least significant set bit, 0 for 0."""

    def fn(x: np.ndarray) -> np.ndarray:
        u = x.astype(np.uint64) & np.uint64((1 << bits) - 1)
        low = u & (~u + np.uint64(1))  # isolates the lowest set bit
        pos = np.zeros(u.shape, np.int64)
        for b in range(bits):
            pos = np.where(low == np.uint64(1 << b), b + 1, pos)
        return pos.astype(np.int32)

    return fn


def _popc(bits: int) -> Callable[[np.ndarray], np.ndarray]:
    def fn(x: np.ndarray) -> np.ndarray:
        u = x.astype(np.uint64) & np.uint64((1 << bits) - 1)
        count = np.zeros(u.shape, np.int64)
        for b in range(bits):
            count += ((u >> np.uint64(b)) & np.uint64(1)).astype(np.int64)
        return count.astype(np.int32)

    return fn


def _clz(bits: int) -> Callable[[np.ndarray], np.ndarray]:
    def fn(x: np.ndarray) -> np.ndarray:
        u = x.astype(np.uint64) & np.uint64((1 << bits) - 1)
        lead = np.full(u.shape, bits, np.int64)
        for b in range(bits):  # the highest set bit wins, so the loop runs upwards
            lead = np.where((u >> np.uint64(b)) & np.uint64(1) == 1, bits - 1 - b, lead)
        return lead.astype(np.int32)

    return fn


# libdevice's integer entry points, on the operands' bits: 32-bit, and `ll` for 64-bit
_EXTERN_INT: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "__nv_ffs": _ffs(32),
    "__nv_ffsll": _ffs(64),
    "__nv_popc": _popc(32),
    "__nv_popcll": _popc(64),
    "__nv_clz": _clz(32),
    "__nv_clzll": _clz(64),
}


@register("tt.extern_elementwise")
def _extern_elementwise(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    symbol = str(_attr(op, "symbol", ""))
    int_fn = _EXTERN_INT.get(symbol)
    if int_fn is not None:
        if len(args) != 1:
            raise Unsupported(op.name, f"{symbol} takes 1 operand, got {len(args)}")
        return [int_fn(np.asarray(args[0])).astype(to_numpy(_rty(op)))]
    if symbol in ("__nv_powi", "__nv_powif"):  # a float base to an integer power
        if len(args) != 2:
            raise Unsupported(op.name, f"{symbol} takes 2 operands, got {len(args)}")
        base = to_float(args[0], op.operand_types[0])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return [from_float(np.power(base, np.asarray(args[1]).astype(base.dtype)), _rty(op))]
    if symbol in ("__nv_signbit", "__nv_signbitf", "__nv_signbitd"):  # nonzero for a set sign bit
        if len(args) != 1:
            raise Unsupported(op.name, f"{symbol} takes 1 operand, got {len(args)}")
        signed = np.signbit(to_float(args[0], op.operand_types[0]))
        return [signed.astype(to_numpy(_rty(op)))]
    if symbol in ("__nv_llrint", "__nv_llrintf"):  # nearest integer, ties to even, as an integer
        if len(args) != 1:
            raise Unsupported(op.name, f"{symbol} takes 1 operand, got {len(args)}")
        with np.errstate(invalid="ignore"):
            rounded = np.rint(to_float(args[0], op.operand_types[0]).astype(np.float64))
            return [np.nan_to_num(rounded, nan=0.0).astype(to_numpy(_rty(op)))]
    entry = _EXTERN.get(symbol)
    if entry is None:
        raise Unsupported(op.name, f"libdevice symbol {symbol!r}")
    arity, fn = entry
    if len(args) != arity:
        raise Unsupported(op.name, f"{symbol} takes {arity} operands, got {len(args)}")
    decoded = [to_float(a, op.operand_types[i]) for i, a in enumerate(args)]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return [from_float(fn(*decoded), _rty(op))]


def _straight_line(region: Region) -> bool:
    """True when the region is one block of ops that neither branch nor carry regions.

    Such a region can be run once on whole columns instead of once per element, which is what
    makes `tt.map_elementwise` over a 512-element tensor take microseconds.
    """
    return len(region.blocks) == 1 and all(
        not o.regions and not o.successors for o in region.blocks[0].ops
    )


@register("tt.map_elementwise")
def _map_elementwise(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """Apply the region to every element (to every `pack` consecutive elements).

    With `pack = k` the region takes `k` consecutive elements of each operand, operand-major
    and pack-minor, and returns `k` elements of each result in the same order: for two
    operands and two results, the block arguments are `a0, a1, b0, b1` and the returned values
    are `c0, c1, d0, d1`.
    """
    pack = max(_int_attr(op, "pack", 1), 1)
    srcs = [np.asarray(a) for a in args]
    shape = srcs[0].shape
    n = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if n % pack:
        raise Unsupported(op.name, f"{n} elements is not a multiple of pack={pack}")
    region = op.regions[0]
    lanes = [s.reshape(n // pack, pack) for s in srcs]
    outs = [np.empty((n // pack, pack), dtype=to_numpy(t)) for t in op.result_types]
    if _straight_line(region):
        columns = [lane[:, p] for lane in lanes for p in range(pack)]
        for k, value in enumerate(interp.run_region(region, list(columns))):
            outs[k // pack][:, k % pack] = np.asarray(value)
    else:
        for i in range(n // pack):
            block_args = [np.asarray(lane[i, p]) for lane in lanes for p in range(pack)]
            for k, value in enumerate(interp.run_region(region, block_args)):
                outs[k // pack][i, k % pack] = np.asarray(value)
    return [o.reshape(shape) for o in outs]


# --------------------------------------------------------------------------- tt: misc


@register("tt.clampf")
def _clampf(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    x, lo, hi = (to_float(a, op.operand_types[i]) for i, a in enumerate(args[:3]))
    propagate = _int_attr(op, "propagateNan", 0) != 0
    out = np.clip(x, lo, hi) if propagate else np.fmin(np.fmax(x, lo), hi)
    return [from_float(out, _rty(op))]


def _mulhi64(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The high 64 bits of a 64x64 unsigned product, by the schoolbook split into halves.

    numpy has no 128-bit integer, so the product is assembled from four 32x32 partial products
    exactly as hardware does; every intermediate fits in 64 bits.
    """
    mask = np.uint64(0xFFFFFFFF)
    shift = np.uint64(32)
    a_lo, a_hi = a & mask, a >> shift
    b_lo, b_hi = b & mask, b >> shift
    with np.errstate(over="ignore"):
        lo_lo = a_lo * b_lo
        cross = a_hi * b_lo + (lo_lo >> shift)
        carry = a_lo * b_hi + (cross & mask)
        return a_hi * b_hi + (cross >> shift) + (carry >> shift)


@register("tt.mulhiui")
def _mulhiui(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    a, b = _unsigned(np.asarray(args[0])), _unsigned(np.asarray(args[1]))
    bits = a.dtype.itemsize * 8
    if bits >= 64:
        return _out(_mulhi64(a.astype(np.uint64), b.astype(np.uint64)), op)
    wide = np.dtype(f"uint{bits * 2}")
    return _out((a.astype(wide) * b.astype(wide)) >> bits, op)


@register("tt.fp_to_fp")
def _fp_to_fp(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """A float conversion. A downcast saturates: the NVIDIA lowering is `cvt.rn.satfinite`
    (and `cvt.rz.satfinite` for `rounding = rtz`), so an infinite or out-of-range source becomes
    the largest finite value of its sign and only a NaN stays a NaN; `test_typeconvert_downcast`
    and its `_clamping` variant pin exactly that. An upcast is exact."""
    src_ty, dst_ty = op.operand_types[0], _rty(op)
    x = to_float(args[0], src_ty)
    rounding = _attr(op, "rounding")
    toward_zero = (
        rounding is not None and str(rounding).strip("0123456789 :i") == "" and int(rounding) == 0
    )
    if _float_width(dst_ty) >= _float_width(src_ty):
        return [from_float(x, dst_ty)]
    out = from_float_rtz(x, dst_ty) if toward_zero else from_float(x, dst_ty)
    back = to_float(out, dst_ty)
    # an infinite source, or a finite one the nearest rounding pushed to infinity or to the
    # NaN of a kind without infinities, lands on the largest finite value of its sign
    over = ~np.isnan(x) & (np.isinf(x) | np.isinf(back) | (np.isnan(back) & np.isfinite(x)))
    limit = max_finite(dst_ty)
    saturated = from_float(np.where(over, np.copysign(limit, x), 0.0).astype(np.float32), dst_ty)
    return [np.where(over, saturated, out).astype(out.dtype)]


def _float_width(ty: Type) -> int:
    """Bits of a float type, so a conversion knows whether it narrows."""
    return int(ty.scalar.width or 0)


@register("tt.assert")
def _assert(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    if not np.all(np.asarray(args[0], bool)):
        raise AssertionError(str(_attr(op, "message", "device assert")))
    return []


@register("tt.print")
def _print(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    prefix = str(_attr(op, "prefix", ""))
    for value in args:
        interp.output.append(f"{interp.program_id} {prefix}{np.asarray(value)}")
    return []


# --------------------------------------------------------------------------- tt: descriptors


@register("tt.make_tensor_descriptor")
def _make_desc(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    ty = _rty(op)
    block = tuple(ty.shape or ())
    ndim = len(block)
    shape = tuple(_scalar(a) for a in args[1 : 1 + ndim])
    strides = tuple(_scalar(a) for a in args[1 + ndim : 1 + 2 * ndim])
    elem = ty.elem if ty.elem is not None else ty
    return [
        Descriptor(_scalar(args[0]), shape, strides, block, elem, _pad_of(_attr(op, "padding")))
    ]


def _pad_of(attr: object) -> str:
    """`nan` or `zero` from the `padding` attribute of a descriptor, however it is spelled.

    The generic printer writes `#tt.padding_option<nan>` on TTGIR and the enum's value on
    TTIR (`padding = 2 : i32`, where PAD_ZERO is 1 and PAD_NAN is 2); an older reading took
    both for zero (test_tensor_descriptor_padding: our zeros where the device had NaN)."""
    text = str(attr).strip().lower()
    return "nan" if "nan" in text or text.split(":")[0].strip() == "2" else "zero"


_SCRATCH_BASE = 0x7000_0000_0000


@register("ttg.global_scratch_alloc")
def _global_scratch_alloc(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """A fresh global buffer of `nbytes`, zero-filled, at an address no argument uses."""
    nbytes = int(_attr(op, "nbytes", 0))
    align = max(int(_attr(op, "alignment", 128)), 1)
    nxt = getattr(interp, "scratch_next", _SCRATCH_BASE)
    base = (nxt + align - 1) // align * align
    interp.memory.register(base, np.zeros(max(nbytes, 1), dtype=np.uint8))
    interp.scratch_next = base + max(nbytes, 1)
    return [np.array(base, dtype=np.int64)]


@register("ttng.tensormap_create")
def _tensormap_create(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """Remember what the TMA descriptor written at `desc_ptr` describes.

    The operands come innermost dimension first, as the hardware wants them; the byte strides
    cover every dimension but the innermost one. The table on the interpreter is what
    `reinterpret_tensor_descriptor` reads back, since the 128 bytes themselves are opaque.
    """
    desc_ptr, global_addr, box, dims, strides, _elem_stride = _segments(op, args, 6)
    table = getattr(interp, "tensormaps", None)
    if table is None:
        table = interp.tensormaps = {}
    table[_scalar(desc_ptr[0])] = {
        "base": _scalar(global_addr[0]),
        "box": [_scalar(b) for b in box],
        "dims": [_scalar(d) for d in dims],
        "byte_strides": [_scalar(s) for s in strides],
        # fill_mode 1 is CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA: the NaN padding
        "pad": "nan" if _int_attr(op, "fill_mode", 0) == 1 else "zero",
        # CUtensorMapDataType: UINT8 0, UINT16 1, UINT32 2, INT32 3, UINT64 4, INT64 5, ...
        "unsigned": _int_attr(op, "elem_type", -1) in (0, 1, 2, 4),
    }
    return []


nop("ttng.tensormap_fenceproxy_acquire")


@register("tt.reinterpret_tensor_descriptor", "ttng.reinterpret_tensor_descriptor")
def _reinterpret_desc(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    if isinstance(args[0], Descriptor):
        return [args[0]]
    entry = (getattr(interp, "tensormaps", None) or {}).get(_scalar(args[0]))
    if entry is None:
        raise Unsupported(op.name, "a device descriptor's shape is not recoverable at level 1")
    ty = _rty(op)
    elem = ty.elem if ty.elem is not None else ty
    elem_bytes = np.dtype(to_numpy(elem)).itemsize
    shape = tuple(reversed(entry["dims"]))
    outer = [s // elem_bytes for s in reversed(entry["byte_strides"])]
    strides = tuple(outer + [1])
    block = tuple(ty.shape or reversed(entry["box"]))
    return [
        Descriptor(
            entry["base"],
            shape,
            strides,
            block,
            elem,
            entry.get("pad", "zero"),
            bool(entry.get("unsigned", False)),
        )
    ]


def _desc_pointers(desc: Descriptor, offsets: Sequence[Value]) -> tuple[np.ndarray, np.ndarray]:
    """Addresses of a descriptor block, and the mask of the lanes inside the tensor."""
    ptrs, mask, _pad = _desc_addressing(desc, offsets)
    return ptrs, mask


def _desc_addressing(
    desc: Descriptor, offsets: Sequence[Value]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Addresses of a descriptor block, the mask of the lanes inside the tensor, and the mask
    of the lanes just past the inner extent that share a 16-byte granule with a lane inside
    it (in bounds on every other dimension). The semantics never write the latter; the TMA
    unit does, with the block's values (triton#11583), and a comparison may excuse exactly
    those bytes."""
    step = max(1, to_numpy(desc.elem).itemsize)
    ptrs = np.full(desc.block_shape, desc.base, dtype=np.int64)
    mask = np.ones(desc.block_shape, dtype=bool)
    outer_ok = np.ones(desc.block_shape, dtype=bool)
    granule = np.zeros(desc.block_shape, dtype=bool)
    last = len(desc.block_shape) - 1
    for dim, extent in enumerate(desc.block_shape):
        bshape = [1] * len(desc.block_shape)
        bshape[dim] = extent
        off = (_scalar(offsets[dim]) + np.arange(extent, dtype=np.int64)).reshape(bshape)
        ptrs = ptrs + step * off * np.int64(desc.strides[dim])
        inside = (off >= 0) & (off < desc.shape[dim])
        mask = mask & inside
        if dim == last:
            edge = (desc.shape[dim] - 1) * step // 16
            granule = granule | ((off >= desc.shape[dim]) & (off * step // 16 == edge))
        else:
            outer_ok = outer_ok & inside
    return ptrs, mask, granule & outer_ok


def _note_pad_writes(
    interp: Interp, desc: Descriptor, offsets: Sequence[Value], values: np.ndarray
) -> None:
    """Remember what a descriptor store would have put in the padded granule."""
    ptrs, _mask, pad = _desc_addressing(desc, offsets)
    if not pad.any():
        return
    flat_vals = np.ascontiguousarray(np.asarray(values)).reshape(-1)
    for addr, value in zip(ptrs[pad].tolist(), flat_vals[pad.reshape(-1)], strict=True):
        interp.memory.pad_writes[int(addr)] = value.tobytes()


def _desc(op: Op, args: list[Value]) -> Descriptor:
    desc = args[0]
    if not isinstance(desc, Descriptor):
        raise Unsupported(op.name, "operand 0 is not a descriptor")
    return desc


def _desc_fill(desc: Descriptor, dtype: np.dtype) -> np.ndarray:
    """What a descriptor load writes into the lanes that fall outside the tensor."""
    return np.array(np.nan if desc.pad == "nan" else 0).astype(dtype)


def _desc_block(
    interp: Interp, desc: Descriptor, ptrs: np.ndarray, mask: np.ndarray, dtype: np.dtype
) -> np.ndarray:
    """The block a descriptor load brings in: the lanes inside the tensor from memory, the
    rest the fill, and every f32 rounded to tf32 when the host descriptor asked for it
    (`round_f32_to_tf32`: `CU_TENSOR_MAP_DATA_TYPE_TFLOAT32` rounds on the way in; the
    emulated lowering of `main` spells the same rounding out on the loaded tensor)."""
    block = interp.memory.load(ptrs, mask, _desc_fill(desc, dtype), dtype)
    if desc.tf32 and np.dtype(dtype) == np.float32:
        block = _round_f32_to_tf32(block)
    return block


def _round_f32_to_tf32(x: np.ndarray) -> np.ndarray:
    """Round to nearest even at bit 13 of the f32 pattern (the tf32 LSB); Inf and NaN keep
    their bits, and a mantissa that carries out lands on the next exponent, as in
    `RewriteTensorDescriptorToPointer.cpp`."""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    special = (bits & np.uint32(0x7F80_0000)) == np.uint32(0x7F80_0000)
    bias = ((bits >> np.uint32(13)) & np.uint32(1)) + np.uint32(0xFFF)
    rounded = (bits + bias) & np.uint32(0xFFFF_E000)
    return np.where(special, bits, rounded).astype(np.uint32).view(np.float32)


def _gather_addressing(
    desc: Descriptor, x_offsets: Value, y_offset: Value
) -> tuple[np.ndarray, np.ndarray]:
    """Addresses of the rows a TMA gather reads: row `i` is the descriptor's box (one row by
    `block_shape[-1]` columns) at `[x_offsets[i], y_offset]`, clipped to the tensor like any
    box, so a row index past the shape reads as the fill and a column past it is masked."""
    if len(desc.shape) != 2:
        raise Unsupported("tt.descriptor_gather", f"rank {len(desc.shape)} descriptor")
    step = max(1, to_numpy(desc.elem).itemsize)
    rows = np.asarray(x_offsets).astype(np.int64).reshape(-1, 1)
    cols = (_scalar(y_offset) + np.arange(desc.block_shape[-1], dtype=np.int64)).reshape(1, -1)
    ptrs = desc.base + step * (rows * np.int64(desc.strides[0]) + cols * np.int64(desc.strides[1]))
    mask = (rows >= 0) & (rows < desc.shape[0]) & (cols >= 0) & (cols < desc.shape[1])
    return ptrs, np.broadcast_to(mask, ptrs.shape)


@register("tt.descriptor_gather")
def _desc_gather(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    desc = _desc(op, args)
    ptrs, mask = _gather_addressing(desc, args[1], args[2])
    dtype = to_numpy(_rty(op))
    return [_desc_block(interp, desc, ptrs, mask, dtype)]


@register("tt.descriptor_scatter")
def _desc_scatter(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    desc = _desc(op, args)
    ptrs, mask = _gather_addressing(desc, args[1], args[2])
    interp.memory.store(ptrs, np.asarray(args[3]).reshape(ptrs.shape), mask)
    return []


@register("tt.descriptor_load")
def _desc_load(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    desc = _desc(op, args)
    ptrs, mask = _desc_pointers(desc, args[1:])
    dtype = to_numpy(_rty(op))
    return [_desc_block(interp, desc, ptrs, mask, dtype)]


@register("tt.descriptor_store")
def _desc_store(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    desc = _desc(op, args)
    ptrs, mask = _desc_pointers(desc, args[2:])
    interp.memory.store(ptrs, np.asarray(args[1]), mask)
    _note_pad_writes(interp, desc, args[2:], np.asarray(args[1]))
    return []


_REDUCE_KINDS = ("add", "min", "max", "and", "or", "xor")
# TT_DescriptorReduceKindAttr: ADD=1 MIN=2 MAX=3 INC=4 DEC=5 AND=6 OR=7 XOR=8
_REDUCE_KIND_BY_VALUE = {
    "1": "add",
    "2": "min",
    "3": "max",
    "4": "inc",
    "5": "dec",
    "6": "and",
    "7": "or",
    "8": "xor",
}


def _reduce_kind(op: Op) -> str:
    """The kind of a descriptor reduce, from `#tt.descriptor_reduce_kind<...>` or a bare name.
    `inc` and `dec` (the atomicInc/atomicDec wrap-around) are left unsupported."""
    text = str(_attr(op, "kind", "")).lower()
    found = re.search(r"<(\w+)>", text)
    kind = found.group(1) if found else text
    kind = _REDUCE_KIND_BY_VALUE.get(kind, kind)  # TTGIR spells the enum by its value
    if kind not in _REDUCE_KINDS:
        raise Unsupported(op.name, f"reduce kind {kind!r}")
    return kind


def _desc_reduce(
    interp: Interp, desc: Descriptor, src: np.ndarray, coords: Sequence[Value], kind: str
) -> None:
    """A descriptor store that combines each element with what global memory holds.

    Element by element and relaxed, as `cp.reduce.async.bulk.tensor` is documented; the block
    clips to the tensor like a store does. Integers are signless in the IR; `min` and `max`
    compare as unsigned when the descriptor says so (a host descriptor over a `uint32`
    tensor, a tensormap with an unsigned data type: `test_tensor_descriptor_reduce`).
    """
    ptrs, mask = _desc_pointers(desc, coords)
    dtype = to_numpy(desc.elem)
    cur = interp.memory.load(ptrs, mask, np.zeros((), dtype), dtype)
    src = np.asarray(src).astype(dtype, copy=False)
    if desc.elem.scalar.kind == "float":
        if kind not in ("add", "min", "max"):
            raise Unsupported("tt.descriptor_reduce", f"{kind} on a float tensor")
        a, b = to_float(cur, desc.elem), to_float(src, desc.elem)
        if kind == "add":
            combined = a + b
        else:
            combined = np.minimum(a, b) if kind == "min" else np.maximum(a, b)
        new = from_float(combined, desc.elem)
    else:
        fns = {
            "add": np.add,
            "min": np.minimum,
            "max": np.maximum,
            "and": np.bitwise_and,
            "or": np.bitwise_or,
            "xor": np.bitwise_xor,
        }
        if desc.unsigned and kind in ("min", "max"):
            udt = uint_dtype(desc.elem)
            new = fns[kind](cur.view(udt), src.view(udt)).view(dtype)
        else:
            with np.errstate(over="ignore"):
                new = fns[kind](cur, src).astype(dtype)
    interp.memory.store(ptrs, new, mask)


@register("tt.descriptor_reduce")
def _desc_reduce_op(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    _desc_reduce(interp, _desc(op, args), args[1], args[2:], _reduce_kind(op))
    return []


# --------------------------------------------------------------------------- tt: reduce, scan


def _slice_at(x: np.ndarray, axis: int, i: int) -> np.ndarray:
    return np.take(x, i, axis=axis)


@register("tt.reduce")
def _reduce(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    # The IR does not order a reduction, and its combiner is associative by contract. The fold
    # is a balanced tree over neighbours, left operand on the left: what devices and torch do,
    # which in a narrow float is a different number from a left fold (512 bf16 squares: 528
    # pairwise, 496 left to right, 529.3 exactly). A level of the tree is one evaluation of the
    # region over all its pairs, so a reduction costs log2(n) evaluations, not n.
    axis = _int_attr(op, "axis")
    level = [np.moveaxis(np.asarray(a), axis, 0) for a in args]
    while level[0].shape[0] > 1:
        n = level[0].shape[0]
        pairs = n - (n % 2)
        left = [x[0:pairs:2] for x in level]
        right = [x[1:pairs:2] for x in level]
        merged = [np.asarray(v) for v in interp.run_region(op.regions[0], [*left, *right])]
        if n % 2:
            merged = [
                np.concatenate([m, x[n - 1 : n].astype(m.dtype, copy=False)], axis=0)
                for m, x in zip(merged, level, strict=True)
            ]
        level = merged
    acc = [x[0] for x in level]
    # The combine region already produced values in the storage encoding of the result type,
    # so this only fixes the dtype: `cast_to` would *re-encode* them, and for bf16 and the fp8
    # kinds that reads the bit pattern as a number (a bf16 256.0 is the uint16 0x4380, which
    # re-encoded becomes 0x4687, i.e. 17280 -- `test_sum_dtype`'s bf16 sum).
    return [
        np.asarray(v).astype(to_numpy(t), copy=False)
        for v, t in zip(acc, op.result_types, strict=True)
    ]


@register("tt.scan")
def _scan(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    axis = _int_attr(op, "axis")
    reverse = bool(_attr(op, "reverse", False))
    srcs = [np.flip(np.asarray(a), axis) if reverse else np.asarray(a) for a in args]
    outs = [
        np.empty(s.shape, dtype=to_numpy(t)) for s, t in zip(srcs, op.result_types, strict=True)
    ]
    views = [np.moveaxis(o, axis, 0) for o in outs]
    acc = [_slice_at(s, axis, 0) for s in srcs]
    for i in range(srcs[0].shape[axis]):
        if i:
            cur = [_slice_at(s, axis, i) for s in srcs]
            acc = [np.asarray(v) for v in interp.run_region(op.regions[0], [*acc, *cur])]
        for view, value in zip(views, acc, strict=True):
            view[i] = np.asarray(value).reshape(view[i].shape)
    return [np.flip(o, axis) if reverse else o for o in outs]


# --------------------------------------------------------------------------- control flow


@register(
    "scf.yield",
    "tt.reduce.return",
    "tt.scan.return",
    "tt.map_elementwise.return",
    "ttg.warp_yield",
    "nvws.warp_group.yield",
)
def _yield(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    raise Yield(list(args))


@register("tt.call")
def _call(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """Run the callee's body on the operands; `tt.return` yields its results."""
    callee = str(_attr(op, "callee", "")).lstrip("@")
    func = interp.module.funcs.get(callee)
    if func is None:
        raise Unsupported(op.name, f"no function {callee!r} in the module")
    return interp.run_region(func.regions[0], list(args))


@register("ub.poison")
def _poison(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """An undefined value. Zero is as good as any other choice; a result that depends on it
    is undefined on the device too, and the comparison says so by differing."""
    return [np.zeros(tuple(t.shape or ()), dtype=to_numpy(t)) for t in op.result_types]


@register("tt.return", "ttg.warp_return", "nvws.warp_group.return")
def _return(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    raise Yield(list(args))


@register("scf.condition")
def _condition(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    raise Condition(bool(np.asarray(args[0])), list(args[1:]))


@register("cf.br")
def _br(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    raise Branch(op.successors[0], list(args))


@register("cf.cond_br")
def _cond_br(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    rest = list(args[1:])
    sizes = op.attrs.get("operandSegmentSizes")
    n_true = int(sizes[1]) if isinstance(sizes, (list, tuple)) else len(rest)
    if bool(np.asarray(args[0])):
        raise Branch(op.successors[0], rest[:n_true])
    raise Branch(op.successors[1], rest[n_true:])


@register("scf.for")
def _for(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    lo, hi, step = (_scalar(a) for a in args[:3])
    if step == 0:
        raise Poison("scf.for with a zero step")
    carried = list(args[3:])
    iv_type = op.regions[0].blocks[0].args[0][1] if op.regions[0].blocks[0].args else None
    for i in range(lo, hi, step):
        iv = np.asarray(i).astype(to_numpy(iv_type) if iv_type else np.int64)
        carried = interp.run_region(op.regions[0], [iv, *carried])
    return carried


@register("scf.if")
def _if(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    taken = 0 if bool(np.asarray(args[0])) else 1
    if taken >= len(op.regions) or not op.regions[taken].blocks:
        return []
    return interp.run_region(op.regions[taken], [])


@register("scf.while")
def _while(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    carried = list(args)
    while True:
        cond, carried = interp.run_condition(op.regions[0], carried)
        if not cond:
            return carried
        carried = interp.run_region(op.regions[1], carried)


@register("scf.index_switch")
def _index_switch(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    cases = _int_list(op, "cases")
    key = _scalar(args[0])
    region = op.regions[0]
    if key in cases:
        region = op.regions[1 + cases.index(key)]
    return interp.run_region(region, [])


# --------------------------------------------------------------------------- ttg: shared memory


def _md(op: Op, args: list[Value], i: int = 0) -> MemDesc:
    md = args[i]
    if not isinstance(md, MemDesc):
        raise Unsupported(op.name, f"operand {i} is not a memdesc")
    return md


@register("ttg.local_alloc")
def _local_alloc(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    ty = _rty(op)
    elem = ty.elem if ty.elem is not None else ty
    data = np.zeros(ty.shape or (), dtype=to_numpy(ty))
    if args:
        data[...] = np.asarray(args[0])
    return [MemDesc(data, elem)]


@register("ttg.local_load")
def _local_load(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    return [np.array(_md(op, args).data, dtype=to_numpy(_rty(op)))]


@register("ttg.local_store")
def _local_store(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    _md(op, args, 1).data[...] = np.asarray(args[0])
    return []


@register("ttg.memdesc_index")
def _memdesc_index(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    md = _md(op, args)
    return [md.with_data(md.data[_scalar(args[1])])]


def _slice_view(data: np.ndarray, offsets: list[int], want: tuple[int, ...]) -> np.ndarray:
    """A *view* of `data` at `offsets` with shape `want`; leading offsets beyond the rank of
    `want` pick one index. A view and not a copy, so a store through it reaches the allocation
    and every other view of it, as on the device."""
    pad = len(offsets) - len(want)
    index = tuple(
        int(start) if axis < pad else slice(int(start), int(start) + want[axis - pad])
        for axis, start in enumerate(offsets)
    )
    return data[index]


@register("ttg.memdesc_subview")
def _memdesc_subview(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    md = _md(op, args)
    want = tuple(_rty(op).shape or ())
    return [md.with_data(_slice_view(md.data, [_scalar(a) for a in args[1:]], want))]


@register("ttg.memdesc_subslice")
def _memdesc_subslice(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    md = _md(op, args)
    want = tuple(_rty(op).shape or ())
    return [md.with_data(_slice_view(md.data, _int_list(op, "offsets"), want))]


@register("ttg.memdesc_reinterpret")
def _memdesc_reinterpret(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """The same bytes under another element type, shape or layout."""
    md = _md(op, args)
    ty = _rty(op)
    elem = ty.elem if ty.elem is not None else ty
    if not md.data.flags["C_CONTIGUOUS"]:
        raise Unsupported(op.name, "reinterpreting a strided view of shared memory")
    flat = md.data.reshape(-1).view(to_numpy(elem))
    want = int(np.prod(ty.shape or (1,)))
    if flat.size < want:
        raise Unsupported(op.name, "the new view is larger than the bytes it reinterprets")
    # a smaller view is a prefix of the allocation's bytes, as on the device
    return [MemDesc(flat[:want].reshape(ty.shape or ()), elem, dict(md.attrs))]


def _cluster_wide(op: Op, i: int = 0) -> None:
    """In a cluster a gather or scatter addresses shared memory through DSMEM: the buffer may
    be split over the CTAs (a `CGALayout` with a non-zero basis) or replicated with the
    indices carrying the CTA bits (gluon's `test_shared_gather_cga` broadcast cases, all-zero
    bases). Either way one CTA's semantics cannot say what lands; a shared encoding prints a
    `CGALayout` only when the module has more than one CTA, so that is the test."""
    enc = op.operand_types[i].encoding or "" if i < len(op.operand_types) else ""
    if "CGALayout" in enc:
        raise Unsupported(
            op.name, "a gather or scatter in a cluster may address another CTA (DSMEM)"
        )


@register("ttg.local_gather")
def _local_gather(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    _cluster_wide(op)
    md = _md(op, args)
    idx = np.asarray(args[1]).astype(np.int64)
    return [np.take_along_axis(md.data, idx, axis=_int_attr(op, "axis"))]


@register("ttg.local_scatter")
def _local_scatter(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """`dst[.., indices[I], ..] = values[I]` along `axis`. The pretty form reads
    `local_scatter %dst[%indices], %values`, but the operands are declared `dst, values,
    indices` and that is their order in the generic form (an older reading swapped the two
    tensors: `test_scatter_padded` indexed the buffer with its values)."""
    _cluster_wide(op)
    md = _md(op, args)
    values = np.asarray(args[1])
    idx = np.asarray(args[2]).astype(np.int64)
    np.put_along_axis(md.data, idx, values.astype(md.data.dtype), axis=_int_attr(op, "axis"))
    return []


def _rmw_apply(kind: str, old: np.ndarray, val: np.ndarray) -> np.ndarray:
    """One read-modify-write step of `tt.atomic_rmw`'s kinds on plain numpy values."""
    if kind in ("add", "fadd"):
        return old + val
    if kind == "and":
        return old & val
    if kind == "or":
        return old | val
    if kind == "xor":
        return old ^ val
    if kind in ("max", "umax"):
        return np.maximum(old, val)
    if kind in ("min", "umin"):
        return np.minimum(old, val)
    if kind == "xchg":
        return val
    raise Unsupported("ttg.local_atomic_scatter_rmw", f"rmw op {kind!r}")


@register("ttg.local_atomic_scatter_rmw")
def _local_atomic_scatter_rmw(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """`dst[.., indices[I], ..] = rmw(dst[..], values[I])` along `axis`, one position after
    the other in index order (the device serializes colliding positions in an order of its
    own); the result is the old value at each updated position, zero where masked off."""
    raw = _attr(op, "atomic_rmw_op", _attr(op, "rmw_op"))
    kind = _RMW_KINDS.get(int(raw), "") if isinstance(raw, (int, float)) else str(raw).lower()
    md = _md(op, args)
    values = np.asarray(args[1])
    idx = np.asarray(args[2]).astype(np.int64)
    mask = np.asarray(args[3]).astype(bool) if len(args) > 3 else np.ones(values.shape, bool)
    axis = _int_attr(op, "axis")
    old = np.zeros(values.shape, dtype=md.data.dtype)
    for pos in np.ndindex(*values.shape):
        if not mask[pos]:
            continue
        coord = list(pos)
        coord[axis] = int(idx[pos])
        where = tuple(coord)
        old[pos] = md.data[where]
        md.data[where] = _rmw_apply(kind, md.data[where], values[pos])
    return [old.astype(to_numpy(_rty(op)))]


@register("ttg.memdesc_reshape")
def _memdesc_reshape(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    md = _md(op, args)
    return [md.with_data(md.data.reshape(_rty(op).shape or ()))]


@register("ttg.memdesc_trans")
def _memdesc_trans(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    md = _md(op, args)
    return [md.with_data(np.transpose(md.data, _int_list(op, "order")))]


@register("ttg.async_copy_global_to_local")
def _async_copy(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    md = _md(op, args, 1)
    ptrs = np.asarray(args[0], dtype=np.int64)
    mask = np.asarray(args[2], bool) if len(args) > 2 else None
    other = np.asarray(args[3]) if len(args) > 3 else None
    md.data[...] = interp.memory.load(ptrs, mask, other, md.data.dtype)
    return [np.zeros((), dtype=np.int8) for _ in op.result_types]


# `llvm.intr.assume` states a fact the optimiser may use; it computes nothing.
nop("llvm.intr.assume")

nop(
    "ttg.local_dealloc",
    "ttg.async_commit_group",
    "ttg.async_wait",
    "ttg.async_bundle",
    "ttg.barrier",
    "gpu.barrier",
)


@register("ttg.warp_specialize")
def _warp_specialize(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """The default region here, every partition in a thread of its own, one running at a time
    (`Interp.run_scheduled`): a partition gives the turn up when a barrier phase or an aref
    slot it waits for is not there yet. The partitions are the regions of the
    `ttg.warp_specialize.partitions` op in the second region, over its operands."""
    tasks: list[tuple[str, Region, list[Value]]] = []
    for region in op.regions[1:]:
        for block in region.blocks:
            for inner in block.ops:
                if inner.name != "ttg.warp_specialize.partitions":
                    raise Unsupported(op.name, f"unexpected {inner.name} next to the partitions")
                captured = [interp.value(n) for n in inner.operands]
                tasks += [
                    (f"partition {i}", part, list(captured)) for i, part in enumerate(inner.regions)
                ]
    return interp.run_scheduled(op.regions[0], [], tasks)


@register("ttg.warp_specialize.partitions")
def _partitions(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    raise Unsupported(op.name, "partitions outside a ttg.warp_specialize")


@register("nvws.warp_group")
def _warp_group(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """`nvws.warp_group`: the first region yields the results, the others are worker
    partitions without arguments; scheduled like `ttg.warp_specialize`."""
    tasks = [(f"warp group {i}", region, []) for i, region in enumerate(op.regions[1:], 1)]
    return interp.run_scheduled(op.regions[0], [], tasks)


# ------------------------------------------------------------- nvws: asynchronous references
#
# `nvws-insert-aref` turns the multi-buffered loads of a warp-specialized loop into arefs while
# the loop is still one body with `ttg.partition` attributes; only `tritongpu-partition-loops`
# splits it. Up to there the producer's put and the consumer's get of one iteration follow each
# other in program order, and a slot cursor per side is the whole synchronization.


def _aref(op: Op, args: list[Value], i: int = 0) -> Aref:
    if i >= len(args) or not isinstance(args[i], Aref):
        raise Unsupported(op.name, f"operand {i} is not an aref")
    return args[i]


def _aref_views(aref: Aref, slot: int) -> list[Value]:
    return [b.with_data(b.data[slot]) for b in aref.buffers]


@register("nvws.aref.create")
def _aref_create(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    buffers = [_md(op, args, i) for i in range(len(args))]
    if not buffers or any(b.data.ndim < 1 for b in buffers):
        raise Unsupported(op.name, "an aref needs buffers with a depth axis")
    depths = {int(b.data.shape[0]) for b in buffers}
    if len(depths) != 1:
        raise Unsupported(op.name, f"buffers of different depths {sorted(depths)}")
    return [Aref(buffers, depths.pop())]


def _aref_enter(interp: Interp, op: Op, args: list[Value], side: str) -> list[Value]:
    """The producer waits for its slot to be released, the consumer for it to be published;
    in one loop body that never blocks, across partitions it hands the turn over."""
    aref_arg, stage, _phase = _segments(op, args, 3)
    aref = _aref(op, aref_arg)
    if stage:
        slot = _scalar(stage[0]) % aref.depth
    else:
        slot = (aref.put_cursor if side == "put" else aref.get_cursor) % aref.depth
    if side == "put":
        interp.block_until(lambda: not aref.full[slot], f"{op.name} slot {slot}")
    else:
        interp.block_until(lambda: aref.full[slot], f"{op.name} slot {slot}")
    return [*_aref_views(aref, slot), ArefToken(aref, slot, side)]


def _aref_exit(interp: Interp, op: Op, args: list[Value], side: str) -> list[Value]:
    aref = _aref(op, args)
    if len(args) < 2 or not isinstance(args[1], ArefToken):
        raise Unsupported(op.name, "the second operand is not the token of an enter")
    token = args[1]
    if side == "put":
        aref.put_cursor += 1
        aref.full[token.slot] = True
    else:
        aref.get_cursor += 1
        aref.full[token.slot] = False
    interp.progress()
    return []


@register("nvws.aref.put.enter")
def _aref_put_enter(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    return _aref_enter(interp, op, args, "put")


@register("nvws.aref.get.enter")
def _aref_get_enter(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    return _aref_enter(interp, op, args, "get")


@register("nvws.aref.put.exit")
def _aref_put_exit(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    return _aref_exit(interp, op, args, "put")


@register("nvws.aref.get.exit")
def _aref_get_exit(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    return _aref_exit(interp, op, args, "get")


@register("nvws.aref.buffer")
def _aref_buffer(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """The views of a slot already entered: the token's slot, or the explicit stage."""
    aref = _aref(op, args)
    if len(args) < 2 or not isinstance(args[1], ArefToken):
        raise Unsupported(op.name, "the second operand is not the token of an enter")
    slot = _scalar(args[2]) % aref.depth if len(args) > 2 else args[1].slot
    return _aref_views(aref, slot)


@register("nvws.descriptor_load")
def _nvws_desc_load(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """`tt.descriptor_load` whose tile lands in the shared-memory operand; synchronous."""
    desc, md = _desc(op, args), _md(op, args, len(args) - 1)
    ptrs, mask = _desc_pointers(desc, args[1:-1])
    dtype = md.data.dtype
    md.data[...] = _desc_block(interp, desc, ptrs, mask, dtype)
    return []


@register("nvws.descriptor_gather")
def _nvws_desc_gather(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """`tt.descriptor_gather` whose rows land in the shared-memory operand; synchronous."""
    desc, md = _desc(op, args), _md(op, args, 3)
    ptrs, mask = _gather_addressing(desc, args[1], args[2])
    dtype = md.data.dtype
    md.data[...] = _desc_block(interp, desc, ptrs, mask, dtype).reshape(md.data.shape)
    return []


# ------------------------------------------------------------- ttng: barriers and TMA copies

# An mbarrier orders the asynchronous copies and the partitions against each other. Level 1
# runs a copy synchronously, so a phase that a copy completes is complete when the wait is
# reached; a phase that another partition completes is what `Interp.run_scheduled` waits for.
# The barrier state lives beside the memdesc, keyed by the address of its shared memory.


def _barrier_key(md: MemDesc) -> int:
    return int(md.data.__array_interface__["data"][0])


def _barrier(interp: Interp, op: Op, args: list[Value], i: int = 0) -> Barrier:
    md = _md(op, args, i)
    key = _barrier_key(md)
    if key not in interp.mbarriers:  # a barrier used before its init: one arrival per phase
        interp.mbarriers[key] = Barrier(count=1, pending=1)
    return interp.mbarriers[key]


def _pred_off(args: list[Value], i: int) -> bool:
    return i < len(args) and not bool(np.asarray(args[i]))


@register("ttng.init_barrier")
def _init_barrier(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    count = int(_attr(op, "count", 1))
    interp.mbarriers[_barrier_key(_md(op, args))] = Barrier(count=count, pending=count)
    return []


@register("ttng.inval_barrier")
def _inval_barrier(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    interp.mbarriers.pop(_barrier_key(_md(op, args)), None)
    return []


@register("ttng.arrive_barrier")
def _arrive_barrier(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    if _pred_off(args, 1):
        return []
    _barrier(interp, op, args).arrive(int(_attr(op, "count", 1)))
    interp.progress()
    return []


@register("ttng.barrier_expect")
def _barrier_expect(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    if _pred_off(args, 1):
        return []
    _barrier(interp, op, args).expect(int(_attr(op, "size", 0)))
    interp.progress()
    return []


@register("ttng.wait_barrier")
def _wait_barrier(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    alloc, phase, pred, _deps = _segments(op, args, 4)
    if pred and not bool(np.asarray(pred[0])):
        return []
    barrier = _barrier(interp, op, alloc)
    parity = _scalar(phase[0]) & 1
    interp.block_until(lambda: barrier.passes(parity), f"{op.name} parity {parity}")
    return []


nop("ttng.fence_async_shared", "ttng.async_tma_store_wait")


nop("ttng.cluster_barrier")


# ------------------------------------------------------- ttng: tensor memory and tcgen05


def _tokens(op: Op, skip: int = 0) -> list[Value]:
    """Placeholders for the `!ttg.async.token` results after the first `skip` real ones."""
    return [np.zeros((), dtype=np.int8) for _ in op.result_types[skip:]]


@register("ttng.tmem_alloc")
def _tmem_alloc(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """Tensor memory as a plain buffer, like `ttg.local_alloc`: the blockM/blockN/colStride of
    the encoding are how the hardware lays the columns out, not what the elements mean."""
    ty = _rty(op)
    elem = ty.elem if ty.elem is not None else ty
    data = np.zeros(ty.shape or (), dtype=to_numpy(ty))
    if args:
        data[...] = np.asarray(args[0])
    return [MemDesc(data, elem), *_tokens(op, 1)]


@register("ttng.tmem_store")
def _tmem_store(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """`(dst, [dep], src, pred)`: the operands follow the declaration, destination first."""
    md = _md(op, args, 0)
    src, pred = args[-2], args[-1]
    if bool(np.asarray(pred)):
        md.data[...] = np.asarray(src)
    return _tokens(op)


@register("ttng.tmem_load")
def _tmem_load(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    sizes = op.attrs.get("resultSegmentSizes")
    if isinstance(sizes, (list, tuple)) and len(sizes) == 3 and int(sizes[2]):
        raise Unsupported(op.name, "a reduction result (redOp) is not modelled")
    md = _md(op, args, 0)
    return [np.array(md.data, dtype=to_numpy(_rty(op))), *_tokens(op, 1)]


@register("ttng.tmem_subslice")
def _tmem_subslice(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """A view of `size` columns (or rows, `dim`) of a tensor-memory buffer, aliasing it."""
    md = _md(op, args, 0)
    dim = _int_attr(op, "dim", 1)
    offset = _int_attr(op, "offset", 0)
    shape = tuple(_rty(op).shape or ())
    if len(shape) != md.data.ndim or dim >= md.data.ndim:
        raise Unsupported(op.name, f"rank {len(shape)} slice of a rank {md.data.ndim} buffer")
    index = [slice(None)] * md.data.ndim
    index[dim] = slice(offset, offset + shape[dim])
    return [MemDesc(md.data[tuple(index)], md.elem, dict(md.attrs))]


@register("ttng.tmem_copy")
def _tmem_copy(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """Shared to tensor memory, synchronous at level 1. With the default layout the logical
    elements do not change; the blocked-scales layout, where 32x128b chunks are duplicated
    over four warps, does not preserve the element count and is left unsupported."""
    src, dst = _md(op, args, 0), _md(op, args, 1)
    if src.data.size != dst.data.size:
        raise Unsupported(op.name, "the blocked-scales layout (tensor_memory_scales_encoding)")
    dst.data[...] = np.asarray(src.data).reshape(dst.data.shape).astype(dst.data.dtype)
    return _tokens(op)


nop("ttng.tmem_wait")


def _mma_operand(md: MemDesc, acc_dtype: type, unsigned: bool) -> np.ndarray:
    if is_float(md.elem):
        x = to_float(np.asarray(md.data), md.elem).astype(acc_dtype)
        return _to_tf32(x) if md.elem.scalar.name == "f32" else x
    bits = np.asarray(md.data)
    if unsigned:  # the same bytes read as unsigned integers
        bits = bits.view(np.dtype(f"u{bits.dtype.itemsize}"))
    return bits.astype(np.int64)


@register("ttng.tc_gen5_mma")
def _tc_gen5_mma(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """`d = (d if useD else 0) + a @ b`, synchronously; every barrier given (whose predicate
    holds) sees one arrival, which is what the hardware's commit does when the MMA lands.
    f32 operands are tf32 on the tensor core; an `ieee` dot reaches here already split into
    the three tf32 products of its emulation."""
    a_, b_, d_, _dep, use_d, pred, barriers, barrier_preds = _segments(op, args, 8)
    if _attr(op, "two_ctas") is not None or _attr(op, "multicast") is not None:
        raise Unsupported(op.name, "a two-CTA or multicast MMA spans the cluster")
    if pred and not bool(np.asarray(pred[0])):
        return _tokens(op)
    a, b, d = _md(op, a_), _md(op, b_), _md(op, d_)
    unsigned = _attr(op, "is_unsigned") is not None
    if is_float(d.elem):
        acc_dtype = np.float64 if d.elem.scalar.name == "f64" else np.float32
        prod = np.matmul(_mma_operand(a, acc_dtype, False), _mma_operand(b, acc_dtype, False))
        acc = to_float(np.asarray(d.data), d.elem).astype(acc_dtype)
        out = prod + (acc if bool(np.asarray(use_d[0])) else 0)
        d.data[...] = from_float(out, d.elem)
    else:
        prod = np.matmul(_mma_operand(a, np.int64, unsigned), _mma_operand(b, np.int64, unsigned))
        acc = np.asarray(d.data).astype(np.int64)
        with np.errstate(over="ignore"):
            d.data[...] = (prod + (acc if bool(np.asarray(use_d[0])) else 0)).astype(d.data.dtype)
    for i, bar in enumerate(barriers):
        if i < len(barrier_preds) and not bool(np.asarray(barrier_preds[i])):
            continue
        _barrier(interp, op, [bar]).arrive(1)
    if barriers:
        interp.progress()
    return _tokens(op)


@register("ttng.tc_gen5_mma_scaled")
def _tc_gen5_mma_scaled(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """`d = (useD ? d : 0) + matmul(scale(a, a_scale), scale(b, b_scale))`, done at once.

    The operands live in shared memory and the scales in tensor memory as the logical
    `[M, K / group]` and `[N, K / group]` arrays `tt.dot_scaled` also takes (the blocked-scales
    TMEM layout is how the hardware stores them, `tmem_alloc` from a register tensor keeps the
    logical array). Decoding, scaling and the NaN groups follow `tt.dot_scaled`; the products
    accumulate in f32. `two_ctas` and `multicast` span the cluster and are unsupported."""
    a_, b_, d_, _dep, as_, bs_, use_d, pred, barriers, barrier_preds = _segments(op, args, 10)
    if _attr(op, "two_ctas") is not None or _attr(op, "multicast") is not None:
        raise Unsupported(op.name, "a two-CTA or multicast MMA spans the cluster")
    if pred and not bool(np.asarray(pred[0])):
        return _tokens(op)
    a, b, d = _md(op, a_), _md(op, b_), _md(op, d_)
    a_scale, b_scale = _md(op, as_), _md(op, bs_)
    types = op.operand_types
    ta, tb = types[0], types[1]
    tas, tbs = types[3 + len(_dep)], types[4 + len(_dep)]
    fmt_a, fmt_b = _scaled_format(op, "a_type"), _scaled_format(op, "b_type")
    compute = _SCALED_TYPES["f16" if "f16" in (fmt_a, fmt_b) else "bf16"]
    lhs = _scaled_operand(op, (a.data, ta), (a_scale.data, tas), fmt_a, 0, compute)
    rhs = _scaled_operand(op, (b.data, tb), (b_scale.data, tbs), fmt_b, 1, compute)
    acc_dtype = np.float64 if d.elem.scalar.name == "f64" else np.float32
    with np.errstate(invalid="ignore"):
        prod = np.matmul(lhs.astype(acc_dtype), rhs.astype(acc_dtype))
    acc = to_float(np.asarray(d.data), d.elem).astype(acc_dtype)
    d.data[...] = from_float(prod + (acc if bool(np.asarray(use_d[0])) else 0), d.elem)
    for i, bar in enumerate(barriers):
        if i < len(barrier_preds) and not bool(np.asarray(barrier_preds[i])):
            continue
        _barrier(interp, op, [bar]).arrive(1)
    if barriers:
        interp.progress()
    return _tokens(op)


@register("ttng.tc_gen5_commit")
def _tc_gen5_commit(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """Every prior MMA has already landed at level 1: the commit is one arrival."""
    barrier, pred, _descs = _segments(op, args, 3)
    if pred and not bool(np.asarray(pred[0])):
        return []
    _barrier(interp, op, barrier).arrive(1)
    interp.progress()
    return []


@register("ttng.warp_group_dot")
def _warp_group_dot(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """A `tt.dot` whose operands may live in shared memory; the asynchrony is level 3's."""
    plain = [np.asarray(a.data) if isinstance(a, MemDesc) else a for a in args[:3]]
    return _dot(interp, op, plain)


@register("ttng.warp_group_dot_wait")
def _warp_group_dot_wait(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    return list(args[: len(op.result_types)])


@register("ttng.async_tma_copy_global_to_local")
def _tma_to_local(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """A synchronous `tt.descriptor_load` whose result lands in a shared-memory buffer."""
    desc_arg, coords, offsets, barrier, dest, pred = _segments(op, args, 6)
    if offsets:
        raise Unsupported(op.name, "im2col offsets are not modelled at level 1")
    if pred and not bool(np.asarray(pred[0])):
        return []
    desc, md = _desc(op, desc_arg), _md(op, dest)
    ptrs, mask = _desc_pointers(desc, coords)
    dtype = md.data.dtype
    md.data[...] = _desc_block(interp, desc, ptrs, mask, dtype)
    if barrier:  # the copy is done: the bytes the barrier was told to expect have landed
        _barrier(interp, op, barrier).complete_tx(int(md.data.nbytes))
        interp.progress()
    return []


@register("ttng.async_tma_gather")
def _tma_gather_to_local(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """A synchronous `tt.descriptor_gather` whose rows land in a shared-memory buffer; the
    barrier learns that the bytes it was told to expect have arrived."""
    if len(args) > 5 and not bool(np.asarray(args[5])):
        return []
    desc, md = _desc(op, args), _md(op, args, 4)
    ptrs, mask = _gather_addressing(desc, args[1], args[2])
    dtype = md.data.dtype
    md.data[...] = _desc_block(interp, desc, ptrs, mask, dtype).reshape(md.data.shape)
    _barrier(interp, op, args, 3).complete_tx(int(md.data.nbytes))
    interp.progress()
    return []


@register("ttng.async_tma_scatter")
def _tma_scatter_to_global(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """A synchronous `tt.descriptor_scatter` whose rows come from a shared-memory buffer."""
    desc, md = _desc(op, args), _md(op, args, 3)
    ptrs, mask = _gather_addressing(desc, args[1], args[2])
    interp.memory.store(ptrs, np.asarray(md.data).reshape(ptrs.shape), mask)
    return []


@register("ttng.async_tma_copy_local_to_global")
def _tma_to_global(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """A synchronous `tt.descriptor_store` whose source is a shared-memory buffer."""
    desc, md = _desc(op, args), _md(op, args, len(args) - 1)
    ptrs, mask = _desc_pointers(desc, args[1:-1])
    interp.memory.store(ptrs, np.asarray(md.data), mask)
    _note_pad_writes(interp, desc, args[1:-1], np.asarray(md.data))
    return []


@register("ttng.async_tma_reduce")
def _tma_reduce(interp: Interp, op: Op, args: list[Value]) -> list[Value]:
    """A synchronous `tt.descriptor_reduce` whose source is a shared-memory buffer."""
    desc, md = _desc(op, args), _md(op, args, len(args) - 1)
    _desc_reduce(interp, desc, np.asarray(md.data), args[1:-1], _reduce_kind(op))
    return []
