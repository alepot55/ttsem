"""One hand-built op per implemented op name, checked against numpy."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import (
    block,
    buffer,
    evaluate,
    interp,
    memdesc,
    one,
    op,
    ptr,
    region,
    tensor,
    tensordesc,
    ty,
)

from ttsem.ir_types import DenseAttr, FloatBits, Type
from ttsem.memory import Memory, MemoryFault
from ttsem.values import Descriptor, MemDesc, Poison, Unsupported, from_float, to_float

I32 = ty("i32")
I8 = ty("i8")
I1 = ty("i1")
I64 = ty("i64")
F32 = ty("f32")
BF16 = ty("bf16")
T4 = tensor((4,), "i32")
T4F = tensor((4,), "f32")
T4B = tensor((4,), "i1")


def i32(*xs: int) -> np.ndarray:
    return np.array(xs, dtype=np.int32)


def binop(name: str, a: np.ndarray, b: np.ndarray, t=T4, rt=None) -> np.ndarray:
    return one(op(name, [t, t], rt or t), [a, b])


# --------------------------------------------------------------------------- arith int


def test_constant_scalar() -> None:
    assert one(op("arith.constant", [], I32, attrs={"value": 7})) == np.int32(7)


def test_constant_dense_splat() -> None:
    got = one(op("arith.constant", [], T4, attrs={"value": DenseAttr(3, I32, (4,))}))
    assert np.array_equal(got, i32(3, 3, 3, 3)) and got.dtype == np.int32


def test_constant_dense_list() -> None:
    attr = DenseAttr([1, 2, 3, 4], I32, (4,))
    assert np.array_equal(one(op("arith.constant", [], T4, attrs={"value": attr})), i32(1, 2, 3, 4))


def test_constant_float_dense() -> None:
    attr = DenseAttr(1.5, F32, (4,))
    got = one(op("arith.constant", [], T4F, attrs={"value": attr}))
    assert np.array_equal(got, np.full(4, 1.5, np.float32))


@pytest.mark.parametrize(
    ("name", "expect"),
    [
        ("arith.addi", [6, 8, 10, 12]),
        ("arith.subi", [-4, -4, -4, -4]),
        ("arith.muli", [5, 12, 21, 32]),
        ("arith.andi", [1, 2, 3, 0]),
        ("arith.ori", [5, 6, 7, 12]),
        ("arith.xori", [4, 4, 4, 12]),
        ("arith.maxsi", [5, 6, 7, 8]),
        ("arith.minsi", [1, 2, 3, 4]),
    ],
)
def test_int_binops(name: str, expect: list[int]) -> None:
    assert np.array_equal(binop(name, i32(1, 2, 3, 4), i32(5, 6, 7, 8)), i32(*expect))


def test_int_arithmetic_wraps_at_the_width() -> None:
    t = tensor((2,), "i8")
    got = binop("arith.addi", np.array([127, -128], np.int8), np.array([1, -1], np.int8), t)
    assert np.array_equal(got, np.array([-128, 127], np.int8))


def test_maxui_minui_read_the_operands_unsigned() -> None:
    a, b = i32(-1, 3), i32(2, 4)
    assert np.array_equal(binop("arith.maxui", a, b, tensor((2,), "i32")), i32(-1, 4))
    assert np.array_equal(binop("arith.minui", a, b, tensor((2,), "i32")), i32(2, 3))


@pytest.mark.parametrize(
    ("name", "expect"),
    [
        ("arith.divsi", [-2, 2]),
        ("arith.remsi", [-1, 1]),
        ("arith.divui", [1431655763, 2]),
        ("arith.remui", [0, 1]),
    ],
)
def test_divide_and_remainder(name: str, expect: list[int]) -> None:
    a, b = i32(-7, 7), i32(3, 3)
    assert np.array_equal(binop(name, a, b, tensor((2,), "i32")), i32(*expect))


@pytest.mark.parametrize("name", ["arith.divsi", "arith.divui", "arith.remsi", "arith.remui"])
def test_division_by_zero_is_poison(name: str) -> None:
    with pytest.raises(Poison):
        binop(name, i32(1, 2, 3, 4), i32(1, 0, 1, 1))


def test_signed_division_overflow_is_poison() -> None:
    lo = np.array([np.iinfo(np.int32).min, 1], np.int32)
    with pytest.raises(Poison):
        binop("arith.divsi", lo, i32(-1, 1), tensor((2,), "i32"))


@pytest.mark.parametrize(
    ("name", "expect"),
    [
        ("arith.shli", [8, 0, -4]),
        ("arith.shrui", [0, 0, 2147483647]),
        ("arith.shrsi", [0, 0, -1]),
    ],
)
def test_shifts_including_out_of_range_amounts(name: str, expect: list[int]) -> None:
    a, b = i32(1, 1, -2), i32(3, 32, 1)
    assert np.array_equal(binop(name, a, b, tensor((3,), "i32")), i32(*expect))


@pytest.mark.parametrize(
    ("pred", "expect"),
    [
        (0, [False, False, True]),
        (1, [True, True, False]),
        (2, [True, False, False]),
        (3, [True, False, True]),
        (4, [False, True, False]),
        (5, [False, True, True]),
        (6, [False, False, False]),
        (7, [False, False, True]),
        (8, [True, True, False]),
        (9, [True, True, True]),
    ],
)
def test_cmpi_predicates(pred: int, expect: list[bool]) -> None:
    a, b = i32(-1, 3, 2), i32(1, 1, 2)
    got = one(op("arith.cmpi", [T4, T4], tensor((3,), "i1"), attrs={"predicate": pred}), [a, b])
    assert np.array_equal(got, np.array(expect))


def test_cmpi_rejects_an_unknown_predicate() -> None:
    with pytest.raises(Unsupported):
        one(op("arith.cmpi", [T4, T4], T4B, attrs={"predicate": 42}), [i32(1), i32(1)])


def test_select() -> None:
    cond = np.array([True, False, True, False])
    got = one(op("arith.select", [T4B, T4, T4], T4), [cond, i32(1, 2, 3, 4), i32(5, 6, 7, 8)])
    assert np.array_equal(got, i32(1, 6, 3, 8))


# --------------------------------------------------------------------------- arith casts


def test_extsi_of_i1_gives_minus_one() -> None:
    got = one(op("arith.extsi", [T4B], T4), [np.array([True, False, True, False])])
    assert np.array_equal(got, i32(-1, 0, -1, 0))


def test_extui_of_i1_gives_one() -> None:
    got = one(op("arith.extui", [T4B], T4), [np.array([True, False, True, False])])
    assert np.array_equal(got, i32(1, 0, 1, 0))


def test_extsi_widens_with_the_sign() -> None:
    got = one(op("arith.extsi", [tensor((2,), "i8")], tensor((2,), "i32")), [np.int8([-1, 5])])
    assert np.array_equal(got, np.array([-1, 5], np.int32))


def test_extui_widens_with_zeros() -> None:
    got = one(op("arith.extui", [tensor((2,), "i8")], tensor((2,), "i32")), [np.int8([-1, 5])])
    assert np.array_equal(got, np.array([255, 5], np.int32))


def test_trunci_keeps_the_low_bits() -> None:
    got = one(op("arith.trunci", [T4], tensor((4,), "i8")), [i32(258, -1, 3, 0)])
    assert np.array_equal(got, np.array([2, -1, 3, 0], np.int8))


def test_trunci_to_i1_keeps_the_low_bit() -> None:
    got = one(op("arith.trunci", [T4], T4B), [i32(2, 3, 0, 1)])
    assert np.array_equal(got, np.array([False, True, False, True]))


def test_index_cast_round_trip() -> None:
    to_index = one(op("arith.index_cast", [I32], ty("index")), [np.int32(-3)])
    back = one(op("arith.index_cast", [ty("index")], I32), [to_index])
    assert to_index == -3 and back == -3


@pytest.mark.parametrize(
    ("name", "src", "expect"),
    [
        ("arith.sitofp", np.int32([-1, 2]), [-1.0, 2.0]),
        ("arith.uitofp", np.int32([-1, 2]), [4294967296.0, 2.0]),
    ],
)
def test_int_to_float(name: str, src: np.ndarray, expect: list[float]) -> None:
    got = one(op(name, [tensor((2,), "i32")], tensor((2,), "f32")), [src])
    assert np.array_equal(got, np.array(expect, np.float32))


def test_float_to_int_truncates_toward_zero() -> None:
    src = np.array([-1.9, 2.7], np.float32)
    got = one(op("arith.fptosi", [tensor((2,), "f32")], tensor((2,), "i32")), [src])
    assert np.array_equal(got, np.array([-1, 2], np.int32))


def test_fptoui() -> None:
    src = np.array([1.9, 2.7], np.float32)
    got = one(op("arith.fptoui", [tensor((2,), "f32")], tensor((2,), "i32")), [src])
    assert np.array_equal(got, np.array([1, 2], np.int32))


def test_extf_and_truncf_round_trip_through_bf16() -> None:
    src = np.array([1.5, -2.25], np.float32)
    small = one(op("arith.truncf", [tensor((2,), "f32")], tensor((2,), "bf16")), [src])
    back = one(op("arith.extf", [tensor((2,), "bf16")], tensor((2,), "f32")), [small])
    assert small.dtype == np.uint16 and np.array_equal(back, src)


def test_truncf_to_bf16_rounds_to_nearest_even() -> None:
    # 1 + 2**-9 sits exactly between two bf16 values; RTNE keeps the even significand.
    src = np.array([np.float32(1.0 + 2.0**-9)], np.float32)
    small = one(op("arith.truncf", [tensor((1,), "f32")], tensor((1,), "bf16")), [src])
    assert to_float(small, BF16)[0] == np.float32(1.0)


def test_bitcast_reinterprets_the_bits() -> None:
    src = np.array([1.0, -2.0], np.float32)
    got = one(op("arith.bitcast", [tensor((2,), "f32")], tensor((2,), "i32")), [src])
    assert np.array_equal(got, src.view(np.int32))


def test_ptr_to_int_and_back() -> None:
    addr = one(op("tt.ptr_to_int", [ptr("f32")], I64), [np.int64(4096)])
    back = one(op("tt.int_to_ptr", [I64], ptr("f32")), [addr])
    assert addr == 4096 and back == 4096


# --------------------------------------------------------------------------- arith float


@pytest.mark.parametrize(
    ("name", "expect"),
    [
        ("arith.addf", [4.0, 6.0]),
        ("arith.subf", [-2.0, -2.0]),
        ("arith.mulf", [3.0, 8.0]),
        ("arith.divf", [1.0 / 3.0, 0.5]),
        ("tt.precise_divf", [1.0 / 3.0, 0.5]),
        ("arith.maxnumf", [3.0, 4.0]),
        ("arith.minnumf", [1.0, 2.0]),
        ("arith.maximumf", [3.0, 4.0]),
        ("arith.minimumf", [1.0, 2.0]),
    ],
)
def test_float_binops(name: str, expect: list[float]) -> None:
    a, b = np.array([1.0, 2.0], np.float32), np.array([3.0, 4.0], np.float32)
    got = binop(name, a, b, tensor((2,), "f32"))
    assert np.allclose(got, np.array(expect, np.float32))


def test_maxnumf_ignores_nan_and_maximumf_propagates_it() -> None:
    a = np.array([np.nan], np.float32)
    b = np.array([1.0], np.float32)
    t = tensor((1,), "f32")
    assert binop("arith.maxnumf", a, b, t)[0] == 1.0
    assert np.isnan(binop("arith.maximumf", a, b, t)[0])


def test_negf() -> None:
    got = one(op("arith.negf", [T4F], T4F), [np.array([1.0, -2.0, 0.0, 3.0], np.float32)])
    assert np.array_equal(got, np.array([-1.0, 2.0, -0.0, -3.0], np.float32))


@pytest.mark.parametrize(
    ("pred", "expect"),
    [
        (0, [False, False, False]),
        (1, [False, True, False]),
        (2, [True, False, False]),
        (4, [False, False, False]),
        (6, [True, False, False]),
        (7, [True, True, False]),
        (13, [True, False, True]),
        (14, [False, False, True]),
        (15, [True, True, True]),
    ],
)
def test_cmpf_predicates(pred: int, expect: list[bool]) -> None:
    a = np.array([2.0, 1.0, np.nan], np.float32)
    b = np.array([1.0, 1.0, 1.0], np.float32)
    t = tensor((3,), "f32")
    got = one(op("arith.cmpf", [t, t], tensor((3,), "i1"), attrs={"predicate": pred}), [a, b])
    assert np.array_equal(got, np.array(expect))


def test_bf16_addition_rounds_once() -> None:
    t = tensor((1,), "bf16")
    a = from_float(np.array([1.0], np.float32), BF16)
    b = from_float(np.array([2.0**-9], np.float32), BF16)
    assert to_float(binop("arith.addf", a, b, t), BF16)[0] == np.float32(1.0)


def test_fp8_round_trip() -> None:
    src = np.array([0.0, 1.0, -2.5, 448.0], np.float32)
    small = from_float(src, ty("f8E4M3FN"))
    assert small.dtype == np.uint8
    assert np.array_equal(to_float(small, ty("f8E4M3FN")), src)


# --------------------------------------------------------------------------- math


@pytest.mark.parametrize(
    ("name", "fn"),
    [
        ("math.exp", np.exp),
        ("math.exp2", np.exp2),
        ("math.log", np.log),
        ("math.log2", np.log2),
        ("math.sqrt", np.sqrt),
        ("tt.precise_sqrt", np.sqrt),
        ("math.rsqrt", lambda x: 1.0 / np.sqrt(x)),
        ("math.sin", np.sin),
        ("math.cos", np.cos),
        ("math.tanh", np.tanh),
        ("math.floor", np.floor),
        ("math.ceil", np.ceil),
        ("math.absf", np.abs),
    ],
)
def test_math_unary(name: str, fn) -> None:
    src = np.array([0.5, 2.0, 3.25], np.float32)
    got = one(op(name, [tensor((3,), "f32")], tensor((3,), "f32")), [src])
    assert np.allclose(got, fn(src).astype(np.float32), rtol=1e-6, atol=1e-7)


def test_math_erf() -> None:
    src = np.array([0.0, 1.0], np.float32)
    got = one(op("math.erf", [tensor((2,), "f32")], tensor((2,), "f32")), [src])
    assert np.allclose(got, np.array([0.0, 0.8427008], np.float32))


def test_math_absi() -> None:
    got = one(op("math.absi", [T4], T4), [i32(-1, 2, -3, 4)])
    assert np.array_equal(got, i32(1, 2, 3, 4))


def test_math_fma() -> None:
    t = tensor((2,), "f32")
    a, b, c = (np.array(v, np.float32) for v in ([2.0, 3.0], [4.0, 5.0], [1.0, 1.0]))
    assert np.array_equal(one(op("math.fma", [t, t, t], t), [a, b, c]), np.float32([9.0, 16.0]))


# --------------------------------------------------------------------------- tt shapes


def test_make_range() -> None:
    got = one(op("tt.make_range", [], T4, attrs={"start": 0, "end": 4}))
    assert np.array_equal(got, i32(0, 1, 2, 3)) and got.dtype == np.int32


def test_splat() -> None:
    assert np.array_equal(one(op("tt.splat", [I32], T4), [np.int32(9)]), i32(9, 9, 9, 9))


def test_broadcast() -> None:
    src = np.arange(3, dtype=np.int32).reshape(1, 3)
    got = one(op("tt.broadcast", [tensor((1, 3), "i32")], tensor((2, 3), "i32")), [src])
    assert got.shape == (2, 3) and np.array_equal(got[0], got[1])


def test_expand_dims() -> None:
    got = one(
        op("tt.expand_dims", [T4], tensor((4, 1), "i32"), attrs={"axis": 1}), [i32(1, 2, 3, 4)]
    )
    assert got.shape == (4, 1)


def test_reshape() -> None:
    got = one(op("tt.reshape", [T4], tensor((2, 2), "i32")), [i32(1, 2, 3, 4)])
    assert np.array_equal(got, np.array([[1, 2], [3, 4]], np.int32))


def test_trans() -> None:
    src = np.arange(6, dtype=np.int32).reshape(2, 3)
    got = one(
        op("tt.trans", [tensor((2, 3), "i32")], tensor((3, 2), "i32"), attrs={"order": [1, 0]}),
        [src],
    )
    assert np.array_equal(got, src.T)


def test_join_and_split() -> None:
    a, b = i32(1, 2, 3, 4), i32(5, 6, 7, 8)
    joined = one(op("tt.join", [T4, T4], tensor((4, 2), "i32")), [a, b])
    parts = evaluate(op("tt.split", [tensor((4, 2), "i32")], [T4, T4]), [joined])
    assert np.array_equal(joined[:, 0], a) and np.array_equal(parts[1], b)


def test_cat() -> None:
    got = one(op("tt.cat", [T4, T4], tensor((8,), "i32")), [i32(1, 2, 3, 4), i32(5, 6, 7, 8)])
    assert np.array_equal(got, i32(1, 2, 3, 4, 5, 6, 7, 8))


def test_gather() -> None:
    src = np.arange(6, dtype=np.int32).reshape(2, 3)
    idx = np.array([[2, 0, 1], [1, 1, 0]], np.int32)
    t = tensor((2, 3), "i32")
    got = one(op("tt.gather", [t, t], t, attrs={"axis": 1}), [src, idx])
    assert np.array_equal(got, np.take_along_axis(src, idx, axis=1))


def test_histogram() -> None:
    src = np.array([0, 1, 1, 3], np.int32)
    got = one(op("tt.histogram", [tensor((4,), "i32")], T4), [src])
    assert np.array_equal(got, i32(1, 2, 0, 1))


def test_histogram_drops_masked_lanes() -> None:
    src = np.array([0, 1, 1, 3], np.int32)
    mask = np.array([True, True, False, True])
    t = tensor((4,), "i32")
    got = one(op("tt.histogram", [t, tensor((4,), "i1")], T4), [src, mask])
    assert np.array_equal(got, i32(1, 1, 0, 1))


def test_program_id_and_num_programs() -> None:
    pid = one(op("tt.get_program_id", [], I32, attrs={"axis": 1}), program_id=(3, 5, 7))
    total = one(op("tt.get_num_programs", [], I32, attrs={"axis": 2}), num_programs=(2, 4, 8))
    assert pid == 5 and total == 8


# --------------------------------------------------------------------------- tt memory


def test_addptr_scales_by_the_pointee_size() -> None:
    t = tensor((4,), ptr("i32"))
    got = one(op("tt.addptr", [t, T4], t), [np.full(4, 1024, np.int64), i32(0, 1, 2, 3)])
    assert np.array_equal(got, np.array([1024, 1028, 1032, 1036], np.int64))


def test_addptr_of_i8_scales_by_one() -> None:
    t = tensor((4,), ptr("i8"))
    got = one(op("tt.addptr", [t, T4], t), [np.full(4, 64, np.int64), i32(0, 1, 2, 3)])
    assert np.array_equal(got, np.array([64, 65, 66, 67], np.int64))


def _mem(values: np.ndarray, base: int = 4096) -> tuple[Memory, np.ndarray]:
    memory = Memory()
    memory.register(base, values)
    return memory, values


def test_load_and_store() -> None:
    memory, data = _mem(np.arange(8, dtype=np.int32))
    pt = tensor((4,), ptr("i32"))
    addrs = np.array([4096, 4100, 4104, 4108], np.int64)
    got = one(op("tt.load", [pt], T4), [addrs], memory=memory)
    assert np.array_equal(got, i32(0, 1, 2, 3))
    evaluate(op("tt.store", [pt, T4], []), [addrs, i32(9, 9, 9, 9)], memory=memory)
    assert np.array_equal(data[:4], i32(9, 9, 9, 9))


def test_masked_load_yields_other() -> None:
    memory, _ = _mem(np.arange(8, dtype=np.int32))
    pt = tensor((4,), ptr("i32"))
    addrs = np.array([4096, 4100, 4104, 4108], np.int64)
    mask = np.array([True, False, True, False])
    got = one(op("tt.load", [pt, T4B, T4], T4), [addrs, mask, i32(7, 7, 7, 7)], memory=memory)
    assert np.array_equal(got, i32(0, 7, 2, 7))


def test_masked_load_without_other_yields_zero() -> None:
    memory, _ = _mem(np.arange(8, dtype=np.int32))
    pt = tensor((4,), ptr("i32"))
    addrs = np.array([4096, 4100, 4104, 4108], np.int64)
    mask = np.array([False, True, False, True])
    got = one(op("tt.load", [pt, T4B], T4), [addrs, mask], memory=memory)
    assert np.array_equal(got, i32(0, 1, 0, 3))


def test_masked_store_leaves_the_other_lanes_alone() -> None:
    memory, data = _mem(np.zeros(4, dtype=np.int32))
    pt = tensor((4,), ptr("i32"))
    addrs = 4096 + 4 * np.arange(4, dtype=np.int64)
    mask = np.array([True, False, True, False])
    evaluate(op("tt.store", [pt, T4, T4B], []), [addrs, i32(1, 2, 3, 4), mask], memory=memory)
    assert np.array_equal(data, i32(1, 0, 3, 0))


def test_a_masked_off_lane_never_faults() -> None:
    memory, _ = _mem(np.zeros(4, dtype=np.int32))
    pt = tensor((2,), ptr("i32"))
    addrs = np.array([4096, 1], np.int64)
    got = one(
        op("tt.load", [pt, tensor((2,), "i1")], tensor((2,), "i32")),
        [addrs, np.array([True, False])],
        memory=memory,
    )
    assert np.array_equal(got, np.array([0, 0], np.int32))


def test_a_live_lane_out_of_range_faults() -> None:
    memory, _ = _mem(np.zeros(4, dtype=np.int32))
    with pytest.raises(MemoryFault):
        one(
            op("tt.load", [tensor((1,), ptr("i32"))], tensor((1,), "i32")),
            [np.array([1], np.int64)],
            memory=memory,
        )


def test_atomic_rmw_add_returns_the_old_value() -> None:
    memory, data = _mem(np.array([1, 2, 3, 4], np.int32))
    pt = tensor((4,), ptr("i32"))
    addrs = 4096 + 4 * np.arange(4, dtype=np.int64)
    old = one(
        op("tt.atomic_rmw", [pt, T4], T4, attrs={"atomic_rmw_op": 4}),
        [addrs, i32(10, 10, 10, 10)],
        memory=memory,
    )
    assert np.array_equal(old, i32(1, 2, 3, 4)) and np.array_equal(data, i32(11, 12, 13, 14))


def test_atomic_rmw_respects_the_mask() -> None:
    memory, data = _mem(np.array([1, 2], np.int32))
    pt = tensor((2,), ptr("i32"))
    addrs = 4096 + 4 * np.arange(2, dtype=np.int64)
    attrs = {"atomic_rmw_op": 4}
    args = [addrs, np.array([5, 5], np.int32), np.array([True, False])]
    evaluate(
        op(
            "tt.atomic_rmw",
            [pt, tensor((2,), "i32"), tensor((2,), "i1")],
            tensor((2,), "i32"),
            attrs=attrs,
        ),
        args,
        memory=memory,
    )
    assert np.array_equal(data, np.array([6, 2], np.int32))


def test_atomic_rmw_rejects_an_unknown_kind() -> None:
    memory, _ = _mem(np.zeros(1, np.int32))
    pt = tensor((1,), ptr("i32"))
    with pytest.raises(Unsupported):
        one(
            op(
                "tt.atomic_rmw",
                [pt, tensor((1,), "i32")],
                tensor((1,), "i32"),
                attrs={"atomic_rmw_op": 99},
            ),
            [np.array([4096], np.int64), np.array([1], np.int32)],
            memory=memory,
        )


def test_atomic_cas() -> None:
    memory, data = _mem(np.array([1, 2], np.int32))
    pt = tensor((2,), ptr("i32"))
    t = tensor((2,), "i32")
    addrs = 4096 + 4 * np.arange(2, dtype=np.int64)
    old = one(
        op("tt.atomic_cas", [pt, t, t], t),
        [addrs, np.int32([1, 9]), np.int32([7, 7])],
        memory=memory,
    )
    assert np.array_equal(old, np.int32([1, 2])) and np.array_equal(data, np.int32([7, 2]))


# --------------------------------------------------------------------------- tt dot


def _dot(a: np.ndarray, b: np.ndarray, c: np.ndarray, at: str, rt: str, **attrs) -> np.ndarray:
    m, k = a.shape
    n = b.shape[1]
    types = [tensor((m, k), at), tensor((k, n), at), tensor((m, n), rt)]
    return one(op("tt.dot", types, tensor((m, n), rt), attrs=attrs), [a, b, c])


def test_dot_int8_accumulates_in_int32() -> None:
    a = np.array([[1, 2], [3, 4]], np.int8)
    b = np.array([[5, 6], [7, 8]], np.int8)
    c = np.ones((2, 2), np.int32)
    got = one(
        op(
            "tt.dot",
            [tensor((2, 2), "i8"), tensor((2, 2), "i8"), tensor((2, 2), "i32")],
            tensor((2, 2), "i32"),
        ),
        [a, b, c],
    )
    assert np.array_equal(got, a.astype(np.int32) @ b.astype(np.int32) + 1)


def test_dot_f32_ieee_is_exact_for_small_integers() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], np.float32)
    c = np.zeros((2, 2), np.float32)
    assert np.array_equal(_dot(a, b, c, "f32", "f32", inputPrecision=2), a @ b)


def test_dot_f32_tf32_drops_the_low_mantissa_bits() -> None:
    a = np.array([[np.float32(1.0 + 2.0**-20)]], np.float32)
    b = np.array([[np.float32(1.0)]], np.float32)
    c = np.zeros((1, 1), np.float32)
    assert _dot(a, b, c, "f32", "f32", inputPrecision=0)[0, 0] == np.float32(1.0)
    assert _dot(a, b, c, "f32", "f32", inputPrecision=2)[0, 0] == a[0, 0]


def test_dot_bf16_decodes_the_bit_patterns() -> None:
    af = np.array([[1.0, 2.0]], np.float32)
    bf = np.array([[3.0], [4.0]], np.float32)
    a, b = from_float(af, BF16), from_float(bf, BF16)
    got = one(
        op(
            "tt.dot",
            [tensor((1, 2), "bf16"), tensor((2, 1), "bf16"), tensor((1, 1), "f32")],
            tensor((1, 1), "f32"),
            attrs={"inputPrecision": 0},
        ),
        [a, b, np.zeros((1, 1), np.float32)],
    )
    assert got[0, 0] == np.float32(11.0)


def test_rejected_ops() -> None:
    with pytest.raises(Unsupported):
        evaluate(op("tt.elementwise_inline_asm", [], []))


# --------------------------------------------------------------------------- tt misc


def test_clampf() -> None:
    t = tensor((3,), "f32")
    x = np.array([-2.0, 0.5, 4.0], np.float32)
    lo, hi = np.zeros(3, np.float32), np.ones(3, np.float32)
    got = one(op("tt.clampf", [t, t, t], t, attrs={"propagateNan": 0}), [x, lo, hi])
    assert np.array_equal(got, np.array([0.0, 0.5, 1.0], np.float32))


def test_clampf_propagates_nan_when_asked() -> None:
    t = tensor((1,), "f32")
    x = np.array([np.nan], np.float32)
    args = [x, np.zeros(1, np.float32), np.ones(1, np.float32)]
    assert np.isnan(one(op("tt.clampf", [t, t, t], t, attrs={"propagateNan": 65535}), args)[0])
    # without propagation the NaN is dropped by fmax and the lower bound wins
    assert one(op("tt.clampf", [t, t, t], t, attrs={"propagateNan": 0}), args)[0] == 0.0


def test_mulhiui() -> None:
    a, b = np.array([1 << 20], np.int32), np.array([1 << 20], np.int32)
    t = tensor((1,), "i32")
    assert one(op("tt.mulhiui", [t, t], t), [a, b])[0] == 256


def test_fp_to_fp() -> None:
    src = np.array([1.5, -2.0], np.float32)
    got = one(
        op("tt.fp_to_fp", [tensor((2,), "f32")], tensor((2,), "bf16"), attrs={"rounding": 1}), [src]
    )
    assert np.array_equal(to_float(got, BF16), src)


def test_fp_to_fp_rounds_toward_zero_when_asked() -> None:
    # 1.1 is 0x3F8CCCCD as f32: chopped to bf16 it is 0x3F8C (1.09375), rounded 0x3F8D
    got = one(
        op("tt.fp_to_fp", [tensor((2,), "f32")], tensor((2,), "bf16"), attrs={"rounding": 0}),
        [np.array([1.1, -1.1], np.float32)],
    )
    assert got.tolist() == [0x3F8C, 0xBF8C]
    nearest = one(
        op("tt.fp_to_fp", [tensor((2,), "f32")], tensor((2,), "bf16"), attrs={"rounding": 1}),
        [np.array([1.1, -1.1], np.float32)],
    )
    assert nearest.tolist() == [0x3F8D, 0xBF8D]


def test_assert_passes_and_fails() -> None:
    t = tensor((2,), "i1")
    evaluate(op("tt.assert", [t], [], attrs={"message": "boom"}), [np.array([True, True])])
    with pytest.raises(AssertionError):
        evaluate(op("tt.assert", [t], [], attrs={"message": "boom"}), [np.array([True, False])])


def test_print_records_its_output() -> None:
    from conftest import interp as make

    machine = make()
    target = op("tt.print", [T4], [], attrs={"prefix": "x: "})
    machine.scopes = [{"%a0": i32(1, 2, 3, 4)}]
    machine.eval_op(target)
    assert len(machine.output) == 1 and "x: " in machine.output[0]


# --------------------------------------------------------------------------- descriptors


def _descriptor(base: int = 4096) -> Descriptor:
    return Descriptor(base, (3, 3), (4, 1), (2, 2), F32, "zero")


def test_make_tensor_descriptor() -> None:
    result = tensordesc((2, 2), "f32")
    types = [ptr("f32"), I32, I32, I64, I64]
    args = [np.int64(4096), np.int32(3), np.int32(3), np.int64(4), np.int64(1)]
    desc = evaluate(op("tt.make_tensor_descriptor", types, result), args)[0]
    assert isinstance(desc, Descriptor)
    assert desc.base == 4096 and desc.shape == (3, 3) and desc.block_shape == (2, 2)


def test_descriptor_load_clips_to_the_shape() -> None:
    memory = Memory()
    memory.register(4096, np.arange(16, dtype=np.float32))
    desc = _descriptor()
    types = [tensordesc((2, 2), "f32"), I32, I32]
    args = [desc, np.int32(2), np.int32(2)]
    got = one(op("tt.descriptor_load", types, tensor((2, 2), "f32")), args, memory=memory)
    # only element (2, 2) is inside the 3x3 tensor; the rest is the zero fill
    assert np.array_equal(got, np.array([[10.0, 0.0], [0.0, 0.0]], np.float32))


def test_descriptor_store_clips_to_the_shape() -> None:
    memory = Memory()
    data = np.zeros(16, np.float32)
    memory.register(4096, data)
    desc = _descriptor()
    types = [tensordesc((2, 2), "f32"), tensor((2, 2), "f32"), I32, I32]
    args = [desc, np.ones((2, 2), np.float32), np.int32(2), np.int32(2)]
    evaluate(op("tt.descriptor_store", types, []), args, memory=memory)
    assert data[10] == 1.0 and data.sum() == 1.0


def test_reinterpret_tensor_descriptor_passes_a_descriptor_through() -> None:
    desc = _descriptor()
    got = evaluate(
        op("tt.reinterpret_tensor_descriptor", [ptr("i8")], tensordesc((2, 2), "f32")), [desc]
    )
    assert got[0] is desc


def test_reinterpret_tensor_descriptor_rejects_a_raw_pointer() -> None:
    with pytest.raises(Unsupported):
        evaluate(
            op("tt.reinterpret_tensor_descriptor", [ptr("i8")], tensordesc((2, 2), "f32")),
            [np.int64(4096)],
        )


# --------------------------------------------------------------------------- ttg


def test_convert_layout_is_the_identity() -> None:
    src = i32(1, 2, 3, 4)
    assert np.array_equal(one(op("ttg.convert_layout", [T4], T4), [src]), src)


def test_local_alloc_load_and_store() -> None:
    md = evaluate(op("ttg.local_alloc", [], memdesc((2, 2), "i32")))[0]
    assert isinstance(md, MemDesc)
    t = tensor((2, 2), "i32")
    payload = np.array([[1, 2], [3, 4]], np.int32)
    evaluate(op("ttg.local_store", [t, memdesc((2, 2), "i32")], []), [payload, md])
    got = one(op("ttg.local_load", [memdesc((2, 2), "i32")], t), [md])
    assert np.array_equal(got, payload)


def test_local_alloc_with_an_initialiser() -> None:
    payload = np.array([[1, 2], [3, 4]], np.int32)
    md = evaluate(
        op("ttg.local_alloc", [tensor((2, 2), "i32")], memdesc((2, 2), "i32")), [payload]
    )[0]
    assert isinstance(md, MemDesc) and np.array_equal(md.data, payload)


def test_memdesc_index_aliases_the_allocation() -> None:
    outer = memdesc((2, 2), "i32")
    md = evaluate(op("ttg.local_alloc", [], outer))[0]
    view = evaluate(op("ttg.memdesc_index", [outer, I32], memdesc((2,), "i32")), [md, np.int32(1)])[
        0
    ]
    assert isinstance(view, MemDesc) and isinstance(md, MemDesc)
    evaluate(
        op("ttg.local_store", [tensor((2,), "i32"), memdesc((2,), "i32")], []), [i32(5, 6), view]
    )
    assert np.array_equal(md.data[1], i32(5, 6))


def test_memdesc_subview() -> None:
    outer = memdesc((4, 4), "i32")
    md = evaluate(op("ttg.local_alloc", [], outer))[0]
    assert isinstance(md, MemDesc)
    md.data[...] = np.arange(16, dtype=np.int32).reshape(4, 4)
    types = [outer, I32, I32]
    view = evaluate(
        op("ttg.memdesc_subview", types, memdesc((2, 2), "i32")), [md, np.int32(1), np.int32(1)]
    )[0]
    assert isinstance(view, MemDesc)
    assert np.array_equal(view.data, np.array([[5, 6], [9, 10]], np.int32))


def test_memdesc_reshape_and_trans() -> None:
    outer = memdesc((2, 3), "i32")
    md = evaluate(op("ttg.local_alloc", [], outer))[0]
    assert isinstance(md, MemDesc)
    md.data[...] = np.arange(6, dtype=np.int32).reshape(2, 3)
    flat = evaluate(op("ttg.memdesc_reshape", [outer], memdesc((6,), "i32")), [md])[0]
    trans = evaluate(
        op("ttg.memdesc_trans", [outer], memdesc((3, 2), "i32"), attrs={"order": [1, 0]}), [md]
    )[0]
    assert isinstance(flat, MemDesc) and isinstance(trans, MemDesc)
    assert np.array_equal(flat.data, np.arange(6, dtype=np.int32))
    assert np.array_equal(trans.data, md.data.T)


def test_async_copy_global_to_local_is_a_masked_copy() -> None:
    memory = Memory()
    memory.register(4096, np.arange(8, dtype=np.int32))
    md = evaluate(op("ttg.local_alloc", [], memdesc((4,), "i32")))[0]
    addrs = 4096 + 4 * np.arange(4, dtype=np.int64)
    types = [tensor((4,), ptr("i32")), memdesc((4,), "i32"), T4B]
    args = [addrs, md, np.array([True, False, True, False])]
    evaluate(op("ttg.async_copy_global_to_local", types, []), args, memory=memory)
    assert isinstance(md, MemDesc) and np.array_equal(md.data, i32(0, 0, 2, 0))


@pytest.mark.parametrize(
    "name",
    [
        "ttg.local_dealloc",
        "ttg.async_commit_group",
        "ttg.async_wait",
        "ttg.async_bundle",
        "ttg.barrier",
        "gpu.barrier",
    ],
)
def test_no_op_ops(name: str) -> None:
    assert evaluate(op(name, [], [])) == []


def test_warp_specialize_runs_the_default_then_each_partition() -> None:
    from conftest import interp as make

    machine = make()
    default = region(block([], [op("ttg.warp_yield", [I32], [], operands=["%seed"], results=[])]))
    partition = region(
        block(
            [("%p0", I32)], [op("tt.assert", [I1], [], operands=["%p0"], attrs={"message": "no"})]
        )
    )
    inner = op("ttg.warp_specialize.partitions", [I32], [], regions=[partition], operands=["%seed"])
    outer = op(
        "ttg.warp_specialize", [], I32, regions=[default, region(block([], [inner]))], operands=[]
    )
    machine.scopes = [{"%seed": np.int32(1)}]
    machine.eval_op(outer)
    assert machine.value("%r0") == 1


def test_warp_specialize_rejects_a_partition_that_waits() -> None:
    from conftest import interp as make

    machine = make()
    default = region(block([], [op("ttg.warp_yield", [], [], operands=[], results=[])]))
    partition = region(block([], [op("ttng.wait_barrier", [], [])]))
    inner = op("ttg.warp_specialize.partitions", [], [], regions=[partition], operands=[])
    outer = op(
        "ttg.warp_specialize", [], [], regions=[default, region(block([], [inner]))], operands=[]
    )
    machine.scopes = [{}]
    with pytest.raises(Unsupported):
        machine.eval_op(outer)


# ------------------------------------------------------------- ttng: barriers and TMA copies

BAR = memdesc((1,), "i64")
BARRIER_OPS = [
    ("ttng.init_barrier", [], {"count": 1}),
    ("ttng.inval_barrier", [], {}),
    ("ttng.arrive_barrier", [], {"count": 1}),
    ("ttng.barrier_expect", [(I1, np.array(True))], {"size": 512}),
    # parity 1 on a fresh barrier: the phase before the first one counts as complete
    ("ttng.wait_barrier", [(I32, np.int32(1))], {"operandSegmentSizes": [1, 1, 0, 0]}),
]


def _alloc(ty_) -> MemDesc:
    md = evaluate(op("ttg.local_alloc", [], ty_))[0]
    assert isinstance(md, MemDesc)
    return md


@pytest.mark.parametrize(("name", "extra", "attrs"), BARRIER_OPS)
def test_ttng_barrier_ops_leave_the_buffer_alone(name, extra, attrs) -> None:
    md = _alloc(BAR)
    md.data[...] = 7
    types = [BAR, *(t for t, _ in extra)]
    args = [md, *(v for _, v in extra)]
    assert evaluate(op(name, types, [], attrs=attrs), args) == []
    assert md.data[0] == 7


def test_a_wait_on_a_phase_nothing_completes_is_reported_outside_a_warp_specialize() -> None:
    """Parity 0 of a fresh barrier needs an arrival; in straight-line code none can come."""
    md = _alloc(BAR)
    wait = op("ttng.wait_barrier", [BAR, I32], [], attrs={"operandSegmentSizes": [1, 1, 0, 0]})
    with pytest.raises(Unsupported, match="nothing before it completes"):
        evaluate(wait, [md, np.int32(0)])


@pytest.mark.parametrize(
    ("name", "attrs"),
    [
        ("ttng.fence_async_shared", {"bCluster": False}),
        ("ttng.async_tma_store_wait", {"pendings": 0}),
    ],
)
def test_ttng_fence_and_store_wait_are_no_ops(name: str, attrs: dict) -> None:
    assert evaluate(op(name, [], [], attrs=attrs)) == []


def _tma_load_op(
    segments: list[int], types: list, name: str = "ttng.async_tma_copy_global_to_local"
):
    return op(name, types, [], attrs={"operandSegmentSizes": segments})


def _tma_load_types() -> list:
    return [tensordesc((2, 2), "f32"), I32, I32, BAR, memdesc((2, 2), "f32"), I1]


def test_ttng_async_tma_copy_global_to_local_clips_and_fills() -> None:
    memory = Memory()
    memory.register(4096, np.arange(16, dtype=np.float32))
    md, bar = _alloc(memdesc((2, 2), "f32")), _alloc(BAR)
    args = [_descriptor(), np.int32(2), np.int32(2), bar, md, np.array(True)]
    evaluate(_tma_load_op([1, 2, 0, 1, 1, 1], _tma_load_types()), args, memory=memory)
    # only element (2, 2) is inside the 3x3 tensor; the rest of the block is the zero fill
    assert np.array_equal(md.data, np.array([[10.0, 0.0], [0.0, 0.0]], np.float32))


def test_ttng_async_tma_copy_global_to_local_honours_a_false_predicate() -> None:
    memory = Memory()
    memory.register(4096, np.arange(16, dtype=np.float32))
    md, bar = _alloc(memdesc((2, 2), "f32")), _alloc(BAR)
    md.data[...] = 5.0
    args = [_descriptor(), np.int32(0), np.int32(0), bar, md, np.array(False)]
    evaluate(_tma_load_op([1, 2, 0, 1, 1, 1], _tma_load_types()), args, memory=memory)
    assert np.array_equal(md.data, np.full((2, 2), 5.0, np.float32))


def test_ttng_async_tma_copy_global_to_local_rejects_im2col_offsets() -> None:
    memory = Memory()
    memory.register(4096, np.arange(16, dtype=np.float32))
    md, bar = _alloc(memdesc((2, 2), "f32")), _alloc(BAR)
    types = [*_tma_load_types()[:3], ty("i16"), *_tma_load_types()[3:]]
    args = [_descriptor(), np.int32(0), np.int32(0), np.int16(1), bar, md, np.array(True)]
    with pytest.raises(Unsupported):
        evaluate(_tma_load_op([1, 2, 1, 1, 1, 1], types), args, memory=memory)


def test_ttng_async_tma_copy_global_to_local_needs_its_segment_sizes() -> None:
    md, bar = _alloc(memdesc((2, 2), "f32")), _alloc(BAR)
    args = [_descriptor(), np.int32(0), np.int32(0), bar, md, np.array(True)]
    with pytest.raises(Unsupported):
        evaluate(op("ttng.async_tma_copy_global_to_local", _tma_load_types(), []), args)


def test_ttng_async_tma_copy_local_to_global_clips_to_the_shape() -> None:
    memory = Memory()
    data = np.zeros(16, np.float32)
    memory.register(4096, data)
    md = _alloc(memdesc((2, 2), "f32"))
    md.data[...] = 1.0
    types = [tensordesc((2, 2), "f32"), I32, I32, memdesc((2, 2), "f32")]
    args = [_descriptor(), np.int32(2), np.int32(2), md]
    evaluate(op("ttng.async_tma_copy_local_to_global", types, []), args, memory=memory)
    assert data[10] == 1.0 and data.sum() == 1.0


def test_unsigned_keeps_zero_dim_shape() -> None:
    from ttsem.ops import _unsigned

    x = np.array(-1, dtype=np.int8)
    u = _unsigned(x)
    assert u.shape == () and int(u) == 255
    v = _unsigned(np.array([[-1, 2]], dtype=np.int16)[:, ::-1])
    assert v.shape == (1, 2) and v.tolist() == [[2, 65535]]


def test_remf_keeps_the_sign_of_the_dividend() -> None:
    got = one(
        op("arith.remf", [F32, F32], F32),
        [np.array([-7.5, 7.5], np.float32), np.array([2.0, -2.0], np.float32)],
    )
    assert got.tolist() == [-1.5, 1.5]


# --------------------------------------------------------------------------- hex constants


def test_constant_hex_float_is_a_bit_pattern() -> None:
    attr = DenseAttr(FloatBits(0x7FC0, 16), BF16, (4,))
    got = one(op("arith.constant", [], tensor((4,), "bf16"), attrs={"value": attr}))
    assert got.dtype == np.uint16
    assert got.tolist() == [0x7FC0] * 4


def test_constant_hex_float_scalar_infinity() -> None:
    got = one(op("arith.constant", [], F32, attrs={"value": FloatBits(0xFF800000, 32)}))
    assert got.shape == () and np.isneginf(got)


def test_constant_hex_float_f16_nan() -> None:
    attr = DenseAttr(FloatBits(0x7E00, 16), ty("f16"), (2,))
    got = one(op("arith.constant", [], tensor((2,), "f16"), attrs={"value": attr}))
    assert got.dtype == np.float16 and np.isnan(got).all()


def test_constant_hex_float_list_mixes_with_decimals() -> None:
    attr = DenseAttr([FloatBits(0x7F800000, 32), 1.5], F32, (2,))
    got = one(op("arith.constant", [], tensor((2,), "f32"), attrs={"value": attr}))
    assert np.isposinf(got[0]) and got[1] == 1.5


def test_constant_hex_float_on_an_integer_type_is_rejected() -> None:
    attr = DenseAttr(FloatBits(0x7FC0, 16), I32, (2,))
    with pytest.raises(Unsupported):
        one(op("arith.constant", [], T4, attrs={"value": attr}))


def test_constant_rank_two_list_reshapes() -> None:
    attr = DenseAttr([[1, 2], [3, 4]], I32, (2, 2))
    got = one(op("arith.constant", [], tensor((2, 2), "i32"), attrs={"value": attr}))
    assert got.tolist() == [[1, 2], [3, 4]]


# --------------------------------------------------------------------- zero-dim float values


def test_bf16_scalar_add_stays_zero_dimensional() -> None:
    # `np.ascontiguousarray` used to make this shape (1,), and the `tt.store` that consumes a
    # reduced bf16 scalar then failed to broadcast against a 0-d address.
    a = from_float(np.float32(1.5), BF16)
    got = one(op("arith.addf", [BF16, BF16], BF16), [a, a])
    assert got.shape == () and to_float(got, BF16) == np.float32(3.0)


def test_bitcast_of_a_scalar_stays_zero_dimensional() -> None:
    got = one(op("arith.bitcast", [F32], I32), [np.array(1.0, np.float32)])
    assert got.shape == () and int(got) == 0x3F800000


def test_fp8_scalar_decodes_to_zero_dimensional() -> None:
    f8 = ty("f8E4M3FN")
    got = one(op("arith.extf", [f8], F32), [from_float(np.float32(2.0), f8)])
    assert got.shape == () and got == np.float32(2.0)


# --------------------------------------------------------------------------- new ops


def test_unsplat_makes_a_scalar() -> None:
    got = one(op("tt.unsplat", [tensor((1,), "i32")], I32), [i32(7)])
    assert got.shape == () and int(got) == 7


def test_unsplat_rejects_a_wider_tensor() -> None:
    with pytest.raises(Unsupported):
        one(op("tt.unsplat", [T4], I32), [i32(1, 2, 3, 4)])


def test_atomic_poll_matches_on_the_first_load() -> None:
    memory = Memory()
    buffer(memory, 4096, np.array([1, 2], dtype=np.int32))
    target = op("tt.atomic_poll", [ptr("i32"), T4, I64], T4B)
    ptrs = np.full(4, 4096, dtype=np.int64)
    got = one(target, [ptrs, i32(1, 1, 1, 1), np.int64(0)], memory=memory)
    assert got.tolist() == [True] * 4


def test_atomic_poll_with_a_timeout_reports_a_mismatch() -> None:
    memory = Memory()
    buffer(memory, 4096, np.array([1, 2], dtype=np.int32))
    target = op("tt.atomic_poll", [ptr("i32"), T4, I64], T4B)
    ptrs = np.array([4096, 4100, 4096, 4100], dtype=np.int64)
    got = one(target, [ptrs, i32(1, 1, 1, 1), np.int64(0)], memory=memory)
    assert got.tolist() == [True, False, True, False]


def test_atomic_poll_without_a_timeout_that_would_block_is_unsupported() -> None:
    memory = Memory()
    buffer(memory, 4096, np.array([0], dtype=np.int32))
    target = op("tt.atomic_poll", [ptr("i32"), I32], I1)
    with pytest.raises(Unsupported):
        one(target, [np.int64(4096), np.int32(1)], memory=memory)


def test_mulhiui_64_bit() -> None:
    a = np.array([1 << 40, (1 << 64) - 1], dtype=np.uint64).view(np.int64)
    b = np.array([1 << 40, (1 << 64) - 1], dtype=np.uint64).view(np.int64)
    t = tensor((2,), "i64")
    got = one(op("tt.mulhiui", [t, t], t), [a, b])
    expect = [((1 << 80) >> 64), (((1 << 64) - 1) ** 2) >> 64]
    assert got.view(np.uint64).tolist() == expect


def test_mulhiui_32_bit_is_unchanged() -> None:
    a = i32(1 << 20, -1)
    got = one(op("tt.mulhiui", [tensor((2,), "i32")] * 2, tensor((2,), "i32")), [a, a])
    assert got.view(np.uint32).tolist() == [(1 << 40) >> 32, ((1 << 32) - 1) ** 2 >> 32]


def _map_body(pack: int, names: list[str]) -> region:
    """A straight-line `divmod` region over `pack` lanes of two operands."""
    args = [(f"%a{p}", I32) for p in range(pack)] + [(f"%b{p}", I32) for p in range(pack)]
    ops = []
    results = []
    for name in names:
        for p in range(pack):
            res = f"%{name}{p}"
            ops.append(op(name, [I32, I32], I32, results=[res], operands=[f"%a{p}", f"%b{p}"]))
            results.append(res)
    ops.append(
        op("tt.map_elementwise.return", [I32] * len(results), [], operands=results, results=[])
    )
    return region(block(args, ops))


def test_map_elementwise_runs_the_region_on_every_element() -> None:
    body = _map_body(1, ["arith.divui", "arith.remui"])
    target = op(
        "tt.map_elementwise",
        [T4, T4],
        [T4, T4],
        attrs={"pack": 1},
        regions=[body],
    )
    a, b = i32(10, 11, 12, 13), i32(3, 4, 5, 6)
    got = evaluate(target, [a, b])
    assert list(got[0]) == [3, 2, 2, 2]
    assert list(got[1]) == [1, 3, 2, 1]


def test_map_elementwise_pack_two_groups_lanes() -> None:
    body = _map_body(2, ["arith.divui", "arith.remui"])
    target = op(
        "tt.map_elementwise",
        [T4, T4],
        [T4, T4],
        attrs={"pack": 2},
        regions=[body],
    )
    a, b = i32(10, 11, 12, 13), i32(3, 4, 5, 6)
    got = evaluate(target, [a, b])
    assert list(got[0]) == [3, 2, 2, 2]
    assert list(got[1]) == [1, 3, 2, 1]


def test_map_elementwise_runs_per_element_when_the_region_branches() -> None:
    # `cf.cond_br` makes the region non-straight-line, so it cannot be run on whole columns.
    body = region(
        block(
            [("%a0", I32), ("%b0", I32)],
            [
                op(
                    "arith.cmpi",
                    [I32, I32],
                    I1,
                    attrs={"predicate": 2},
                    results=["%c"],
                    operands=["%a0", "%b0"],
                ),
                op(
                    "cf.cond_br",
                    [I1],
                    [],
                    operands=["%c"],
                    results=[],
                    successors=["^t", "^f"],
                    attrs={"operandSegmentSizes": [1, 0, 0]},
                ),
            ],
            label="^bb0",
        ),
        block(
            [],
            [op("tt.map_elementwise.return", [I32], [], operands=["%one"], results=[])],
            label="^t",
        ),
        block(
            [],
            [op("tt.map_elementwise.return", [I32], [], operands=["%zero"], results=[])],
            label="^f",
        ),
    )
    target = op("tt.map_elementwise", [T4, T4], T4, attrs={"pack": 1}, regions=[body])
    machine = interp()
    machine.scopes = [
        {"%a0": i32(1, 5, 3, 9), "%a1": i32(4, 4, 4, 4), "%one": np.int32(1), "%zero": np.int32(0)}
    ]
    machine.eval_op(target)
    assert list(machine.value("%r0")) == [1, 0, 1, 0]


@pytest.mark.parametrize(
    ("symbol", "value", "expect"),
    [
        ("__nv_expf", 1.0, float(np.exp(np.float32(1.0)))),
        ("__nv_logf", 4.0, float(np.log(np.float32(4.0)))),
        ("__nv_sqrtf", 9.0, 3.0),
        ("__nv_rsqrtf", 4.0, 0.5),
        ("__nv_fabsf", -2.5, 2.5),
        ("__nv_floorf", 1.7, 1.0),
        ("__nv_ceilf", 1.2, 2.0),
        ("__nv_rintf", 2.5, 2.0),
    ],
)
def test_extern_elementwise_unary(symbol: str, value: float, expect: float) -> None:
    got = one(
        op("tt.extern_elementwise", [F32], F32, attrs={"symbol": symbol}),
        [np.array(value, np.float32)],
    )
    assert float(got) == pytest.approx(expect)


def test_extern_elementwise_double_variant() -> None:
    got = one(
        op("tt.extern_elementwise", [ty("f64")], ty("f64"), attrs={"symbol": "__nv_rint"}),
        [np.array(-2.5, np.float64)],
    )
    assert got.dtype == np.float64 and float(got) == -2.0


def test_extern_elementwise_binary_and_ternary() -> None:
    powf = op("tt.extern_elementwise", [F32, F32], F32, attrs={"symbol": "__nv_powf"})
    assert float(one(powf, [np.float32(2.0), np.float32(10.0)])) == 1024.0
    fmaf = op("tt.extern_elementwise", [F32, F32, F32], F32, attrs={"symbol": "__nv_fmaf"})
    assert float(one(fmaf, [np.float32(2.0), np.float32(3.0), np.float32(1.0)])) == 7.0


def test_extern_elementwise_unknown_symbol_names_it() -> None:
    target = op("tt.extern_elementwise", [F32], F32, attrs={"symbol": "__nv_j0f"})
    with pytest.raises(Unsupported, match="__nv_j0f"):
        one(target, [np.float32(1.0)])


def test_llvm_intr_assume_is_a_no_op() -> None:
    assert evaluate(op("llvm.intr.assume", [I1], []), [np.array(True)]) == []


def test_token_type_is_an_opaque_byte() -> None:
    token = Type(kind="other", name="!ttg.async.token")
    got = one(op("ub.poison", [], token))
    assert got.dtype == np.int8 and got.shape == ()


# --------------------------------------------------------------------------- i1 arithmetic


@pytest.mark.parametrize(
    ("name", "expect"),
    [
        # `x + y` and `x - y` over [F,F,T,T] and [F,T,F,T] wrap mod 2 (triton#10919)
        ("arith.addi", [False, True, True, False]),
        ("arith.subi", [False, True, True, False]),
        ("arith.muli", [False, False, False, True]),
        ("arith.andi", [False, False, False, True]),
        ("arith.ori", [False, True, True, True]),
        ("arith.xori", [False, True, True, False]),
        # signed: `true` is -1, so it is the smaller of the two
        ("arith.maxsi", [False, False, False, True]),
        ("arith.minsi", [False, True, True, True]),
        # unsigned: `true` is 1, so it is the larger
        ("arith.maxui", [False, True, True, True]),
        ("arith.minui", [False, False, False, True]),
    ],
)
def test_i1_binops_wrap_mod_two(name: str, expect: list[bool]) -> None:
    x = np.array([False, False, True, True])
    y = np.array([False, True, False, True])
    got = one(op(name, [T4B, T4B], T4B), [x, y])
    assert got.dtype == np.bool_ and got.tolist() == expect


def test_i1_signed_compare_reads_true_as_minus_one() -> None:
    x = np.array([True, False])
    y = np.array([False, True])
    slt = one(op("arith.cmpi", [T4B, T4B], T4B, attrs={"predicate": 2}), [x, y])
    ult = one(op("arith.cmpi", [T4B, T4B], T4B, attrs={"predicate": 6}), [x, y])
    assert slt.tolist() == [True, False]
    assert ult.tolist() == [False, True]


def test_histogram_drops_values_at_or_above_the_bin_count() -> None:
    # `test_histogram_out_of_range` on the device: 4 bins, and 4, 5, -1 belong to none of them.
    src = np.array([0, 1, 2, 3, 4, 4, 5, -1], np.int32)
    got = one(op("tt.histogram", [tensor((8,), "i32")], T4), [src])
    assert np.array_equal(got, i32(1, 1, 1, 1))


def test_histogram_of_one_bin_drops_a_one() -> None:
    src = np.array([1], np.int32)
    got = one(op("tt.histogram", [tensor((1,), "i32")], tensor((1,), "i32")), [src])
    assert got.tolist() == [0]


# --------------------------------------------------------------------------- fused fma


@pytest.mark.parametrize(
    ("a", "b", "c", "expect"),
    [
        # the eight cases of `test_math_fma_op_special_values`, which an unfused `a*b+c` in
        # float64 gets wrong on the third and the last
        (float("nan"), 1.0, 1.0, "nan"),
        (float("inf"), 0.0, 1.0, "nan"),
        (2.0, 3.0, float("inf"), "inf"),
        (-1.7976931348623157e308, 1.7976931348623157e308, float("inf"), "inf"),
        (-0.0, 1.0, -0.0, "-0.0"),
        (1.0, 1.0, -1.0, "0.0"),
        (1.7976931348623157e308, 2.0, 1.7976931348623157e308, "inf"),
        (-5e-324, 0.5, 0.0, "-0.0"),
    ],
)
def test_fma_rounds_once_over_the_exact_product(a: float, b: float, c: float, expect: str) -> None:
    from ttsem.ops import _fma_scalar

    got = _fma_scalar(a, b, c)
    assert repr(got) == expect if expect != "nan" else np.isnan(got)


_F32_MAX = float(np.finfo(np.float32).max)
_F32_SUB = float(np.finfo(np.float32).smallest_subnormal)
_F32_TINY = float(np.finfo(np.float32).tiny)


@pytest.mark.parametrize(
    "a, b, c, expect",
    [
        (4097.0, 4097.0, 2.0**-30, 16785410.0),  # just above a float32 midpoint
        (4097.0, 4097.0, -(2.0**-30), 16785408.0),  # just below it
        (float(31 * 2**59), float(1082401 * 2**44), -1.0, _F32_MAX),  # under the overflow midpoint
        (float(31 * 2**59), float(1082401 * 2**44), 0.0, math.inf),  # on it: ties to even overflow
        (_F32_MAX, 1.0, 1.0, _F32_MAX),
        (_F32_MAX, 2.0, -_F32_MAX, _F32_MAX),
        (-_F32_SUB, 0.5, 0.0, -0.0),  # an underflow tie rounds to even and keeps its sign
        (_F32_SUB, 0.5, _F32_SUB, 2 * _F32_SUB),
        (_F32_TINY, 0.5, 0.0, _F32_TINY / 2),
    ],
)
def test_fma_rounds_once_into_float32(a: float, b: float, c: float, expect: float) -> None:
    """The cases of `test_math_fma_op_edge_cases[float32]`: a `float64` rounding followed by a
    cast to `float32` gets the first four wrong, the device (one rounding) gets them all right."""
    got = one(op("math.fma", [F32, F32, F32], F32), [np.float32(a), np.float32(b), np.float32(c)])
    assert got.dtype == np.float32
    assert float(got) == expect
    assert math.copysign(1.0, float(got)) == math.copysign(1.0, expect)


def test_fma_op_keeps_the_bits_the_unfused_expression_loses() -> None:
    f64 = ty("f64")
    a, c = np.float64(0.1), np.float64(-0.010000000000000002)
    got = one(op("math.fma", [f64, f64, f64], f64), [a, a, c])
    assert float(a * a + c) == 0.0  # the unfused expression cancels exactly
    assert float(got) == -8.326672684688674e-19


def test_reduce_of_bf16_does_not_re_encode_the_accumulator() -> None:
    # a left fold of 1024 bf16 ones saturates at 256; `cast_to` on the result used to read the
    # bit pattern 0x4380 as a number and encode it again, giving 17280 (`test_sum_dtype`).
    body = region(
        block(
            [("%l", BF16), ("%r", BF16)],
            [
                op("arith.addf", [BF16, BF16], BF16, results=["%s"], operands=["%l", "%r"]),
                op("tt.reduce.return", [BF16], [], operands=["%s"], results=[]),
            ],
        )
    )
    src = np.full(1024, from_float(np.float32(1.0), BF16), dtype=np.uint16)
    target = op("tt.reduce", [tensor((1024,), "bf16")], BF16, attrs={"axis": 0}, regions=[body])
    got = one(target, [src])
    assert got.dtype == np.uint16 and float(to_float(got, BF16)) == 256.0


def test_atomic_add_on_bf16_adds_values_not_bit_patterns() -> None:
    memory = Memory()
    data = buffer(memory, 4096, np.array([from_float(np.float32(1.5), BF16)] * 2, np.uint16))
    t = tensor((2,), "bf16")
    target = op("tt.atomic_rmw", [tensor((2,), ptr("bf16")), t], t, attrs={"atomic_rmw_op": 5})
    ptrs = np.array([4096, 4098], dtype=np.int64)
    vals = np.array([from_float(np.float32(0.5), BF16)] * 2, np.uint16)
    old = one(target, [ptrs, vals], memory=memory)
    assert [float(v) for v in to_float(old, BF16)] == [1.5, 1.5]
    assert [float(v) for v in to_float(data, BF16)] == [2.0, 2.0]


def test_atomic_max_on_bf16_orders_by_value() -> None:
    memory = Memory()
    negative = from_float(np.float32(-4.0), BF16)
    data = buffer(memory, 4096, np.array([negative], np.uint16))
    t = tensor((1,), "bf16")
    target = op("tt.atomic_rmw", [tensor((1,), ptr("bf16")), t], t, attrs={"atomic_rmw_op": 6})
    vals = np.array([from_float(np.float32(-1.0), BF16)], np.uint16)
    one(target, [np.array([4096], dtype=np.int64), vals], memory=memory)
    assert float(to_float(data, BF16)[0]) == -1.0


# ------------------------------------------------------- rounded integer division


@pytest.mark.parametrize(
    ("name", "expect"),
    [
        # 7/2, -7/2, 7/-2, -7/-2, 6/2 -- the sign of an operand is what separates the three
        ("arith.divsi", [3, -3, -3, 3, 3]),  # toward zero, for contrast
        ("arith.floordivsi", [3, -4, -4, 3, 3]),
        ("arith.ceildivsi", [4, -3, -3, 4, 3]),
    ],
)
def test_rounded_division_directions(name: str, expect: list[int]) -> None:
    t = tensor((5,), "i32")
    a = i32(7, -7, 7, -7, 6)
    b = i32(2, 2, -2, -2, 2)
    assert one(op(name, [t, t], t), [a, b]).tolist() == expect


def test_ceildivui_reads_the_operands_unsigned() -> None:
    t = tensor((3,), "i32")
    a = i32(7, -1, 8)  # -1 is 4294967295 unsigned
    b = i32(2, 2, 2)
    got = one(op("arith.ceildivui", [t, t], t), [a, b])
    assert got.view(np.uint32).tolist() == [4, 2147483648, 4]


@pytest.mark.parametrize("name", ["arith.floordivsi", "arith.ceildivsi", "arith.ceildivui"])
def test_rounded_division_by_zero_is_poison(name: str) -> None:
    t = tensor((2,), "i32")
    with pytest.raises(Poison):
        one(op(name, [t, t], t), [i32(1, 2), i32(1, 0)])


@pytest.mark.parametrize("name", ["arith.floordivsi", "arith.ceildivsi"])
def test_rounded_division_of_int_min_by_minus_one_is_poison(name: str) -> None:
    t = tensor((1,), "i32")
    with pytest.raises(Poison):
        one(op(name, [t, t], t), [i32(-(2**31)), i32(-1)])


def test_store_with_an_all_false_mask_touches_nothing() -> None:
    memory = Memory()
    data = np.arange(4, dtype=np.int32)
    memory.register(4096, data)
    addrs = np.array([4096, 4100, 4104, 4108], dtype=np.int64)
    memory.store(addrs, np.full(4, 9, np.int32), np.zeros(4, bool))
    assert data.tolist() == [0, 1, 2, 3]
    got = memory.load(addrs, np.zeros(4, bool), np.int32(7), np.dtype(np.int32))
    assert got.tolist() == [7, 7, 7, 7]


def test_two_lanes_storing_different_values_to_one_address_is_poison() -> None:
    import pytest

    from ttsem.memory import Memory
    from ttsem.values import Poison

    mem = Memory()
    mem.register(0, np.zeros(16, np.int32))
    addrs = np.array([0, 4, 4, 8], np.int64)
    mem.store(addrs, np.array([1, 2, 2, 3], np.int32), None)  # replicated: the same value twice
    assert mem.load(np.array([4]), None, None, np.int32)[0] == 2
    with pytest.raises(Poison):
        mem.store(addrs, np.array([1, 5, 6, 3], np.int32), None)


def test_fp8_encode_keeps_the_sign_of_zero() -> None:
    # tiny values round to zero; the zero keeps the sign of the input, as the device does
    tiny = np.array([1e-9, -1e-9, 0.0, -0.0], np.float32)
    assert from_float(tiny, ty("f8E5M2")).tolist() == [0x00, 0x80, 0x00, 0x80]
    assert from_float(tiny, ty("f8E4M3FN")).tolist() == [0x00, 0x80, 0x00, 0x80]


def test_make_tensor_descriptor_reads_the_padding_option() -> None:
    result = tensordesc((2, 2), "f32")
    types = [ptr("f32"), I32, I32, I64, I64]
    args = [np.int64(4096), np.int32(3), np.int32(3), np.int64(4), np.int64(1)]
    o = op("tt.make_tensor_descriptor", types, result)
    o.attrs["padding"] = "#tt.padding_option<nan>"
    assert evaluate(o, args)[0].pad == "nan"
    o.attrs["padding"] = "#tt.padding_option<zero>"
    assert evaluate(o, args)[0].pad == "zero"
    o.attrs["padding"] = 2  # the TTIR spelling: the enum's value, PAD_NAN
    assert evaluate(o, args)[0].pad == "nan"
    o.attrs["padding"] = 1
    assert evaluate(o, args)[0].pad == "zero"


def test_descriptor_reduce_combines_with_memory() -> None:
    memory = Memory()
    data = np.arange(16, dtype=np.float32)
    memory.register(4096, data)
    desc = _descriptor()
    types = [tensordesc((2, 2), "f32"), tensor((2, 2), "f32"), I32, I32]
    o = op("tt.descriptor_reduce", types, [])
    o.attrs["kind"] = "#tt.descriptor_reduce_kind<add>"
    half = np.full((2, 2), 0.5, np.float32)
    evaluate(o, [desc, half, np.int32(1), np.int32(1)], memory=memory)
    # the block at (1, 1) of the 3x3 tensor with row stride 4: elements 5, 6, 9, 10
    assert data[[5, 6, 9, 10]].tolist() == [5.5, 6.5, 9.5, 10.5]
    o.attrs["kind"] = "#tt.descriptor_reduce_kind<max>"
    seven = np.full((2, 2), 7.0, np.float32)
    evaluate(o, [desc, seven, np.int32(1), np.int32(1)], memory=memory)
    assert data[[5, 6, 9, 10]].tolist() == [7.0, 7.0, 9.5, 10.5]
    # a block that reaches past the tensor clips like a store
    evaluate(o, [desc, np.full((2, 2), 99.0, np.float32), np.int32(2), np.int32(2)], memory=memory)
    assert data[10] == 99.0 and data[11] == 11.0 and data[14] == 14.0
    o.attrs["kind"] = "#tt.descriptor_reduce_kind<inc>"
    with pytest.raises(Unsupported):
        evaluate(o, [desc, seven, np.int32(1), np.int32(1)], memory=memory)


def test_descriptor_reduce_on_integers_is_bitwise() -> None:
    memory = Memory()
    data = np.full(16, 0b1100, np.int32)
    memory.register(4096, data)
    desc = Descriptor(4096, (3, 3), (4, 1), (2, 2), I32, "zero")
    types = [tensordesc((2, 2), "i32"), tensor((2, 2), "i32"), I32, I32]
    o = op("tt.descriptor_reduce", types, [])
    o.attrs["kind"] = "#tt.descriptor_reduce_kind<xor>"
    evaluate(o, [desc, np.full((2, 2), 0b1010, np.int32), np.int32(0), np.int32(0)], memory=memory)
    assert data[0] == 0b0110 and data[1] == 0b0110 and data[3] == 0b1100


def test_fp_to_fp_downcast_saturates_like_cvt_satfinite() -> None:
    from ttsem.values import from_float_rtz, max_finite

    big = np.finfo(np.float32).max  # past every narrower type's range, bf16 included
    src = np.array([np.inf, -np.inf, big, -big, np.nan, 1.5], np.float32)
    for name in ("f8E5M2", "f8E4M3FN", "f16", "bf16"):
        got = one(op("tt.fp_to_fp", [tensor((6,), "f32")], tensor((6,), name)), [src])
        back = to_float(got, ty(name)).astype(np.float64)
        limit = max_finite(ty(name))
        assert back[:4].tolist() == [limit, -limit, limit, -limit], name
        assert np.isnan(back[4]) and back[5] == 1.5
    # round toward zero: 1.1 in e5m2 is 1.0, not 1.25; 65519 in f16 is 65504, not inf
    assert from_float_rtz(np.array([1.1, -1.1], np.float32), ty("f8E5M2")).tolist() == [0x3C, 0xBC]
    assert from_float_rtz(np.array([65519.0], np.float32), ty("f16")).tolist() == [65504.0]
    assert from_float_rtz(np.array([1.00390625], np.float32), ty("bf16")).tolist() == [0x3F80]


def test_reduce_kind_accepts_the_enum_value() -> None:
    memory = Memory()
    data = np.zeros(16, np.int32)
    memory.register(4096, data)
    desc = Descriptor(4096, (3, 3), (4, 1), (2, 2), I32, "zero")
    types = [tensordesc((2, 2), "i32"), tensor((2, 2), "i32"), I32, I32]
    o = op("tt.descriptor_reduce", types, [])
    o.attrs["kind"] = 1  # TTGIR prints #tt.descriptor_reduce_kind<add> as its value
    evaluate(o, [desc, np.full((2, 2), 5, np.int32), np.int32(0), np.int32(0)], memory=memory)
    assert data[0] == 5 and data[5] == 5 and data[2] == 0


def test_select_picks_a_descriptor() -> None:
    a = Descriptor(4096, (3, 3), (4, 1), (2, 2), F32, "zero")
    b = Descriptor(8192, (3, 3), (4, 1), (2, 2), F32, "nan")
    types = [ty("i1"), tensordesc((2, 2), "f32"), tensordesc((2, 2), "f32")]
    assert (
        evaluate(op("arith.select", types, tensordesc((2, 2), "f32")), [np.bool_(True), a, b])[0]
        is a
    )
    assert (
        evaluate(op("arith.select", types, tensordesc((2, 2), "f32")), [np.bool_(False), a, b])[0]
        is b
    )


def test_descriptor_store_notes_the_padded_granule() -> None:
    # a 3-wide f32 tensor ends 12 bytes into a 16-byte granule: element 3 of each row shares
    # the granule of element 2, element 4 (next granule) and a row past the end do not
    memory = Memory()
    data = np.zeros(16, np.float32)
    memory.register(4096, data)
    desc = Descriptor(4096, (3, 3), (4, 1), (2, 4), F32, "zero")
    types = [tensordesc((2, 4), "f32"), tensor((2, 4), "f32"), I32, I32]
    block = np.arange(8, dtype=np.float32).reshape(2, 4) + 1
    evaluate(
        op("tt.descriptor_store", types, []), [desc, block, np.int32(2), np.int32(0)], memory=memory
    )
    assert data.reshape(4, 4)[2].tolist() == [1.0, 2.0, 3.0, 0.0]  # row 2 stored, element 3 clipped
    assert memory.pad_writes == {4096 + (2 * 4 + 3) * 4: np.float32(4.0).tobytes()}


def test_descriptor_reduce_min_max_follow_the_descriptor_signedness() -> None:
    memory = Memory()
    data = np.full(16, -1, np.int32)  # 0xFFFFFFFF: the largest uint32, the smallest int32
    memory.register(4096, data)
    types = [tensordesc((2, 2), "i32"), tensor((2, 2), "i32"), I32, I32]
    o = op("tt.descriptor_reduce", types, [])
    o.attrs["kind"] = "#tt.descriptor_reduce_kind<max>"
    one_block = np.ones((2, 2), np.int32)
    signed = Descriptor(4096, (3, 3), (4, 1), (2, 2), I32, "zero")
    evaluate(o, [signed, one_block, np.int32(0), np.int32(0)], memory=memory)
    assert data[0] == 1  # signed: 1 > -1
    data[:] = -1
    unsigned = Descriptor(4096, (3, 3), (4, 1), (2, 2), I32, "zero", unsigned=True)
    evaluate(o, [unsigned, one_block, np.int32(0), np.int32(0)], memory=memory)
    assert data[0] == -1  # unsigned: 0xFFFFFFFF > 1
