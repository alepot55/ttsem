"""Tests for `mlir.py`: the eight fixtures parse, specific ops and types come out shaped as
expected, and `parse(unparse(parse(text)))` is structurally equal to `parse(text)`.

The round trip is checked ignoring `Op.text` and `Op.loc` (raw source spans that the printer
does not try to reproduce byte for byte) and ignoring `Type.name` on compound kinds (`tensor`,
`ptr`, `memdesc`, `tensordesc`), which DESIGN.md documents as meaningful only for scalar
spellings; `shape`, `elem` and `encoding` are compared as written.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from ttsem import mlir
import pytest
from ttsem.ir_types import Block, DenseAttr, FloatBits, Op, Region, Type

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURE_NAMES = [
    "p1.ttir.generic",
    "p1.ttgir.generic",
    "p2.ttir.generic",
    "p2.ttgir.generic",
    "p3.ttir.generic",
    "p3.ttgir.generic",
    "p4.ttir.generic",
    "p4.ttgir.generic",
]


def _read_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text()


def _find_ops(ops: list[Op], name: str) -> list[Op]:
    found = []
    for op in ops:
        if op.name == name:
            found.append(op)
        for region in op.regions:
            for block in region.blocks:
                found.extend(_find_ops(block.ops, name))
    return found


# --------------------------------------------------------------------------- basic parsing


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_parses_every_fixture(name: str) -> None:
    module = mlir.parse(_read_fixture(name))
    assert len(module.ops) == 1
    assert module.ops[0].name == "tt.func"
    assert "kernel" in module.funcs
    assert module.funcs["kernel"] is module.ops[0]


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_op_keeps_its_source_text(name: str) -> None:
    module = mlir.parse(_read_fixture(name))
    func = module.funcs["kernel"]
    assert func.text is not None and func.text.startswith('"tt.func"')
    body = func.regions[0].blocks[0].ops
    assert body, "expected at least one op in the function body"
    assert all(op.text for op in body)


def test_op_counts_are_stable() -> None:
    # arith.constant is by far the most common op in these kernels; a stable count across
    # runs is a cheap signal that the tokenizer is not silently dropping or duplicating text.
    module = mlir.parse(_read_fixture("p1.ttir.generic"))
    constants = _find_ops(module.ops, "arith.constant")
    assert len(constants) == 8


# --------------------------------------------------------------------------- specific ops


def test_tt_load_with_mask_and_other() -> None:
    module = mlir.parse(_read_fixture("p1.ttir.generic"))
    [load] = _find_ops(module.ops, "tt.load")
    assert len(load.operands) == 3  # pointer, mask, other
    assert load.attrs["cache"] == 1
    assert load.attrs["evict"] == 1
    assert load.attrs["isVolatile"] is False
    assert load.attrs["operandSegmentSizes"] == [1, 1, 1]
    ptr_ty, mask_ty, other_ty = load.operand_types
    assert ptr_ty.kind == "tensor" and ptr_ty.elem.kind == "ptr"
    assert mask_ty.kind == "tensor" and mask_ty.elem.kind == "int" and mask_ty.elem.width == 1
    assert other_ty.kind == "tensor" and other_ty.elem.name == "i32"


def test_tt_reduce_region_and_block_args() -> None:
    module = mlir.parse(_read_fixture("p3.ttir.generic"))
    reduces = _find_ops(module.ops, "tt.reduce")
    assert len(reduces) == 2
    reduce = reduces[0]
    assert reduce.attrs["axis"] == 1
    [region] = reduce.regions
    [block] = region.blocks
    assert block.label == "^bb0"
    assert [name for name, _ in block.args] == ["%arg13", "%arg14"]
    assert all(ty.kind == "float" and ty.name == "f32" for _, ty in block.args)
    combine_names = [op.name for op in block.ops]
    assert combine_names == ["arith.addf", "tt.reduce.return"]


def test_scf_for_with_iter_args() -> None:
    module = mlir.parse(_read_fixture("p3.ttgir.generic"))
    [loop] = _find_ops(module.ops, "scf.for")
    # a 3-result "%44:3 = scf.for" expands to the "%44#0", "%44#1", "%44#2" convention
    assert loop.results == ["%44#0", "%44#1", "%44#2"]
    assert len(loop.operands) == 6  # lower, upper, step, plus three iter_args
    [region] = loop.regions
    [block] = region.blocks
    assert block.args[0][1].kind == "int"  # the induction variable
    assert len(block.args) == 4  # induction variable + 3 iter_args
    assert block.ops[-1].name == "scf.yield"


def test_warp_specialize_has_two_regions() -> None:
    module = mlir.parse(_read_fixture("p4.ttgir.generic"))
    [ws] = _find_ops(module.ops, "ttg.warp_specialize")
    assert len(ws.regions) == 2
    assert ws.attrs["partitionNumWarps"] == [2]
    assert ws.attrs["requestedRegisters"] == [24]
    [inner_for] = _find_ops(ws.regions[0].blocks[0].ops, "scf.for")
    assert inner_for.attrs["tt.warp_specialize"] is True
    partitions = _find_ops(module.ops, "ttg.warp_specialize.partitions")
    assert len(partitions) == 1
    [block] = partitions[0].regions[0].blocks
    assert block.label == "^bb0"
    assert len(block.args) == len(partitions[0].operands)


def test_tt_dot() -> None:
    module = mlir.parse(_read_fixture("p4.ttgir.generic"))
    [dot] = _find_ops(module.ops, "tt.dot")
    assert len(dot.operands) == 3
    assert dot.attrs["inputPrecision"] == 0
    a_ty, b_ty, acc_ty = dot.operand_types
    assert a_ty.elem.name == "bf16" and b_ty.elem.name == "bf16"
    assert acc_ty.elem.name == "f32"


# --------------------------------------------------------------------------- types


def test_tensor_shape_elem_encoding() -> None:
    module = mlir.parse(_read_fixture("p1.ttgir.generic"))
    func = module.funcs["kernel"]
    const = _find_ops(func.regions[0].blocks[0].ops, "arith.constant")[0]
    [ty] = const.result_types
    assert ty.kind == "tensor"
    assert ty.shape == (1024,)
    assert ty.elem.kind == "int" and ty.elem.name == "i32"
    assert ty.encoding is not None and "ttg.blocked" in ty.encoding


def test_ptr_pointee() -> None:
    module = mlir.parse(_read_fixture("p1.ttir.generic"))
    func = module.funcs["kernel"]
    [(name, ty)] = [a for a in func.regions[0].blocks[0].args if a[0] == "%arg0"]
    assert ty.kind == "ptr"
    assert ty.elem.kind == "int" and ty.elem.name == "i8"


def test_memdesc_shape_elem_encoding() -> None:
    module = mlir.parse(_read_fixture("p4.ttgir.generic"))
    allocs = _find_ops(module.ops, "ttg.local_alloc")
    assert allocs
    [ty] = allocs[0].result_types
    assert ty.kind == "memdesc"
    assert ty.shape == (2, 16, 16)
    assert ty.elem.name == "bf16"
    assert ty.encoding is not None and "shared_memory" in ty.encoding and "mutable" in ty.encoding


def test_tensordesc_shape_and_elem() -> None:
    module = mlir.parse(_read_fixture("p4.ttir.generic"))
    func = module.funcs["kernel"]
    tensordescs = [t for _, t in func.regions[0].blocks[0].args if t.kind == "tensordesc"]
    assert tensordescs
    ty = tensordescs[0]
    assert ty.shape == (16, 64)
    assert ty.elem.name == "f32"


# --------------------------------------------------------------------------- dense attrs


def test_dense_splat_scalar() -> None:
    module = mlir.parse(_read_fixture("p1.ttir.generic"))
    const = _find_ops(module.ops, "arith.constant")[0]
    dense = const.attrs["value"]
    assert isinstance(dense, DenseAttr)
    assert dense.value == 512
    assert dense.shape == (1024,)
    assert dense.elem_type.name == "i32"


def test_dense_rank_two_list() -> None:
    # none of the eight fixtures carry a rank-2 dense list literal (only splats), so this
    # exercises the nested-list branch directly against a hand-written generic snippet.
    text = (
        '"builtin.module"() ({\n'
        '  %0 = "arith.constant"() <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>}> '
        ": () -> tensor<2x2xi32>\n"
        "}) : () -> ()"
    )
    module = mlir.parse(text)
    dense = module.ops[0].attrs["value"]
    assert isinstance(dense, DenseAttr)
    assert dense.value == [[1, 2], [3, 4]]
    assert dense.shape == (2, 2)


def _dense_of(payload: str, tensor_type: str) -> DenseAttr:
    text = (
        '"builtin.module"() ({\n'
        f'  %0 = "arith.constant"() <{{value = dense<{payload}> : {tensor_type}}}> '
        f": () -> {tensor_type}\n"
        "}) : () -> ()"
    )
    dense = mlir.parse(text).ops[0].attrs["value"]
    assert isinstance(dense, DenseAttr)
    return dense


@pytest.mark.parametrize(
    ("payload", "tensor_type", "bits", "width"),
    [
        ("0x7FC0", "tensor<4xbf16>", 0x7FC0, 16),  # a bf16 NaN
        ("0x7E00", "tensor<4xf16>", 0x7E00, 16),  # an f16 NaN
        ("0x7F800000", "tensor<4xf32>", 0x7F800000, 32),  # +inf
        ("0xFF800000", "tensor<4xf32>", 0xFF800000, 32),  # -inf
        ("0xFFF8000000000000", "tensor<2xf64>", 0xFFF8000000000000, 64),  # an f64 NaN
    ],
)
def test_dense_hex_float_is_a_bit_pattern(
    payload: str, tensor_type: str, bits: int, width: int
) -> None:
    dense = _dense_of(payload, tensor_type)
    assert dense.value == FloatBits(bits, width)


def test_dense_hex_int_is_a_plain_integer() -> None:
    assert _dense_of("0x10", "tensor<4xi32>").value == 16


def test_dense_hex_list_mixes_bit_patterns_and_decimals() -> None:
    dense = _dense_of("[0x7FC00000, 1.500000e+00]", "tensor<2xf32>")
    assert dense.value == [FloatBits(0x7FC00000, 32), 1.5]


def test_dense_hex_blob_is_raw_little_endian_data() -> None:
    dense = _dense_of('"0x0000803F0000C0BF"', "tensor<2xf32>")
    assert dense.value == [FloatBits(0x3F800000, 32), FloatBits(0xBFC00000, 32)]


def test_dense_hex_blob_of_one_element_is_a_splat() -> None:
    assert _dense_of('"0x0000803F"', "tensor<4xf32>").value == FloatBits(0x3F800000, 32)


def test_scalar_hex_float_attribute_is_a_bit_pattern() -> None:
    text = (
        '"builtin.module"() ({\n'
        '  %0 = "arith.constant"() <{value = 0x7FC00000 : f32}> : () -> f32\n'
        "}) : () -> ()"
    )
    assert mlir.parse(text).ops[0].attrs["value"] == FloatBits(0x7FC00000, 32)


@pytest.mark.parametrize(
    "payload",
    ["dense<0x7FC0> : tensor<4xbf16>", "dense<[0x7FC00000, 1.500000e+00]> : tensor<2xf32>"],
)
def test_hex_float_round_trips_through_the_unparser(payload: str) -> None:
    text = (
        '"builtin.module"() ({\n'
        f'  %0 = "arith.constant"() <{{value = {payload}}}> : () -> tensor<4xi32>\n'
        "}) : () -> ()"
    )
    module = mlir.parse(text)
    assert mlir.parse(mlir.unparse(module)).ops[0].attrs == module.ops[0].attrs


# --------------------------------------------------------------------------- aliases


def test_attribute_alias_is_substituted() -> None:
    text = (
        "#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], "
        "warpsPerCTA = [1], order = [0]}>\n"
        '"builtin.module"() ({\n'
        '  %0 = "arith.constant"() <{value = dense<1> : tensor<32xi32, #blocked>}> '
        ": () -> tensor<32xi32, #blocked>\n"
        "}) : () -> ()"
    )
    module = mlir.parse(text)
    [ty] = module.ops[0].result_types
    assert ty.encoding is not None
    assert ty.encoding.startswith("#ttg.blocked<")
    assert "#blocked" not in ty.encoding  # substituted, not echoed back verbatim


# --------------------------------------------------------------------------- control flow


def test_cf_br_and_cf_cond_br_successors() -> None:
    text = (
        '"builtin.module"() ({\n'
        '  "tt.func"() <{function_type = (i32) -> (i32), sym_name = "k", '
        'sym_visibility = "public"}> ({\n'
        "  ^bb0(%arg0: i32):\n"
        '    "cf.br"(%arg0)[^bb1] : (i32) -> ()\n'
        "  ^bb1(%x: i32):\n"
        '    "tt.return"(%x) : (i32) -> ()\n'
        "  }) : () -> ()\n"
        "}) : () -> ()"
    )
    module = mlir.parse(text)
    func = module.funcs["k"]
    entry = func.regions[0].blocks[0]
    br = entry.ops[0]
    assert br.name == "cf.br"
    assert br.successors == ["^bb1"]
    assert br.operands == ["%arg0"]
    target = func.regions[0].blocks[1]
    assert target.label == "^bb1"
    assert target.args == [("%x", target.args[0][1])]
    assert target.args[0][1].kind == "int"


# --------------------------------------------------------------------------- round trip


def _norm_type(t: Type | None) -> Type | None:
    if t is None:
        return None
    name = t.name if t.kind in ("int", "float", "index", "other") else ""
    return dataclasses.replace(t, name=name, elem=_norm_type(t.elem))


def _norm_op(op: Op) -> Op:
    return dataclasses.replace(
        op,
        text=None,
        loc=None,
        operand_types=[_norm_type(t) for t in op.operand_types],
        result_types=[_norm_type(t) for t in op.result_types],
        regions=[_norm_region(r) for r in op.regions],
    )


def _norm_region(r: Region) -> Region:
    return Region(blocks=[_norm_block(b) for b in r.blocks])


def _norm_block(b: Block) -> Block:
    return Block(
        label=b.label,
        args=[(name, _norm_type(ty)) for name, ty in b.args],
        ops=[_norm_op(o) for o in b.ops],
    )


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_round_trip(name: str) -> None:
    text = _read_fixture(name)
    once = mlir.parse(text)
    twice = mlir.parse(mlir.unparse(once))
    assert [_norm_op(o) for o in once.ops] == [_norm_op(o) for o in twice.ops]


# --------------------------------------------------------------------------- to_generic


def test_to_generic_passes_through_generic_text() -> None:
    text = _read_fixture("p1.ttir.generic")
    assert mlir.to_generic(text) == text


def test_to_generic_rejects_pretty_form_without_triton_opt() -> None:
    with pytest.raises(mlir.NotGeneric):
        mlir.to_generic("module {\n  tt.func @kernel() {\n    tt.return\n  }\n}\n")


def test_block_arguments_may_carry_locations() -> None:
    text = (
        '"builtin.module"() ({\n'
        '  "tt.func"() <{function_type = (i32) -> (), sym_name = "k"}> ({\n'
        '  ^bb0(%arg0: i32 loc("x"(#loc)), %arg1: !tt.ptr<f32> loc(unknown)):\n'
        '    "tt.return"() : () -> ()\n'
        "  }) : () -> ()\n"
        "}) : () -> ()\n"
    )
    module = mlir.parse(text)
    args = module.funcs["k"].regions[0].blocks[0].args
    assert [name for name, _ in args] == ["%arg0", "%arg1"]
    assert args[1][1].kind == "ptr"
