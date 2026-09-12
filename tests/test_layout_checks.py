"""Tests for `layout_checks.py` and `interp2.py`.

The modules here are hand-written generic-form IR, small enough to read. The
gather cases are the real ones of
[triton#11600](https://github.com/triton-lang/triton/issues/11600): the two
layouts are the ones `setOptimizedGatherLayout` picks for a `tl.gather` on
`src 8x2` with `idx 8x8` along axis 1 with two warps, on the 3.8.0 release and
on main. The release computes `sizePerThread[axis]` from the source extent
alone, so the index layout is left with a warp basis on the gather axis and the
lowering's own assertion fires; main takes `max(src, idx)` and the same shapes
come out warp local. The check must separate the two.
"""

from __future__ import annotations

from ttsem import mlir
import pytest
from ttsem.interp2 import LayoutInterp, check_module, launch_config, module_attrs
from ttsem.layout_checks import (
    LayoutViolation,
    checked_op_names,
    is_gather_warp_local,
    shared_offsets_are_a_permutation,
)
from ttsem.layouts import parse_encoding, to_linear_layout
from ttsem.linear_layout import LinearLayout
from ttsem.memory import Memory

# 2 warps, 32 lanes; both cover an 8x8 tensor exactly.
BLOCKED_A = (
    "#ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], "
    "warpsPerCTA = [1, 2], order = [1, 0]}>"
)
BLOCKED_B = (
    "#ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], "
    "warpsPerCTA = [2, 1], order = [1, 0]}>"
)
BLOCKED_1D = (
    "#ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>"
)
# Two lane bases are the same, so after removing broadcast the layout still is
# not injective: an element is held twice and another by nobody.
LINEAR_COLLIDING = (
    "#ttg.linear<{register = [], lane = [[1], [1], [2], [4], [8]], warp = [[16]], block = []}>"
)
# The layout `setOptimizedGatherLayout` picks for src 8x2, idx 8x8, axis 1, 2 warps.
GATHER_RELEASE = (
    "#ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], "
    "warpsPerCTA = [1, 2], order = [1, 0]}>"
)
GATHER_MAIN = (
    "#ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], "
    "warpsPerCTA = [1, 2], order = [1, 0]}>"
)
NVMMA_SHARED = (
    "#ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>"
)


def module(body: str, num_warps: int = 2, threads: int = 32) -> str:
    """A generic-form module with one function holding `body`."""
    return (
        '"builtin.module"() ({\n'
        '  "tt.func"() <{function_type = () -> (), sym_name = "k"}> ({\n'
        f"{body}\n"
        '    "tt.return"() : () -> ()\n'
        "  }) : () -> ()\n"
        f'}}) {{"ttg.num-warps" = {num_warps} : i32, '
        f'"ttg.threads-per-warp" = {threads} : i32}} : () -> ()'
    )


def run_checks(text: str):
    """`check_module` on generic text, with the module's own launch config."""
    num_warps, threads = launch_config(text)
    return check_module(mlir.parse(text), num_warps, threads)


def convert(src_ty: str, dst_ty: str) -> str:
    return f'    %1 = "ttg.convert_layout"(%0) : ({src_ty}) -> {dst_ty}'


def gather(src_enc: str, idx_enc: str, efficient: bool = True) -> str:
    attrs = "axis = 1 : i32" + (", efficient_layout" if efficient else "")
    src = f"tensor<8x2xf32, {src_enc}>"
    idx = f"tensor<8x8xi32, {idx_enc}>"
    out = f"tensor<8x8xf32, {idx_enc}>"
    return f'    %2 = "tt.gather"(%0, %1) <{{{attrs}}}> : ({src}, {idx}) -> {out}'


# --------------------------------------------------------------------------
# convert_layout
# --------------------------------------------------------------------------


def test_convert_layout_between_two_blocked_layouts_is_clean():
    text = module(convert(f"tensor<8x8xf32, {BLOCKED_A}>", f"tensor<8x8xf32, {BLOCKED_B}>"))
    violations, gaps, checked = run_checks(text)
    assert violations == [] and gaps == {}
    assert set(checked) == {"ttg.convert_layout"}


def test_convert_layout_flags_a_layout_that_is_not_a_bijection():
    text = module(convert(f"tensor<32xi32, {LINEAR_COLLIDING}>", f"tensor<32xi32, {BLOCKED_1D}>"))
    violations, gaps, _ = run_checks(text)
    assert gaps == {}
    assert len(violations) == 1
    assert violations[0].op_name == "ttg.convert_layout"
    assert "bijection" in violations[0].reason


def test_convert_layout_flags_disagreeing_shapes():
    text = module(convert(f"tensor<8x8xf32, {BLOCKED_A}>", f"tensor<8x4xf32, {BLOCKED_A}>"))
    violations, _, _ = run_checks(text)
    assert len(violations) == 1
    assert "logical shapes differ" in violations[0].reason


def test_an_encoding_the_port_cannot_build_is_a_gap_not_a_pass():
    mfma = "#ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 2], instrShape = [32, 32]}>"
    text = module(convert(f"tensor<32x32xf32, {mfma}>", f"tensor<32x32xf32, {mfma}>"))
    violations, gaps, checked = run_checks(text)
    assert violations == []
    assert list(gaps) == [mfma]
    assert "amd_mfma" in gaps[mfma]
    assert set(checked) == {"ttg.convert_layout"}


# --------------------------------------------------------------------------
# tt.gather and triton#11600
# --------------------------------------------------------------------------


def test_gather_is_warp_local_with_the_layout_main_picks():
    """main computes sizePerThread[axis] from max(src, idx), so the columns fit."""
    violations, gaps, checked = run_checks(module(gather(GATHER_MAIN, GATHER_MAIN)))
    assert violations == [] and gaps == {}
    assert set(checked) == {"tt.gather"}


def test_gather_of_issue_11600_is_flagged():
    """The 3.8.0 layout leaves the index warp basis on the gather axis."""
    violations, gaps, _ = run_checks(module(gather(GATHER_RELEASE, GATHER_RELEASE)))
    assert gaps == {}
    assert len(violations) == 1
    assert violations[0].op_name == "tt.gather"
    assert "not warp local" in violations[0].reason
    assert "index layout moves dim1 with the warp" in violations[0].reason


def test_gather_without_efficient_layout_is_not_checked():
    """Before the layout is chosen the lowering may still fall back to shared memory."""
    violations, _, checked = run_checks(
        module(gather(GATHER_RELEASE, GATHER_RELEASE, efficient=False))
    )
    assert violations == []
    assert set(checked) == {"tt.gather"}


def test_warp_local_predicate_reports_each_failing_condition():
    src = to_linear_layout(parse_encoding(GATHER_RELEASE), [8, 2])
    idx = to_linear_layout(parse_encoding(GATHER_RELEASE), [8, 8])
    assert is_gather_warp_local(src, src, 1, 2) is None
    assert "index layout" in is_gather_warp_local(src, idx, 1, 2)
    assert "source layout" in is_gather_warp_local(idx, idx, 1, 2)


# --------------------------------------------------------------------------
# Shared memory round trip
# --------------------------------------------------------------------------


def test_shared_round_trip_through_an_nvmma_swizzle_is_the_identity():
    memdesc = f"!ttg.memdesc<16x64xf16, {NVMMA_SHARED}, #ttg.shared_memory, mutable>"
    body = (
        f'    %1 = "ttg.local_load"(%0) : ({memdesc}) -> tensor<16x64xf16, {BLOCKED_A}>\n'
        f'    "ttg.local_store"(%1, %0) : (tensor<16x64xf16, {BLOCKED_A}>, {memdesc}) -> ()'
    )
    violations, gaps, checked = run_checks(module(body))
    assert violations == [] and gaps == {}
    assert set(checked) == {"ttg.local_load", "ttg.local_store"}


def test_a_padded_shared_layout_is_still_a_round_trip():
    """`fp4Padded` puts a padded offset beside every real one, evenly."""
    padded = parse_encoding(
        "#ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, "
        "elementBitWidth = 8, fp4Padded = true}>"
    )
    layout = to_linear_layout(padded, [128, 64])
    assert layout.total_in_dim_size() == 2 * layout.total_out_dim_size()
    assert shared_offsets_are_a_permutation(layout)


def test_a_shared_layout_that_starves_an_element_is_rejected():
    starved = LinearLayout({"offset": [[0, 1], [1, 0], [0, 1]]}, [("dim0", 4), ("dim1", 2)], False)
    assert not shared_offsets_are_a_permutation(starved)


# --------------------------------------------------------------------------
# Slice encodings
# --------------------------------------------------------------------------


REDUCE_REGION = (
    '({\n    ^bb0(%x: f32, %y: f32):\n      "tt.reduce.return"(%x) : (f32) -> ()\n    })'
)


def reduce_op(result_ty: str, axis: int = 1) -> str:
    return (
        f'    %1 = "tt.reduce"(%0) <{{axis = {axis} : i32}}> {REDUCE_REGION} : '
        f"(tensor<8x8xf32, {BLOCKED_A}>) -> {result_ty}"
    )


def test_reduce_result_is_the_slice_of_the_operand():
    sliced = f"tensor<8xf32, #ttg.slice<{{dim = 1, parent = {BLOCKED_A}}}>>"
    violations, gaps, checked = run_checks(module(reduce_op(sliced)))
    assert violations == [] and gaps == {}
    assert set(checked) == {"tt.reduce"}


def test_reduce_slicing_the_wrong_dim_is_flagged():
    sliced = f"tensor<8xf32, #ttg.slice<{{dim = 0, parent = {BLOCKED_A}}}>>"
    violations, _, _ = run_checks(module(reduce_op(sliced)))
    assert len(violations) == 1
    assert "slices dim 0, not the axis 1" in violations[0].reason


def test_reduce_onto_a_foreign_layout_is_flagged():
    sliced = f"tensor<8xf32, #ttg.slice<{{dim = 1, parent = {BLOCKED_B}}}>>"
    violations, _, _ = run_checks(module(reduce_op(sliced)))
    assert len(violations) == 1
    assert "slices a different layout" in violations[0].reason


def expand_op(operand_ty: str, axis: int = 1) -> str:
    return (
        f'    %1 = "tt.expand_dims"(%0) <{{axis = {axis} : i32}}> : '
        f"({operand_ty}) -> tensor<8x1xf32, {BLOCKED_A}>"
    )


def test_expand_dims_operand_is_the_slice_of_the_result():
    sliced = f"tensor<8xf32, #ttg.slice<{{dim = 1, parent = {BLOCKED_A}}}>>"
    violations, gaps, checked = run_checks(module(expand_op(sliced)))
    assert violations == [] and gaps == {}
    assert set(checked) == {"tt.expand_dims"}


def test_expand_dims_of_a_non_slice_is_flagged():
    violations, _, _ = run_checks(module(expand_op(f"tensor<8xf32, {BLOCKED_1D}>")))
    assert len(violations) == 1
    assert "not a slice encoding" in violations[0].reason


# --------------------------------------------------------------------------
# The interpreter
# --------------------------------------------------------------------------


def make_range_module(dst_encoding: str) -> str:
    body = (
        f'    %0 = "tt.make_range"() <{{end = 32 : i32, start = 0 : i32}}> : '
        f"() -> tensor<32xi32, {BLOCKED_1D}>\n"
        f'    %1 = "ttg.convert_layout"(%0) : (tensor<32xi32, {BLOCKED_1D}>) -> '
        f"tensor<32xi32, {dst_encoding}>"
    )
    return module(body)


def test_layout_interp_runs_level_one_and_checks_as_it_goes():
    text = make_range_module(BLOCKED_1D)
    num_warps, threads = launch_config(text)
    interp = LayoutInterp(mlir.parse(text), Memory(), num_warps=num_warps, threads_per_warp=threads)
    interp.run("k", [])
    assert set(interp.checked) == {"ttg.convert_layout"}
    assert interp.layout_gaps == {}
    assert interp.unsupported == set()


def test_layout_interp_raises_on_the_first_violation():
    text = make_range_module(LINEAR_COLLIDING)
    num_warps, threads = launch_config(text)
    interp = LayoutInterp(mlir.parse(text), Memory(), num_warps=num_warps, threads_per_warp=threads)
    with pytest.raises(LayoutViolation) as excinfo:
        interp.run("k", [])
    assert excinfo.value.op_name == "ttg.convert_layout"


def test_launch_config_reads_the_module_attributes():
    text = module("", num_warps=8, threads=64)
    assert launch_config(text) == (8, 64)
    assert module_attrs(text)["ttg.num-warps"] == 8
    assert launch_config('"builtin.module"() ({}) : () -> ()'.replace("({})", "({\n^bb0:\n})")) == (
        None,
        None,
    )


def test_the_checked_op_names_are_the_documented_four_families():
    assert checked_op_names() == [
        "tt.expand_dims",
        "tt.gather",
        "tt.reduce",
        "ttg.convert_layout",
        "ttg.local_load",
        "ttg.local_store",
    ]
