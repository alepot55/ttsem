"""Tests for `linear_layout.py` and `layouts.py`.

Every expected layout here comes from a Triton source of truth, never from
running our own code:

- most cases are transcribed from the C++ unit tests in
  `unittest/Dialect/TritonGPU/LinearLayoutConversionsTest.cpp`, one test
  function per `TEST_F`, keeping the C++ test name in the docstring;
- three cases come from lit tests, where Triton itself asserts that a
  `#ttg.linear<...>` and a legacy encoding describe the same tensor
  (`test/TritonGPU/canonicalize.mlir`);
- the algebra tests use the worked examples written in the comments of
  `include/triton/Tools/LinearLayout.h`.
"""

from __future__ import annotations

from pathlib import Path

from ttsem import mlir
import pytest
from ttsem.layouts import (
    Blocked,
    DotOperand,
    EncodingParseError,
    Linear,
    NvidiaMma,
    NvmmaShared,
    Slice,
    SwizzledShared,
    cga_from_split_params,
    cga_one_cta,
    drop_pipelining_dims,
    parse_encoding,
    to_linear_layout,
)
from ttsem.linear_layout import LayoutError, LinearLayout

# --------------------------------------------------------------------------
# Fixtures mirroring the helpers of LinearLayoutConversionsTest
# --------------------------------------------------------------------------


def blocked(spt, tpw, wpb, cpg, c_split, order, c_order) -> Blocked:
    return Blocked(
        tuple(spt),
        tuple(tpw),
        tuple(wpb),
        tuple(order),
        cga_from_split_params(cpg, c_split, c_order),
    )


def mma(v_major, v_minor, instr_shape, warps, cpg=None, c_split=None, c_order=None) -> NvidiaMma:
    rank = len(warps)
    cga = cga_one_cta(rank) if cpg is None else cga_from_split_params(cpg, c_split, c_order)
    return NvidiaMma(v_major, v_minor, tuple(warps), tuple(instr_shape), cga)


def dot(parent, op_idx, k_width) -> DotOperand:
    return DotOperand(op_idx, parent, k_width)


def shared(vec, per_phase, max_phase, cpg, c_split, order, c_order) -> SwizzledShared:
    return SwizzledShared(
        vec, per_phase, max_phase, tuple(order), cga_from_split_params(cpg, c_split, c_order)
    )


def nvmma(swizzle, transposed, elem_bits, cpg, c_split, c_order, fp4_padded=False) -> NvmmaShared:
    return NvmmaShared(
        swizzle, transposed, elem_bits, fp4_padded, cga_from_split_params(cpg, c_split, c_order)
    )


def ll(bases, out_dims, require_surjective=True) -> LinearLayout:
    return LinearLayout(bases, out_dims, require_surjective)


DIM0 = ["dim0"]
DIM01 = ["dim0", "dim1"]
DIM012 = ["dim0", "dim1", "dim2"]
DIM0123 = ["dim0", "dim1", "dim2", "dim3"]


# --------------------------------------------------------------------------
# LinearLayout algebra, from the examples in LinearLayout.h
# --------------------------------------------------------------------------


def test_apply_swizzle_example():
    """The 4x4 swizzle worked through in the header: L(t, w) = (t, w ^ t)."""
    layout = ll({"t": [[1, 1], [2, 2]], "w": [[0, 1], [0, 2]]}, ["x", "y"])
    assert layout.apply({"t": 0, "w": 0}) == {"x": 0, "y": 0}
    assert layout.apply({"t": 0, "w": 3}) == {"x": 0, "y": 3}
    assert layout.apply({"t": 3, "w": 0}) == {"x": 3, "y": 3}
    assert layout.apply({"t": 3, "w": 3}) == {"x": 3, "y": 0}


def test_product_disjoint_dims():
    """identity1D(4, i1, o1) * identity1D(8, i2, o2) is the 2D identity."""
    product = LinearLayout.identity1D(4, "i1", "o1") * LinearLayout.identity1D(8, "i2", "o2")
    assert product == ll({"i1": [[1, 0], [2, 0]], "i2": [[0, 1], [0, 2], [0, 4]]}, ["o1", "o2"])


def test_product_shared_dims():
    """The four shared-dim examples spelled out above `operator*`."""
    identity = LinearLayout.identity1D
    zeros = LinearLayout.zeros1D
    assert identity(4, "i", "o") * identity(2, "i", "o") == identity(8, "i", "o")
    mod4 = identity(4, "i", "o") * zeros(2, "i", "o")
    assert [mod4.apply({"i": x})["o"] for x in range(8)] == [x % 4 for x in range(8)]
    div2 = zeros(2, "i", "o") * identity(4, "i", "o")
    assert [div2.apply({"i": x})["o"] for x in range(8)] == [x // 2 for x in range(8)]
    split = identity(4, "i", "o1") * identity(8, "i", "o2")
    assert [tuple(split.apply({"i": x}).values()) for x in range(32)] == [
        (x % 4, x // 4) for x in range(32)
    ]


def test_num_consecutive_in_out():
    assert LinearLayout.identity1D(8, "i", "o").num_consecutive_in_out() == 8
    permuted = ll({"i": [[4], [8], [1], [2]]}, ["o"])
    assert permuted.num_consecutive_in_out() == 1
    partial = ll({"i": [[1], [2], [8], [4]]}, ["o"])
    assert partial.num_consecutive_in_out() == 4


def test_invert_and_pseudoinvert():
    layout = ll({"i": [[4], [8], [1], [2]]}, ["o"])
    assert layout.is_invertible()
    inverse = layout.invert()
    for x in range(16):
        assert inverse.apply({"o": layout.apply({"i": x})["o"]}) == {"i": x}
    broadcast = LinearLayout.identity1D(4, "i", "o") * LinearLayout.zeros1D(2, "i", "o")
    assert not broadcast.is_invertible()
    pseudo = broadcast.pseudoinvert()
    for y in range(4):
        assert broadcast.apply(pseudo.apply({"o": y})) == {"o": y}


def test_invert_and_compose_is_a_change_of_basis():
    src = LinearLayout.identity1D(8, "register", "dim0")
    dst = ll({"register": [[4], [1], [2]]}, ["dim0"])
    cvt = src.invert_and_compose(dst)
    for x in range(8):
        assert dst.apply(cvt.apply({"register": x})) == src.apply({"register": x})


def test_compose_chains_two_layouts():
    """`a.compose(b)` is b after a, so it renames the output of a through b."""
    a = ll({"register": [[1], [2]]}, ["mid"])
    b = ll({"mid": [[2], [1]]}, ["dim0"])
    chained = a.compose(b)
    for x in range(4):
        assert chained.apply({"register": x}) == b.apply(a.apply({"register": x}))
    assert chained.out_dim_names() == ["dim0"]


def test_sublayout_and_zero():
    layout = ll({"a": [[1, 0], [2, 0]], "b": [[0, 1], [0, 2]]}, ["x", "y"])
    assert layout.sublayout(["a"], ["x"]) == ll({"a": [[1], [2]]}, [("x", 4)], False)
    assert layout.sublayout_is_zero(["a"], ["y"])
    assert not layout.sublayout_is_zero(["a"], ["x"])


def test_reshape_outs_and_free_variables():
    flat = LinearLayout.identity1D(16, "i", "o")
    reshaped = flat.reshape_outs([("o0", 4), ("o1", 4)])
    assert reshaped.apply({"i": 6}) == {"o0": 2, "o1": 1}
    broadcast = LinearLayout.identity1D(4, "i", "o") * LinearLayout.zeros1D(4, "i", "o")
    assert broadcast.free_variable_masks() == {"i": 0b1100}


def test_non_surjective_layout_is_rejected():
    with pytest.raises(LayoutError):
        ll({"i": [[1], [4]]}, ["o"])
    assert ll({"i": [[1], [4]]}, [("o", 8)], False).is_surjective() is False


# --------------------------------------------------------------------------
# Blocked layouts
# --------------------------------------------------------------------------


def test_simple_blocked():
    """C++ SimpleBlocked."""
    layout = to_linear_layout(blocked([1], [4], [4], [1], [1], [0], [0]), [16])
    assert layout == ll({"register": [], "lane": [[1], [2]], "warp": [[4], [8]], "block": []}, DIM0)


def test_cta_duplication():
    """C++ CTADuplication."""
    layout = to_linear_layout(blocked([1], [4], [4], [4], [2], [0], [0]), [32])
    assert layout == ll(
        {"register": [], "lane": [[1], [2]], "warp": [[4], [8]], "block": [[16], [0]]}, DIM0
    )


def test_cta_broadcast():
    """C++ CTABroadcast."""
    layout = to_linear_layout(
        blocked([8, 1], [8, 4], [1, 4], [1, 2], [1, 2], [0, 1], [1, 0]), [64, 128]
    )
    assert layout == ll(
        {
            "register": [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]],
            "lane": [[8, 0], [16, 0], [32, 0], [0, 1], [0, 2]],
            "warp": [[0, 4], [0, 8]],
            "block": [[0, 64]],
        },
        DIM01,
    )


def test_shape_larger_than_layout():
    """C++ ShapeLargerThanLayout."""
    layout = to_linear_layout(blocked([1], [4], [4], [1], [1], [0], [0]), [128])
    assert layout == ll(
        {"register": [[16], [32], [64]], "lane": [[1], [2]], "warp": [[4], [8]], "block": []},
        DIM0,
    )


def test_shape_larger_than_layout_2d_degenerate():
    """C++ ShapeLargerThanLayout2DDegenerate."""
    layout = to_linear_layout(
        blocked([1, 1], [4, 1], [4, 1], [1, 1], [1, 1], [0, 1], [1, 0]), [128, 1]
    )
    assert layout == ll(
        {
            "register": [[16, 0], [32, 0], [64, 0]],
            "lane": [[1, 0], [2, 0]],
            "warp": [[4, 0], [8, 0]],
            "block": [],
        },
        DIM01,
    )


def test_shape_smaller_than_layout():
    """C++ ShapeSmallerThanLayout."""
    layout = to_linear_layout(blocked([4], [4], [4], [1], [1], [0], [0]), [8])
    assert layout == ll(
        {"register": [[1], [2]], "lane": [[4], [0]], "warp": [[0], [0]], "block": []}, DIM0
    )


def test_reversed_order():
    """C++ ReversedOrder."""
    layout = to_linear_layout(
        blocked([1, 1], [32, 1], [1, 8], [1, 1], [1, 1], [0, 1], [1, 0]), [1, 64]
    )
    assert layout == ll(
        {
            "register": [[0, 8], [0, 16], [0, 32]],
            "lane": [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            "warp": [[0, 1], [0, 2], [0, 4]],
            "block": [],
        },
        DIM01,
    )


def test_replicate_in_register_dim():
    """C++ ReplicateInRegisterDim."""
    layout = to_linear_layout(blocked([2], [4], [1], [1], [1], [0], [0]), [32])
    assert layout == ll(
        {"register": [[1], [8], [16]], "lane": [[2], [4]], "warp": [], "block": []}, DIM0
    )


def test_one_dim_too_large_another_too_small():
    """C++ OneDimTooLargeAnotherTooSmall."""
    layout = to_linear_layout(
        blocked([1, 4], [8, 4], [4, 1], [2, 2], [2, 1], [1, 0], [1, 0]), [128, 16]
    )
    assert layout == ll(
        {
            "register": [[0, 1], [0, 2], [32, 0]],
            "lane": [[0, 4], [0, 8], [1, 0], [2, 0], [4, 0]],
            "warp": [[8, 0], [16, 0]],
            "block": [[0, 0], [64, 0]],
        },
        DIM01,
    )


def test_repeat_in_ctg_dim_first():
    """C++ RepeatInCTGDimFirst: elements go to distinct CTAs before repeating."""
    layout = to_linear_layout(blocked([1], [1], [4], [2], [2], [0], [0]), [4])
    assert layout == ll({"register": [], "lane": [], "warp": [[1], [0]], "block": [[2]]}, DIM0)


def test_smaller_than_cga_layout():
    """C++ SmallerThanCGALayout."""
    layout = to_linear_layout(blocked([1], [1], [1], [4], [4], [0], [0]), [2])
    assert layout == ll({"register": [], "lane": [], "warp": [], "block": [[1], [0]]}, DIM0)


def test_skinny():
    """C++ Skinny."""
    layout = to_linear_layout(
        blocked([8, 1], [8, 4], [1, 4], [1, 2], [1, 2], [0, 1], [0, 1]), [64, 1]
    )
    assert layout == ll(
        {
            "register": [[1, 0], [2, 0], [4, 0]],
            "lane": [[8, 0], [16, 0], [32, 0], [0, 0], [0, 0]],
            "warp": [[0, 0], [0, 0]],
            "block": [[0, 0]],
        },
        DIM01,
    )


def test_blocked_order():
    """C++ BlockedOrder."""
    layout = to_linear_layout(
        blocked([2, 2], [4, 8], [2, 2], [2, 2], [2, 2], [1, 0], [1, 0]), [1024, 128]
    )
    assert layout == ll(
        {
            "register": [
                [0, 1],
                [1, 0],
                [0, 32],
                [16, 0],
                [32, 0],
                [64, 0],
                [128, 0],
                [256, 0],
            ],
            "lane": [[0, 2], [0, 4], [0, 8], [2, 0], [4, 0]],
            "warp": [[0, 16], [8, 0]],
            "block": [[0, 64], [512, 0]],
        },
        DIM01,
    )


def test_blocked_4d():
    """C++ Blocked4D."""
    layout = to_linear_layout(
        blocked(
            [1, 1, 1, 4],
            [2, 1, 1, 16],
            [1, 2, 4, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [3, 0, 1, 2],
            [3, 2, 1, 0],
        ),
        [2, 1, 1, 1],
    )
    assert layout == ll(
        {
            "register": [],
            "lane": [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
            ],
            "warp": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "block": [],
        },
        DIM0123,
    )


# --------------------------------------------------------------------------
# Dot operands over a blocked parent (the FMA path)
# --------------------------------------------------------------------------


BLOCKED_DOT_PARENT = blocked([2, 4], [8, 4], [2, 4], [1, 1], [1, 1], [1, 0], [1, 0])
BLOCKED_DOT_PARENT_3D = blocked(
    [2, 2, 4], [2, 4, 4], [2, 2, 2], [1, 1, 1], [1, 1, 1], [2, 1, 0], [2, 1, 0]
)


def test_blocked_dot_operand_lhs():
    """C++ BlockedDotOperandLhs."""
    layout = to_linear_layout(dot(BLOCKED_DOT_PARENT, 0, 0), [32, 16])
    assert layout == ll(
        {
            "register": [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0]],
            "lane": [[0, 0], [0, 0], [2, 0], [4, 0], [8, 0]],
            "warp": [[0, 0], [0, 0], [16, 0]],
            "block": [],
        },
        DIM01,
    )


def test_blocked_dot3d_operand_lhs():
    """C++ BlockedDot3dOperandLhs."""
    layout = to_linear_layout(dot(BLOCKED_DOT_PARENT_3D, 0, 0), [16, 32, 4])
    assert layout == ll(
        {
            "register": [[0, 0, 1], [0, 0, 2], [0, 1, 0], [1, 0, 0], [0, 16, 0], [8, 0, 0]],
            "lane": [[0, 0, 0], [0, 0, 0], [0, 2, 0], [0, 4, 0], [2, 0, 0]],
            "warp": [[0, 0, 0], [0, 8, 0], [4, 0, 0]],
            "block": [],
        },
        DIM012,
    )


def test_blocked_dot_operand_rhs():
    """C++ BlockedDotOperandRhs."""
    layout = to_linear_layout(dot(BLOCKED_DOT_PARENT, 1, 0), [16, 64])
    assert layout == ll(
        {
            "register": [[0, 1], [0, 2], [1, 0], [2, 0], [4, 0], [8, 0]],
            "lane": [[0, 4], [0, 8], [0, 0], [0, 0], [0, 0]],
            "warp": [[0, 16], [0, 32], [0, 0]],
            "block": [],
        },
        DIM01,
    )


def test_blocked_dot3d_operand_rhs():
    """C++ BlockedDot3dOperandRhs."""
    layout = to_linear_layout(dot(BLOCKED_DOT_PARENT_3D, 1, 0), [16, 4, 64])
    assert layout == ll(
        {
            "register": [
                [0, 0, 1],
                [0, 0, 2],
                [0, 1, 0],
                [0, 2, 0],
                [1, 0, 0],
                [0, 0, 32],
                [8, 0, 0],
            ],
            "lane": [[0, 0, 4], [0, 0, 8], [0, 0, 0], [0, 0, 0], [2, 0, 0]],
            "warp": [[0, 0, 16], [0, 0, 0], [4, 0, 0]],
            "block": [],
        },
        DIM012,
    )


# --------------------------------------------------------------------------
# nvidia_mma v2
# --------------------------------------------------------------------------


MMAV2_LANE = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]]


def test_mmav2_16x16():
    """C++ MMAv2_16x16."""
    layout = to_linear_layout(mma(2, 0, [16, 8], [1, 1], [1, 1], [1, 1], [0, 1]), [16, 16])
    assert layout == ll(
        {
            "register": [[0, 1], [8, 0], [0, 8]],
            "lane": MMAV2_LANE,
            "warp": [],
            "block": [],
        },
        DIM01,
    )


def test_mmav2_32x32():
    """C++ MMAv2_32x32."""
    layout = to_linear_layout(mma(2, 0, [16, 8], [1, 1], [1, 1], [1, 1], [0, 1]), [32, 32])
    assert layout == ll(
        {
            "register": [[0, 1], [8, 0], [0, 8], [0, 16], [16, 0]],
            "lane": MMAV2_LANE,
            "warp": [],
            "block": [],
        },
        DIM01,
    )


def test_mmav2_extend_dim2():
    """C++ MMAv2_ExtendDim2."""
    layout = to_linear_layout(mma(2, 0, [16, 8], [1, 1], [1, 1], [1, 1], [0, 1]), [16, 128])
    assert layout == ll(
        {
            "register": [[0, 1], [8, 0], [0, 8], [0, 16], [0, 32], [0, 64]],
            "lane": MMAV2_LANE,
            "warp": [],
            "block": [],
        },
        DIM01,
    )


def test_mmav2_cga():
    """C++ MMAv2_Cga."""
    layout = to_linear_layout(
        mma(2, 0, [1, 16, 8], [16, 1, 1], [4, 2, 2], [4, 2, 1], [2, 1, 0]), [64, 128, 128]
    )
    assert layout == ll(
        {
            "register": [
                [0, 0, 1],
                [0, 8, 0],
                [0, 0, 8],
                [0, 0, 16],
                [0, 0, 32],
                [0, 0, 64],
                [0, 16, 0],
                [0, 32, 0],
            ],
            "lane": [[0, 0, 2], [0, 0, 4], [0, 1, 0], [0, 2, 0], [0, 4, 0]],
            "warp": [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0]],
            "block": [[0, 0, 0], [0, 64, 0], [16, 0, 0], [32, 0, 0]],
        },
        DIM012,
    )


def test_mmav2_small_3d():
    """C++ MMAv2_Small3D."""
    layout = to_linear_layout(
        mma(2, 0, [1, 16, 8], [16, 1, 1], [4, 2, 2], [4, 2, 1], [2, 1, 0]), [1, 128, 128]
    )
    assert layout == ll(
        {
            "register": [
                [0, 0, 1],
                [0, 8, 0],
                [0, 0, 8],
                [0, 0, 16],
                [0, 0, 32],
                [0, 0, 64],
                [0, 16, 0],
                [0, 32, 0],
            ],
            "lane": [[0, 0, 2], [0, 0, 4], [0, 1, 0], [0, 2, 0], [0, 4, 0]],
            "warp": [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "block": [[0, 0, 0], [0, 64, 0], [0, 0, 0], [0, 0, 0]],
        },
        DIM012,
    )


# --------------------------------------------------------------------------
# nvidia_mma v3
# --------------------------------------------------------------------------


@pytest.mark.parametrize("instr_shape", [[16, 16, 8], [16, 8, 8]])
def test_mmav3_64x16(instr_shape):
    """C++ MMAv3_64x16, both instruction shapes."""
    layout = to_linear_layout(mma(3, 0, instr_shape, [4, 1], [1, 1], [1, 1], [1, 0]), [64, 16])
    assert layout == ll(
        {
            "register": [[0, 1], [8, 0], [0, 8]],
            "lane": MMAV2_LANE,
            "warp": [[16, 0], [32, 0]],
            "block": [],
        },
        DIM01,
    )


def test_mmav3_128x16():
    """C++ MMAv3_128x16."""
    layout = to_linear_layout(mma(3, 0, [16, 16, 8], [4, 1], [1, 1], [1, 1], [1, 0]), [128, 16])
    assert layout == ll(
        {
            "register": [[0, 1], [8, 0], [0, 8], [64, 0]],
            "lane": MMAV2_LANE,
            "warp": [[16, 0], [32, 0]],
            "block": [],
        },
        DIM01,
    )


def test_mmav3_1024x1024():
    """C++ MMAv3_1024x1024."""
    layout = to_linear_layout(mma(3, 0, [16, 16, 8], [4, 1], [1, 1], [1, 1], [1, 0]), [1024, 1024])
    assert layout == ll(
        {
            "register": [
                [0, 1],
                [8, 0],
                [0, 8],
                [0, 16],
                [0, 32],
                [0, 64],
                [0, 128],
                [0, 256],
                [0, 512],
                [64, 0],
                [128, 0],
                [256, 0],
                [512, 0],
            ],
            "lane": MMAV2_LANE,
            "warp": [[16, 0], [32, 0]],
            "block": [],
        },
        DIM01,
    )


@pytest.mark.parametrize(
    "shape,register,warp",
    [
        ([64, 32], [[0, 1], [8, 0], [0, 8], [0, 16]], [[16, 0], [32, 0], [0, 0]]),
        ([64, 64], [[0, 1], [8, 0], [0, 8], [0, 16]], [[16, 0], [32, 0], [0, 32]]),
        ([128, 64], [[0, 1], [8, 0], [0, 8], [0, 16], [64, 0]], [[16, 0], [32, 0], [0, 32]]),
        (
            [256, 64],
            [[0, 1], [8, 0], [0, 8], [0, 16], [64, 0], [128, 0]],
            [[16, 0], [32, 0], [0, 32]],
        ),
    ],
)
def test_mmav3_4x2_warps(shape, register, warp):
    """C++ MMAv3_4x2Warps."""
    legacy = mma(3, 0, [16, 32, 16], [4, 2], [1, 1], [1, 1], [1, 0])
    layout = to_linear_layout(legacy, shape)
    assert layout == ll(
        {"register": register, "lane": MMAV2_LANE, "warp": warp, "block": []}, DIM01
    )


@pytest.mark.parametrize(
    "shape,register,warp",
    [
        ([16, 16], [[0, 1], [8, 0], [0, 8]], [[0, 0], [0, 0], [0, 0], [0, 0]]),
        ([32, 16], [[0, 1], [8, 0], [0, 8]], [[16, 0], [0, 0], [0, 0], [0, 0]]),
        ([64, 16], [[0, 1], [8, 0], [0, 8]], [[16, 0], [32, 0], [0, 0], [0, 0]]),
        ([128, 16], [[0, 1], [8, 0], [0, 8], [64, 0]], [[16, 0], [32, 0], [0, 0], [0, 0]]),
        ([32, 32], [[0, 1], [8, 0], [0, 8]], [[16, 0], [0, 0], [0, 16], [0, 0]]),
        ([64, 32], [[0, 1], [8, 0], [0, 8]], [[16, 0], [32, 0], [0, 16], [0, 0]]),
    ],
)
def test_mmav3_4x4_warps(shape, register, warp):
    """C++ MMAv3_4x4Warps."""
    legacy = mma(3, 0, [16, 16, 8], [4, 4], [1, 1], [1, 1], [1, 0])
    layout = to_linear_layout(legacy, shape)
    assert layout == ll(
        {"register": register, "lane": MMAV2_LANE, "warp": warp, "block": []}, DIM01
    )


# --------------------------------------------------------------------------
# Dot operands over an mma parent
# --------------------------------------------------------------------------


DOT_A_LANE = [[0, 8], [0, 16], [1, 0], [2, 0], [4, 0]]
DOT_B_LANE = [[8, 0], [16, 0], [0, 1], [0, 2], [0, 4]]


def test_dot_mmav2_tile_kwidth8():
    """C++ DotMMAv2_tile_kwidth8."""
    parent = mma(2, 0, [16, 8], [1, 1])
    assert to_linear_layout(dot(parent, 0, 8), [16, 64]) == ll(
        {
            "register": [[0, 1], [0, 2], [0, 4], [8, 0], [0, 32]],
            "lane": DOT_A_LANE,
            "warp": [],
            "block": [],
        },
        DIM01,
    )
    assert to_linear_layout(dot(parent, 1, 8), [64, 8]) == ll(
        {
            "register": [[1, 0], [2, 0], [4, 0], [32, 0]],
            "lane": DOT_B_LANE,
            "warp": [],
            "block": [],
        },
        DIM01,
    )


def test_dot_mmav2_large_warp4_kwidth8():
    """C++ DotMMAv2_large_warp4_kwidth8."""
    parent = mma(2, 0, [16, 8], [4, 1])
    assert to_linear_layout(dot(parent, 0, 8), [128, 128]) == ll(
        {
            "register": [[0, 1], [0, 2], [0, 4], [8, 0], [0, 32], [0, 64], [64, 0]],
            "lane": DOT_A_LANE,
            "warp": [[16, 0], [32, 0]],
            "block": [],
        },
        DIM01,
    )
    assert to_linear_layout(dot(parent, 1, 8), [128, 64]) == ll(
        {
            "register": [[1, 0], [2, 0], [4, 0], [32, 0], [64, 0], [0, 8], [0, 16], [0, 32]],
            "lane": DOT_B_LANE,
            "warp": [[0, 0], [0, 0]],
            "block": [],
        },
        DIM01,
    )
    assert to_linear_layout(dot(parent, 1, 8), [64, 128]) == ll(
        {
            "register": [[1, 0], [2, 0], [4, 0], [32, 0], [0, 8], [0, 16], [0, 32], [0, 64]],
            "lane": DOT_B_LANE,
            "warp": [[0, 0], [0, 0]],
            "block": [],
        },
        DIM01,
    )


def test_dot_mmav2_3d():
    """C++ DotMMAv2_3D."""
    parent = mma(2, 0, [1, 16, 8], [2, 4, 2])
    assert to_linear_layout(dot(parent, 0, 8), [16, 128, 128]) == ll(
        {
            "register": [
                [0, 0, 1],
                [0, 0, 2],
                [0, 0, 4],
                [0, 8, 0],
                [0, 0, 32],
                [0, 0, 64],
                [0, 64, 0],
                [2, 0, 0],
                [4, 0, 0],
                [8, 0, 0],
            ],
            "lane": [[0, 0, 8], [0, 0, 16], [0, 1, 0], [0, 2, 0], [0, 4, 0]],
            "warp": [[0, 0, 0], [0, 16, 0], [0, 32, 0], [1, 0, 0]],
            "block": [],
        },
        DIM012,
    )
    assert to_linear_layout(dot(parent, 1, 8), [8, 128, 64]) == ll(
        {
            "register": [
                [0, 1, 0],
                [0, 2, 0],
                [0, 4, 0],
                [0, 32, 0],
                [0, 64, 0],
                [0, 0, 16],
                [0, 0, 32],
                [2, 0, 0],
                [4, 0, 0],
            ],
            "lane": [[0, 8, 0], [0, 16, 0], [0, 0, 1], [0, 0, 2], [0, 0, 4]],
            "warp": [[0, 0, 8], [0, 0, 0], [0, 0, 0], [1, 0, 0]],
            "block": [],
        },
        DIM012,
    )


def test_dot_mmav3_warp4_kwidth2():
    """C++ DotMMAv3_warp4_kwidth2."""
    dot_op = dot(mma(3, 0, [16, 16, 8], [4, 1]), 0, 2)
    assert to_linear_layout(dot_op, [64, 16]) == ll(
        {
            "register": [[0, 1], [8, 0], [0, 8]],
            "lane": MMAV2_LANE,
            "warp": [[16, 0], [32, 0]],
            "block": [],
        },
        DIM01,
    )
    assert to_linear_layout(dot_op, [128, 16]) == ll(
        {
            "register": [[0, 1], [8, 0], [0, 8], [64, 0]],
            "lane": MMAV2_LANE,
            "warp": [[16, 0], [32, 0]],
            "block": [],
        },
        DIM01,
    )
    assert to_linear_layout(dot_op, [128, 32]) == ll(
        {
            "register": [[0, 1], [8, 0], [0, 8], [0, 16], [64, 0]],
            "lane": MMAV2_LANE,
            "warp": [[16, 0], [32, 0]],
            "block": [],
        },
        DIM01,
    )


def test_dot_mmav3_mixed_warp_kwidth4():
    """C++ DotMMAv3_mixed_warp_kwidth4."""
    dot_op = dot(mma(3, 0, [16, 16, 8], [4, 2]), 0, 4)
    assert to_linear_layout(dot_op, [128, 64]) == ll(
        {
            "register": [[0, 1], [0, 2], [8, 0], [0, 16], [0, 32], [64, 0]],
            "lane": [[0, 4], [0, 8], [1, 0], [2, 0], [4, 0]],
            "warp": [[16, 0], [32, 0], [0, 0]],
            "block": [],
        },
        DIM01,
    )


def test_dot_mmav2_split_warp_kwidth8():
    """C++ DotMMAv2_split_warp_kwidth8."""
    parent = mma(2, 0, [16, 8], [2, 2])
    assert to_linear_layout(dot(parent, 0, 8), [32, 64]) == ll(
        {
            "register": [[0, 1], [0, 2], [0, 4], [8, 0], [0, 32]],
            "lane": DOT_A_LANE,
            "warp": [[0, 0], [16, 0]],
            "block": [],
        },
        DIM01,
    )
    assert to_linear_layout(dot(parent, 1, 8), [64, 16]) == ll(
        {
            "register": [[1, 0], [2, 0], [4, 0], [32, 0]],
            "lane": DOT_B_LANE,
            "warp": [[0, 8], [0, 0]],
            "block": [],
        },
        DIM01,
    )
    assert to_linear_layout(dot(parent, 0, 8), [64, 128]) == ll(
        {
            "register": [[0, 1], [0, 2], [0, 4], [8, 0], [0, 32], [0, 64], [32, 0]],
            "lane": DOT_A_LANE,
            "warp": [[0, 0], [16, 0]],
            "block": [],
        },
        DIM01,
    )
    assert to_linear_layout(dot(parent, 1, 8), [128, 32]) == ll(
        {
            "register": [[1, 0], [2, 0], [4, 0], [32, 0], [64, 0], [0, 16]],
            "lane": DOT_B_LANE,
            "warp": [[0, 8], [0, 0]],
            "block": [],
        },
        DIM01,
    )


# --------------------------------------------------------------------------
# Slice layouts
# --------------------------------------------------------------------------


def test_slice_dot():
    """C++ SliceDot: a slice whose parent is a dot operand."""
    slice_v2 = Slice(1, dot(mma(2, 0, [16, 8], [1, 1]), 0, 8))
    assert to_linear_layout(slice_v2, [16]) == ll(
        {"register": [[8]], "lane": [[0], [0], [1], [2], [4]], "warp": [], "block": []}, DIM0
    )
    slice_v3 = Slice(0, dot(mma(3, 0, [16, 16, 8], [4, 1]), 0, 2))
    assert to_linear_layout(slice_v3, [16]) == ll(
        {
            "register": [[1], [8]],
            "lane": [[2], [4], [0], [0], [0]],
            "warp": [[0], [0]],
            "block": [],
        },
        DIM0,
    )


def test_slice_of_blocked():
    """C++ SliceOfBlocked."""
    parent = blocked([2, 4], [4, 2], [2, 2], [2, 2], [2, 2], [1, 0], [1, 0])
    assert to_linear_layout(Slice(0, parent), [128]) == ll(
        {
            "register": [[1], [2], [16], [32]],
            "lane": [[4], [0], [0]],
            "warp": [[8], [0]],
            "block": [[64], [0]],
        },
        DIM0,
    )
    assert to_linear_layout(Slice(1, parent), [128]) == ll(
        {
            "register": [[1], [16], [32]],
            "lane": [[0], [2], [4]],
            "warp": [[0], [8]],
            "block": [[0], [64]],
        },
        DIM0,
    )


def test_slice_with_shape1():
    """C++ SliceWithShape1."""
    parent = blocked([1, 4], [8, 4], [2, 2], [1, 1], [1, 1], [0, 1], [1, 0])
    assert to_linear_layout(Slice(0, parent), [1]) == ll(
        {
            "register": [],
            "lane": [[0], [0], [0], [0], [0]],
            "warp": [[0], [0]],
            "block": [],
        },
        DIM0,
    )


def test_slice_4d():
    """C++ Slice4D."""
    parent = blocked(
        [1, 1, 1, 4],
        [2, 1, 1, 16],
        [1, 2, 4, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [3, 0, 1, 2],
        [3, 2, 1, 0],
    )
    assert to_linear_layout(Slice(3, parent), [2, 1, 1]) == ll(
        {
            "register": [],
            "lane": [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]],
            "warp": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "block": [],
        },
        DIM012,
    )


def test_slice_of_mma_v2():
    """C++ SliceOfMmaV2."""
    parent = mma(2, 0, [16, 8], [2, 2], [1, 1], [1, 1], [0, 1])
    assert to_linear_layout(Slice(0, parent), [16]) == ll(
        {
            "register": [[1]],
            "lane": [[2], [4], [0], [0], [0]],
            "warp": [[8], [0]],
            "block": [],
        },
        DIM0,
    )
    assert to_linear_layout(Slice(0, parent), [128]) == ll(
        {
            "register": [[1], [16], [32], [64]],
            "lane": [[2], [4], [0], [0], [0]],
            "warp": [[8], [0]],
            "block": [],
        },
        DIM0,
    )
    assert to_linear_layout(Slice(1, parent), [8]) == ll(
        {
            "register": [],
            "lane": [[0], [0], [1], [2], [4]],
            "warp": [[0], [0]],
            "block": [],
        },
        DIM0,
    )
    assert to_linear_layout(Slice(1, parent), [128]) == ll(
        {
            "register": [[8], [32], [64]],
            "lane": [[0], [0], [1], [2], [4]],
            "warp": [[0], [16]],
            "block": [],
        },
        DIM0,
    )


# --------------------------------------------------------------------------
# Swizzled shared layouts
# --------------------------------------------------------------------------


def test_shared_simple_1d():
    """C++ SharedSimple1D."""
    layout = to_linear_layout(shared(1, 1, 1, [1], [1], [0], [0]), [1024])
    assert layout == LinearLayout.identity1D(1024, "offset", "dim0") * LinearLayout.identity1D(
        1, "block", "dim0"
    )


def test_shared_simple_2d():
    """C++ SharedSimple2D."""
    layout = to_linear_layout(shared(1, 1, 1, [1, 1], [1, 1], [1, 0], [1, 0]), [128, 128])
    expected = (
        LinearLayout.identity1D(128, "offset", "dim1")
        * LinearLayout.identity1D(128, "offset", "dim0")
        * LinearLayout.identity1D(1, "block", "dim0")
    ).transpose_outs(DIM01)
    assert layout == expected


def test_shared_simple_2d_order01():
    """C++ SharedSimple2D_Order01."""
    layout = to_linear_layout(shared(1, 1, 1, [1, 1], [1, 1], [0, 1], [1, 0]), [128, 128])
    expected = (
        LinearLayout.identity1D(128, "offset", "dim0")
        * LinearLayout.identity1D(128, "offset", "dim1")
        * LinearLayout.identity1D(1, "block", "dim0")
    )
    assert layout == expected


def test_shared_swizzled_2d_max_phase_only():
    """C++ SharedSwizzled2D_MaxPhaseOnly."""
    layout = to_linear_layout(shared(1, 1, 4, [1, 1], [1, 1], [1, 0], [1, 0]), [32, 32])
    assert layout == ll(
        {
            "offset": [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [1, 1],
                [2, 2],
                [4, 0],
                [8, 0],
                [16, 0],
            ],
            "block": [],
        },
        DIM01,
    )


def test_shared_swizzled_2d_per_phase_max_phase():
    """C++ SharedSwizzled2D_PerPhaseMaxPhase."""
    layout = to_linear_layout(shared(1, 2, 4, [1, 1], [1, 1], [1, 0], [1, 0]), [32, 32])
    assert layout == ll(
        {
            "offset": [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [1, 0],
                [2, 1],
                [4, 2],
                [8, 0],
                [16, 0],
            ],
            "block": [],
        },
        DIM01,
    )


def test_shared_swizzled_2d_vec():
    """C++ SharedSwizzled2D_Vec."""
    layout = to_linear_layout(shared(2, 1, 4, [1, 1], [1, 1], [1, 0], [1, 0]), [4, 8])
    assert layout == ll({"offset": [[0, 1], [0, 2], [0, 4], [1, 2], [2, 4]], "block": []}, DIM01)


def test_shared_swizzled_2d_per_phase_max_phase_vec():
    """C++ SharedSwizzled2D_PerPhaseMaxPhaseVec."""
    layout = to_linear_layout(shared(2, 2, 4, [1, 1], [1, 1], [1, 0], [1, 0]), [32, 32])
    assert layout == ll(
        {
            "offset": [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [1, 0],
                [2, 2],
                [4, 4],
                [8, 0],
                [16, 0],
            ],
            "block": [],
        },
        DIM01,
    )


def test_shared_swizzled_4d():
    """C++ SharedSwizzled4D."""
    layout = to_linear_layout(
        shared(2, 2, 4, [1, 1, 1, 1], [1, 1, 1, 1], [3, 2, 1, 0], [3, 2, 1, 0]),
        [2, 4, 32, 32],
    )
    assert layout == ll(
        {
            "offset": [
                [0, 0, 0, 1],
                [0, 0, 0, 2],
                [0, 0, 0, 4],
                [0, 0, 0, 8],
                [0, 0, 0, 16],
                [0, 0, 1, 0],
                [0, 0, 2, 2],
                [0, 0, 4, 4],
                [0, 0, 8, 0],
                [0, 0, 16, 0],
                [0, 1, 0, 0],
                [0, 2, 0, 0],
                [1, 0, 0, 0],
            ],
            "block": [],
        },
        DIM0123,
    )


def test_shared_swizzled_2d_order01():
    """C++ SharedSwizzled2D_Order01."""
    layout = to_linear_layout(shared(1, 1, 4, [1, 1], [1, 1], [0, 1], [0, 1]), [4, 8])
    assert layout == ll({"offset": [[1, 0], [2, 0], [1, 1], [2, 2], [0, 4]], "block": []}, DIM01)


def test_shared_1d_swizzle():
    """C++ Shared1DSwizzle: a single column cannot be swizzled."""
    layout = to_linear_layout(shared(2, 2, 4, [1, 1], [1, 1], [1, 0], [1, 0]), [64, 1])
    expected = (
        LinearLayout.identity1D(64, "offset", "dim0")
        * LinearLayout.identity1D(1, "offset", "dim1")
        * LinearLayout.identity1D(1, "block", "dim0")
    )
    assert layout == expected


# --------------------------------------------------------------------------
# nvmma_shared layouts
# --------------------------------------------------------------------------


def test_leading_offset_8x16_4_2():
    """C++ LeadingOffset_8x16_4_2."""
    layout = to_linear_layout(nvmma(32, False, 16, [1, 1], [1, 1], [1, 0]), [8, 16])
    assert layout == ll(
        {
            "offset": [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 8]],
            "block": [],
        },
        DIM01,
    )


def test_leading_offset_128x16_4_2():
    """C++ LeadingOffset_128x16_4_2."""
    layout = to_linear_layout(nvmma(32, False, 16, [1, 1], [1, 1], [1, 0]), [128, 16])
    assert layout == ll(
        {
            "offset": [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [1, 0],
                [2, 0],
                [4, 8],
                [8, 0],
                [16, 0],
                [32, 0],
                [64, 0],
            ],
            "block": [],
        },
        DIM01,
    )


def test_leading_offset_8x32_2_4():
    """C++ LeadingOffset_8x32_2_4."""
    layout = to_linear_layout(nvmma(64, False, 16, [1, 1], [1, 1], [1, 0]), [8, 32])
    assert layout == ll(
        {
            "offset": [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 8], [4, 16]],
            "block": [],
        },
        DIM01,
    )


def test_leading_offset_8x64_1_8():
    """C++ LeadingOffset_8x64_1_8."""
    layout = to_linear_layout(nvmma(128, False, 16, [1, 1], [1, 1], [1, 0]), [8, 64])
    assert layout == ll(
        {
            "offset": [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [0, 32],
                [1, 8],
                [2, 16],
                [4, 32],
            ],
            "block": [],
        },
        DIM01,
    )


def test_leading_offset_8x64_1_8_32b():
    """C++ LeadingOffset_8x64_1_8_32b, a non-surjective layout."""
    layout = to_linear_layout(nvmma(128, False, 32, [1, 1], [1, 1], [1, 0]), [8, 64])
    assert layout == ll(
        {
            "offset": [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [1, 4],
                [2, 8],
                [4, 16],
                [0, 32],
            ],
            "block": [],
        },
        [("dim0", 8), ("dim1", 64)],
        False,
    )


def test_leading_offset_128x128_1_8_128b_transposed():
    """C++ LeadingOffset_128x128_1_8_128b_transposed."""
    layout = to_linear_layout(nvmma(128, True, 32, [1, 1], [1, 1], [1, 0]), [128, 128])
    assert layout == ll(
        {
            "offset": [
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
                [16, 0],
                [4, 1],
                [8, 2],
                [16, 4],
                [0, 8],
                [0, 16],
                [0, 32],
                [0, 64],
                [32, 0],
                [64, 0],
            ],
            "block": [],
        },
        [("dim0", 128), ("dim1", 128)],
    )


def test_leading_offset_32x4x64_1_8_32b():
    """C++ LeadingOffset_32x4x64_1_8_32b."""
    layout = to_linear_layout(nvmma(64, False, 32, [1, 1, 1], [1, 1, 1], [2, 1, 0]), [32, 4, 64])
    assert layout == ll(
        {
            "offset": [
                [0, 0, 1],
                [0, 0, 2],
                [0, 0, 4],
                [0, 0, 8],
                [0, 1, 0],
                [0, 2, 4],
                [1, 0, 8],
                [2, 0, 0],
                [4, 0, 0],
                [8, 0, 0],
                [16, 0, 0],
                [0, 0, 16],
                [0, 0, 32],
            ],
            "block": [],
        },
        [("dim0", 32), ("dim1", 4), ("dim2", 64)],
    )


def test_leading_offset_64x4x32_1_8_32b_transposed():
    """C++ LeadingOffset_64x4x32_1_8_32b_transposed."""
    layout = to_linear_layout(nvmma(64, True, 32, [1, 1, 1], [1, 1, 1], [2, 1, 0]), [64, 4, 32])
    assert layout == ll(
        {
            "offset": [
                [1, 0, 0],
                [2, 0, 0],
                [4, 0, 0],
                [8, 0, 0],
                [0, 0, 4],
                [4, 0, 8],
                [8, 0, 16],
                [0, 1, 0],
                [0, 2, 0],
                [0, 0, 1],
                [0, 0, 2],
                [16, 0, 0],
                [32, 0, 0],
            ],
            "block": [],
        },
        [("dim0", 64), ("dim1", 4), ("dim2", 32)],
    )


# --------------------------------------------------------------------------
# Linear encodings, checked against lit tests where Triton itself asserts
# that a #ttg.linear and a legacy encoding describe the same tensor
# --------------------------------------------------------------------------


def test_linear_equals_blocked_tmem_store():
    """test/TritonGPU/canonicalize.mlir @test_canonicalize_convert_tmem_store."""
    legacy = parse_encoding(
        "#ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], "
        "warpsPerCTA = [4, 2], order = [0, 1]}>"
    )
    linear = parse_encoding(
        "#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], "
        "lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], "
        "warp = [[32, 0], [64, 0], [0, 32]], block = []}>"
    )
    assert to_linear_layout(linear, [128, 64]) == to_linear_layout(legacy, [128, 64])


def test_linear_equals_blocked_infer_trans():
    """test/TritonGPU/canonicalize.mlir @infer_trans."""
    legacy = parse_encoding(
        "#ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 4], "
        "warpsPerCTA = [1, 4], order = [1, 0]}>"
    )
    linear = parse_encoding(
        "#ttg.linear<{register = [[0, 1], [8, 0], [16, 0]], "
        "lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], "
        "warp = [[0, 8], [0, 16]], block = []}>"
    )
    assert to_linear_layout(linear, [32, 32]) == to_linear_layout(legacy, [32, 32])


def test_linear_equals_dot_operand_simplify_trans_trans():
    """test/TritonGPU/canonicalize.mlir @simplify_trans_trans."""
    legacy = parse_encoding(
        "#ttg.dot_op<{opIdx = 0, parent = #ttg.nvidia_mma<{versionMajor = 3, "
        "versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>, kWidth = 2}>"
    )
    linear = parse_encoding(
        "#ttg.linear<{register = [[0, 1], [8, 0], [0, 8], [0, 16], [0, 32], [0, 64], "
        "[0, 128], [64, 0], [128, 0]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], "
        "warp = [[16, 0], [32, 0]], block = []}>"
    )
    assert to_linear_layout(linear, [256, 256]) == to_linear_layout(legacy, [256, 256])


def test_linear_encoding_round_trips_a_blocked_layout():
    """Wrapping a computed layout as `#ttg.linear` is the identity on it."""
    legacy = blocked([1, 4], [2, 16], [4, 1], [1, 1], [1, 1], [1, 0], [1, 0])
    computed = to_linear_layout(legacy, [64, 64])
    assert to_linear_layout(Linear(computed), [64, 64]) == computed


def test_linear_encoding_broadcasts_and_clips():
    """A `#ttg.linear` smaller than the tensor repeats, larger one drops registers."""
    linear = parse_encoding(
        "#ttg.linear<{register = [[1]], lane = [[2], [4], [8], [16], [32]], "
        "warp = [[64]], block = []}>"
    )
    assert to_linear_layout(linear, [256]).bases["register"] == [[1], [128]]
    assert to_linear_layout(linear, [64]).bases["register"] == [[1]]


# --------------------------------------------------------------------------
# Parsing the printed attribute
# --------------------------------------------------------------------------


def test_parse_blocked():
    enc = parse_encoding(
        "#ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], "
        "warpsPerCTA = [4, 1], order = [1, 0]}>"
    )
    assert enc == blocked([1, 4], [2, 16], [4, 1], [1, 1], [1, 1], [1, 0], [1, 0])


def test_parse_blocked_with_cga_layout():
    enc = parse_encoding(
        "#ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], "
        "order = [0], CGALayout = [[16], [0]]}>"
    )
    assert isinstance(enc, Blocked)
    assert enc.cga.bases["block"] == [[16], [0]]


def test_parse_nested_slice_and_dot_op():
    enc = parse_encoding(
        "#ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, "
        "parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, "
        "warpsPerCTA = [1, 1], instrShape = [16, 8]}>, kWidth = 8}>}>"
    )
    assert enc == Slice(1, dot(mma(2, 0, [16, 8], [1, 1]), 0, 8))


def test_parse_shared_encodings():
    swizzled = parse_encoding(
        "#ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>"
    )
    assert swizzled == shared(2, 2, 4, [1, 1], [1, 1], [1, 0], [1, 0])
    nv = parse_encoding(
        "#ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>"
    )
    assert nv == nvmma(128, False, 16, [1, 1], [1, 1], [1, 0])
    ranked = parse_encoding(
        "#ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, "
        "elementBitWidth = 32, rank = 3}>"
    )
    assert isinstance(ranked, NvmmaShared)
    assert ranked.rank == 3 and ranked.transposed


def test_parse_linear():
    enc = parse_encoding(
        "#ttg.linear<{register = [], lane = [[0], [0], [0], [1], [0]], warp = [[0]], block = []}>"
    )
    assert isinstance(enc, Linear)
    assert enc.layout.bases["lane"] == [[0], [0], [0], [1], [0]]
    assert enc.layout.out_dims == {"dim0": 2}


def test_parse_rejects_unknown_and_malformed():
    with pytest.raises(EncodingParseError):
        parse_encoding("#ttg.amd_mfma<{versionMajor = 2}>")
    with pytest.raises(EncodingParseError):
        parse_encoding("#ttg.blocked<{sizePerThread = [1]}> trailing")


# --------------------------------------------------------------------------
# Module attributes
# --------------------------------------------------------------------------


def test_module_attributes_are_checked():
    enc = blocked([1, 4], [2, 16], [4, 1], [1, 1], [1, 1], [1, 0], [1, 0])
    to_linear_layout(enc, [64, 64], num_warps=4, threads_per_warp=32)
    with pytest.raises(LayoutError):
        to_linear_layout(enc, [64, 64], num_warps=8)
    with pytest.raises(LayoutError):
        to_linear_layout(enc, [64, 64], threads_per_warp=64)


def test_drop_pipelining_dims():
    """`dropPipeliningDim`: a multi-buffered memdesc lays out its trailing dims."""
    enc = parse_encoding(
        "#ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, "
        "elementBitWidth = 16}>, #ttg.shared_memory, mutable"
    )
    assert drop_pipelining_dims([2, 16, 64], enc) == [16, 64]
    layout = to_linear_layout(enc, drop_pipelining_dims([2, 16, 64], enc))
    assert layout.out_dims == {"dim0": 16, "dim1": 64}
    with pytest.raises(LayoutError):
        to_linear_layout(enc, [2, 16, 64])


# --------------------------------------------------------------------------
# Every encoding printed in the fixtures
# --------------------------------------------------------------------------


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _encodings_of(module) -> set[tuple[str, tuple[int, ...]]]:
    found: set[tuple[str, tuple[int, ...]]] = set()

    def visit(op) -> None:
        for parsed in list(op.result_types) + list(op.operand_types):
            for candidate in (parsed, parsed.elem):
                if candidate is not None and candidate.encoding:
                    found.add((candidate.encoding, candidate.shape))
        for region in op.regions:
            for block in region.blocks:
                for _, arg_type in block.args:
                    if arg_type.encoding:
                        found.add((arg_type.encoding, arg_type.shape))
                for inner in block.ops:
                    visit(inner)

    for op in module.ops:
        visit(op)
    return found


def test_every_fixture_encoding_has_a_linear_layout():
    """Each layout printed in the TTGIR fixtures parses and converts."""
    kinds: set[str] = set()
    total = 0
    for path in sorted(FIXTURES_DIR.glob("*.ttgir.generic")):
        for text, shape in sorted(_encodings_of(mlir.parse(path.read_text()))):
            encoding = parse_encoding(text)
            layout = to_linear_layout(encoding, drop_pipelining_dims(shape, encoding))
            assert layout.out_dims == {
                f"dim{i}": size for i, size in enumerate(drop_pipelining_dims(shape, encoding))
            }
            kinds.add(type(encoding).__name__)
            total += 1
    assert total > 0
    assert kinds == {
        "Blocked",
        "DotOperand",
        "Linear",
        "NvidiaMma",
        "NvmmaShared",
        "Slice",
        "SwizzledShared",
    }
