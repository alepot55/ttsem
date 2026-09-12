"""Tests of the harness pieces that do not need a Triton install: `split_dump` and `compare`.

Everything that runs a real launch (`capture_launches`, `run_launch`, `validate`) needs Triton
and, without a GPU, the fake driver; those are exercised against the checked-in
`tests/fixtures/` by hand (see the harness/validate module docstrings), not here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ttsem import harness, validate

_HAS_SEMANTICS = (
    importlib.util.find_spec("ttsem.mlir") is not None
    and importlib.util.find_spec("ttsem.interp") is not None
    and importlib.util.find_spec("ttsem.memory") is not None
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"

_NEEDS_TRITON = pytest.mark.skipif(
    not harness.TRITON_AVAILABLE, reason="needs a Triton install to record a launch"
)

SAMPLE_DUMP = """\
// -----// IR Dump Before InlinerPass: inline{threshold=4294967295} \
('builtin.module' operation) //----- //
module {
  tt.func @kernel() {
    tt.return
  }
}
// -----// IR Dump Before SomeFuncPass ('tt.func' operation: @kernel) //----- //
tt.func @kernel() {
  tt.return
}
// -----// IR Dump Before CanonicalizerPass: canonicalize{top-down=true} \
('builtin.module' operation) //----- //
module {
  tt.func @kernel() {
    %0 = arith.constant 1 : i32
    tt.return
  }
}
"""


def test_split_dump_keeps_only_whole_module_dumps() -> None:
    modules = validate.split_dump(SAMPLE_DUMP)
    assert len(modules) == 2


def test_split_dump_drops_per_function_dumps() -> None:
    modules = validate.split_dump(SAMPLE_DUMP)
    for _name, text in modules:
        assert "SomeFuncPass" not in text


def test_split_dump_preserves_pass_order_and_names() -> None:
    modules = validate.split_dump(SAMPLE_DUMP)
    names = [name for name, _text in modules]
    assert names[0].startswith("Before InlinerPass")
    assert names[1].startswith("Before CanonicalizerPass")


def test_split_dump_module_text_excludes_header_line() -> None:
    modules = validate.split_dump(SAMPLE_DUMP)
    _name, first_text = modules[0]
    assert "IR Dump" not in first_text
    assert "tt.func @kernel()" in first_text


def test_split_dump_second_module_has_the_constant() -> None:
    modules = validate.split_dump(SAMPLE_DUMP)
    _name, second_text = modules[1]
    assert "arith.constant 1 : i32" in second_text


def test_split_dump_empty_text() -> None:
    assert validate.split_dump("") == []


def test_split_dump_ignores_non_header_noise() -> None:
    text = "some warning printed to stderr\n" + SAMPLE_DUMP + "\ntrailing noise\n"
    modules = validate.split_dump(text)
    assert len(modules) == 2


def test_compare_equal_arrays() -> None:
    a = np.array([1, 2, 3], dtype=np.int32)
    result = harness.compare(a, a.copy())
    assert result["equal"] is True
    assert result["n_diff"] == 0
    assert result["diffs"] == []


def test_compare_reports_first_diffs() -> None:
    ref = np.array([1, 2, 3, 4], dtype=np.int32)
    dev = np.array([1, 0, 3, 0], dtype=np.int32)
    result = harness.compare(ref, dev)
    assert result["equal"] is False
    assert result["n_diff"] == 2
    assert result["diffs"] == [[1, 2, 0], [3, 4, 0]]


def test_compare_shape_mismatch_is_not_equal() -> None:
    ref = np.array([1, 2, 3], dtype=np.int32)
    dev = np.array([1, 2], dtype=np.int32)
    result = harness.compare(ref, dev)
    assert result["equal"] is False
    assert result["n_diff"] == -1


def test_compare_dtype_mismatch_is_not_equal() -> None:
    ref = np.array([1, 2, 3], dtype=np.int32)
    dev = np.array([1, 2, 3], dtype=np.int64)
    result = harness.compare(ref, dev)
    assert result["equal"] is False


def test_compare_floats_by_bit_pattern_distinguishes_signed_zero() -> None:
    ref = np.array([0.0], dtype=np.float32)
    dev = np.array([-0.0], dtype=np.float32)
    result = harness.compare(ref, dev)
    assert result["equal"] is False
    assert result["n_diff"] == 1


def test_compare_floats_by_bit_pattern_distinguishes_nan_payloads() -> None:
    ref = np.array([np.nan], dtype=np.float32).view(np.uint32)
    ref[0] = 0x7FC00001
    dev = np.array([np.nan], dtype=np.float32).view(np.uint32)
    dev[0] = 0x7FC00002
    ref_f = ref.view(np.float32)
    dev_f = dev.view(np.float32)
    result = harness.compare(ref_f, dev_f)
    assert result["equal"] is False


def test_compare_same_nan_payload_is_equal() -> None:
    bits = np.array([0x7FC00001], dtype=np.uint32)
    a = bits.view(np.float32).copy()
    b = bits.view(np.float32).copy()
    result = harness.compare(a, b)
    assert result["equal"] is True


def test_bind_names_positional_and_keyword() -> None:
    bound = harness._bind_names(["a", "b", "c"], (1, 2), {"c": 3})
    assert bound == {"a": 1, "b": 2, "c": 3}


def test_grid3_pads_to_three_dims() -> None:
    assert harness._grid3((1,)) == (1, 1, 1)
    assert harness._grid3((3, 2)) == (3, 2, 1)
    assert harness._grid3((3, 2, 1)) == (3, 2, 1)


def test_aggregate_prefers_error_over_mismatch() -> None:
    assert harness._aggregate(["match", "error", "mismatch"]) == "error"


def test_aggregate_all_match() -> None:
    assert harness._aggregate(["match", "match"]) == "match"


def test_aggregate_empty_is_error() -> None:
    assert harness._aggregate([]) == "error"


@pytest.mark.skipif(not _HAS_SEMANTICS, reason="mlir.py/interp.py/memory.py not available yet")
@pytest.mark.parametrize("name", ["p1", "p2", "p3"])
@_NEEDS_TRITON
def test_run_launch_matches_device_on_fixtures(name: str) -> None:
    """Level-1 fixtures (no tensor descriptors): the checked-in generic ttir must reproduce
    exactly the recorded (pre, post) state of a real launch."""
    program = FIXTURES / f"{name}.py"
    records = harness.capture_launches(program, "cpu")
    assert len(records) == 1
    generic = (FIXTURES / f"{name}.ttir.generic").read_text()
    result = harness.run_launch(records[0], generic)
    assert result.verdict == "match", result.message or result.diffs
    assert result.n_diff == 0


def test_split_dump_separates_intermediate_modules() -> None:
    from ttsem.validate import split_dump

    dump = """// -----// IR Dump Before P: p ('builtin.module' operation) //----- //
#blocked = #ttg.blocked<{a = [1]}>
module {
  tt.func @k() { tt.return }
}
#blocked = #ttg.blocked<{a = [2]}>
module {
  tt.func @k() { tt.return }
}
// -----// IR Dump Before Q: q ('builtin.module' operation) //----- //
module {
  tt.func @k() { tt.return }
}
"""
    stages = split_dump(dump)
    assert [name for name, _ in stages] == ["Before P: p", "Before P: p (step 1)", "Before Q: q"]
    assert stages[0][1].startswith("#blocked = #ttg.blocked<{a = [1]}>\nmodule")
    assert stages[1][1].startswith("#blocked = #ttg.blocked<{a = [2]}>\nmodule")
    assert stages[2][1].startswith("module")


def test_to_numpy_keeps_bf16_bit_patterns() -> None:
    torch = pytest.importorskip("torch")
    from ttsem.harness import _to_numpy

    t = torch.tensor([1.0, -2.0, 0.5], dtype=torch.bfloat16)
    got = _to_numpy(t)
    assert got.dtype == np.uint16
    assert got.tolist() == [0x3F80, 0xC000, 0x3F00]


def test_descriptor_values_flatten_shape_and_strides() -> None:
    torch = pytest.importorskip("torch")
    from ttsem.harness import descriptor_values

    class Desc:
        base = torch.zeros((8, 16), dtype=torch.float16)
        shape = [8, 16]
        strides = [16, 1]
        block_shape = [8, 8]

    vals = descriptor_values(Desc(), 4096)
    assert len(vals) == 5
    d = vals[0]
    assert (d.base, d.shape, d.strides, d.block_shape, d.elem.name) == (
        4096,
        (8, 16),
        (16, 1),
        (8, 8),
        "f16",
    )
    assert [v.dtype.name for v in vals[1:]] == ["int32", "int32", "int64", "int64"]
    assert [int(v) for v in vals[1:]] == [8, 16, 16, 1]


def test_storage_copy_sees_through_a_reinterpret_wrapper() -> None:
    torch = pytest.importorskip("torch")
    from ttsem.harness import storage_copy

    base = torch.arange(6, dtype=torch.int32)

    class Wrapper:  # what triton.reinterpret returns: a Triton dtype over a torch tensor
        def __init__(self, t: object) -> None:
            self.base = t
            self.dtype = "uint32"
            self.shape = t.shape

        def data_ptr(self) -> int:
            return self.base.data_ptr()

    ptr, arr = storage_copy(Wrapper(base[2:]))
    assert ptr == base.data_ptr()
    assert arr.tolist() == [0, 1, 2, 3, 4, 5]


def test_compare_reports_ulp_distance() -> None:
    from ttsem.harness import compare

    ref = np.array([1.0, 2.0, -3.0], np.float32)
    dev = ref.copy()
    dev[1] = np.nextafter(dev[1], np.float32(10))
    dev[2] = np.nextafter(np.nextafter(dev[2], np.float32(-10)), np.float32(-10))
    cmp = compare(ref, dev)
    assert cmp["n_diff"] == 2 and cmp["max_ulp"] == 2
    dev[0] = np.float32("nan")
    assert compare(ref, dev)["max_ulp"] > 1 << 30


# ------------------------------------------------------- the float comparison policy


def _module(body: str) -> object:
    from ttsem import mlir

    return mlir.parse('"builtin.module"() ({\n' + body + "\n}) : () -> ()")


def test_ulp_distance_is_right_for_f64() -> None:
    # `np.int64(1) << 63` overflows, so the old signed form gave nonsense here.
    ref = np.array([1.0, -1.0], np.float64)
    dev = np.array([np.nextafter(1.0, 10.0), np.nextafter(-1.0, -10.0)], np.float64)
    assert harness._ulp_distance(ref, dev).tolist() == [1, 1]


def test_ulp_distance_separates_the_two_zeros_by_one() -> None:
    ref = np.array([0.0], np.float64)
    dev = np.array([-0.0], np.float64)
    assert harness._ulp_distance(ref, dev).tolist() == [1]


def test_compare_two_nans_are_equal_whatever_the_payload() -> None:
    ref = np.array([0x7FC00001], np.uint32).view(np.float32)
    dev = np.array([0xFFC00007], np.uint32).view(np.float32)
    cmp = harness.compare(ref, dev)
    assert not cmp["equal"]  # `match` stays bit for bit
    assert cmp["approx"] is True


def test_compare_nan_against_a_number_is_not_approx() -> None:
    ref = np.array([np.nan], np.float32)
    dev = np.array([1.0], np.float32)
    assert harness.compare(ref, dev)["approx"] is False


def test_compare_infinities_of_opposite_sign_are_never_within_tolerance() -> None:
    ref = np.array([np.inf], np.float32)
    dev = np.array([-np.inf], np.float32)
    policy = harness.FloatPolicy("f32", rtol=1e-4, atol=1e-5)
    assert harness.compare(ref, dev, policy=policy)["approx"] is False


def test_compare_without_a_policy_stays_bitwise() -> None:
    ref = np.array([1.0], np.float32)
    dev = np.array([np.nextafter(np.float32(1.0), np.float32(10))], np.float32)
    cmp = harness.compare(ref, dev)
    assert cmp["approx"] is False and cmp["max_ulp"] == 1


def test_compare_with_a_relative_tolerance_reports_max_rel_err() -> None:
    ref = np.array([1.0, 100.0], np.float32)
    dev = np.array([1.00001, 100.001], np.float32)
    policy = harness.FloatPolicy("f32", rtol=1e-4, atol=1e-5)
    cmp = harness.compare(ref, dev, policy=policy)
    assert cmp["approx"] is True and cmp["max_rel_err"] < 1e-4


def test_compare_outside_the_relative_tolerance_is_a_mismatch() -> None:
    ref = np.array([1.0], np.float32)
    dev = np.array([1.01], np.float32)
    policy = harness.FloatPolicy("f32", rtol=1e-4, atol=1e-5)
    assert harness.compare(ref, dev, policy=policy)["approx"] is False


def test_compare_decodes_bf16_bit_patterns_for_the_tolerance() -> None:
    # a bf16 buffer travels as uint16, so the element spelling in the policy is what makes the
    # relative tolerance mean anything at all here.
    ref = np.array([0x3F80], np.uint16)  # 1.0
    dev = np.array([0x3F81], np.uint16)  # 1.0078125, one bf16 ulp away
    policy = harness.FloatPolicy("bf16", rtol=1e-2, atol=1e-2)
    assert harness.compare(ref, dev, policy=policy)["approx"] is True
    # without the spelling the same buffer is two plain uint16, and stays bitwise
    assert "approx" not in harness.compare(ref, dev)


def test_compare_integer_buffers_are_never_approx() -> None:
    ref = np.array([1, 2], np.int32)
    dev = np.array([1, 3], np.int32)
    policy = harness.FloatPolicy("f32", rtol=1.0, atol=1.0)
    cmp = harness.compare(ref, dev, policy=policy)
    assert "approx" not in cmp and cmp["n_diff"] == 1


@pytest.mark.skipif(not _HAS_SEMANTICS, reason="mlir.py not available yet")
def test_scan_inexact_finds_a_float_reduction() -> None:
    body = (
        '  "tt.func"() <{sym_name = "k"}> ({\n'
        '    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({\n'
        "    ^bb0(%a: f32, %b: f32):\n"
        '      %s = "arith.addf"(%a, %b) : (f32, f32) -> f32\n'
        '      "tt.reduce.return"(%s) : (f32) -> ()\n'
        "    }) : (tensor<4xf32>) -> f32\n"
        '    "tt.return"() : () -> ()\n'
        "  }) : () -> ()"
    )
    scan = harness.scan_inexact(_module(body))
    assert scan.classes == {"wide"} and scan.elems == {"f32"}


@pytest.mark.skipif(not _HAS_SEMANTICS, reason="mlir.py not available yet")
@pytest.mark.parametrize("flag, expect", [("true", {"reorder"}), ("false", set())])
def test_scan_inexact_reads_can_reorder_on_a_cat(flag: str, expect: set[str]) -> None:
    body = (
        '  "tt.func"() <{sym_name = "k"}> ({\n'
        f'    %0 = "tt.cat"(%arg0, %arg1) <{{can_reorder = {flag}}}> '
        ": (tensor<4xi32>, tensor<4xi32>) -> tensor<8xi32>\n"
        '    "tt.return"() : () -> ()\n'
        "  }) : () -> ()"
    )
    assert harness.scan_inexact(_module(body)).classes == expect


def test_compare_multiset_explains_a_permutation_and_nothing_else() -> None:
    policy = harness.FloatPolicy("i32", multiset=True)
    ref = np.array([1, 2, 3, 4], np.int32)
    same_values = harness.compare(ref, np.array([4, 3, 2, 1], np.int32), policy=policy)
    assert same_values["approx"] and same_values["n_diff"] == 4
    other_values = harness.compare(ref, np.array([4, 3, 2, 2], np.int32), policy=policy)
    assert not other_values.get("approx")
    # without the policy a permutation of an integer buffer stays a plain mismatch
    assert "approx" not in harness.compare(ref, np.array([4, 3, 2, 1], np.int32))


@pytest.mark.skipif(not _HAS_SEMANTICS, reason="mlir.py not available yet")
def test_scan_inexact_finds_only_the_division() -> None:
    body = (
        '  "tt.func"() <{sym_name = "k"}> ({\n'
        '    %0 = "arith.divf"(%arg0, %arg1) : (f32, f32) -> f32\n'
        '    "tt.return"() : () -> ()\n'
        "  }) : () -> ()"
    )
    scan = harness.scan_inexact(_module(body))
    assert scan.classes == {"div"} and scan.elems == {"f32"}


@pytest.mark.skipif(not _HAS_SEMANTICS, reason="mlir.py not available yet")
def test_scan_inexact_is_empty_for_integer_only_ir() -> None:
    body = (
        '  "tt.func"() <{sym_name = "k"}> ({\n'
        '    %0 = "arith.addi"(%arg0, %arg1) : (i32, i32) -> i32\n'
        '    "tt.return"() : () -> ()\n'
        "  }) : () -> ()"
    )
    assert harness.scan_inexact(_module(body)).classes == set()


@pytest.mark.skipif(not _HAS_SEMANTICS, reason="mlir.py not available yet")
def test_scan_inexact_ignores_an_integer_reduction() -> None:
    body = (
        '  "tt.func"() <{sym_name = "k"}> ({\n'
        '    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({\n'
        "    ^bb0(%a: i32, %b: i32):\n"
        '      %s = "arith.addi"(%a, %b) : (i32, i32) -> i32\n'
        '      "tt.reduce.return"(%s) : (i32) -> ()\n'
        "    }) : (tensor<4xi32>) -> i32\n"
        '    "tt.return"() : () -> ()\n'
        "  }) : () -> ()"
    )
    assert harness.scan_inexact(_module(body)).classes == set()


def _scan(classes: set[str], elems: set[str]) -> harness.InexactScan:
    return harness.InexactScan(frozenset(classes), frozenset(elems))


def test_float_policy_division_only_is_two_ulp_of_the_computing_type() -> None:
    policy = harness.float_policy("f32", _scan({"div"}, {"f32"}))
    band = harness.APPROX_ULP * harness.ULP_AS_RTOL["f32"]
    assert policy.rtol == band and policy.atol == band  # relative, or of the buffer's magnitude


def test_float_policy_division_uses_the_computing_type_not_the_buffer() -> None:
    # `test_bin_op` with `/` divides in f32 and stores into an f64 buffer.
    policy = harness.float_policy("f64", _scan({"div"}, {"f32"}))
    assert policy.rtol == harness.APPROX_ULP * 2.0**-23


def test_float_policy_wide_uses_the_band_of_the_element_type() -> None:
    assert harness.float_policy("bf16", _scan({"wide"}, {"bf16"})).rtol == 1e-2
    assert harness.float_policy("f32", _scan({"wide", "div"}, {"f32"})).rtol == 1e-4


def test_float_policy_without_an_inexact_op_is_bitwise() -> None:
    policy = harness.float_policy("f32", harness.InexactScan())
    assert policy.rtol == 0.0 and policy.atol == 0.0


def test_compare_scales_the_absolute_tolerance_by_the_buffer_range() -> None:
    # a cancelled element has an unbounded relative error and a bounded absolute one
    ref = np.array([100.0, 1e-4], np.float32)
    dev = np.array([100.0, -1e-4], np.float32)
    policy = harness.FloatPolicy("f32", rtol=1e-4, atol=1e-5)
    cmp = harness.compare(ref, dev, policy=policy)
    assert cmp["approx"] is True  # 2e-4 <= 1e-5 * 100
    assert cmp["max_rel_err"] == 2.0 and cmp["max_abs_err"] == pytest.approx(2e-4)
    # the same difference in a buffer with no large element is a mismatch
    small = harness.compare(ref[1:], dev[1:], policy=policy)
    assert small["approx"] is False


# ------------------------------------------------------- the run must not eat the record


@pytest.mark.skipif(not _HAS_SEMANTICS, reason="mlir.py/interp.py/memory.py not available yet")
@pytest.mark.parametrize("name", ["p1", "p2", "p3"])
@_NEEDS_TRITON
def test_run_launch_leaves_the_record_untouched(name: str) -> None:
    """`Memory.register` keeps the array it is given, so a run must register a copy.

    Otherwise the interpreter's stores land in `record.pre` and the second run of the same
    record (which is exactly what the per-pass validator does, once per pass dump) starts from
    post-launch inputs. Only a kernel whose stores are idempotent survives that.
    """
    program = FIXTURES / f"{name}.py"
    records = harness.capture_launches(program, "cpu")
    generic = (FIXTURES / f"{name}.ttir.generic").read_text()
    before = {key: value.copy() for key, value in records[0].pre.items()}
    first = harness.run_launch(records[0], generic)
    for key, value in records[0].pre.items():
        assert np.array_equal(value, before[key]), f"{key} was overwritten by the run"
    second = harness.run_launch(records[0], generic)
    assert first.verdict == second.verdict == "match"


def test_bind_names_fills_a_missing_parameter_from_its_default() -> None:
    bound = harness._bind_names(["a", "b", "c"], (1,), {"c": 3}, {"b": 7, "c": 9})
    assert bound == {"a": 1, "b": 7, "c": 3}


def test_culprit_is_the_pass_after_which_the_device_agrees() -> None:
    from ttsem.validate import PassResult, culprit_of

    def st(name: str, verdict: str) -> PassResult:
        return PassResult(name, verdict, 0 if verdict == "match" else 5, [], "")

    stages = [
        st("Before Inliner", "mismatch"),
        st("Before Canonicalizer", "mismatch"),
        st("Before FuseNestedLoops", "mismatch"),
        st("Before Canonicalizer (2)", "unsupported"),
        st("Before LICM", "match"),
        st("Before Pipeline", "match"),
    ]
    assert culprit_of(stages) == "Before FuseNestedLoops"
    assert culprit_of([st("a", "match"), st("b", "mismatch"), st("c", "match")]) == "b"
    assert culprit_of([st("a", "match"), st("b", "mismatch")]) is None
    assert culprit_of([st("a", "match"), st("b", "match")]) is None


@pytest.mark.parametrize(
    ("verdicts", "expect"),
    [
        (["match", "approx"], "approx"),
        (["match", "poison"], "poison"),
        (["approx", "poison"], "poison"),
        (["poison", "unsupported"], "unsupported"),
        (["unsupported", "mismatch"], "mismatch"),
        (["mismatch", "error"], "error"),
        (["match", "match"], "match"),
    ],
)
def test_aggregate_takes_the_worst_launch(verdicts: list[str], expect: str) -> None:
    assert harness._aggregate(verdicts) == expect
    assert harness._aggregate(list(reversed(verdicts))) == expect


@_NEEDS_TRITON
def test_validate_stages_takes_modules_from_any_source() -> None:
    from ttsem.validate import validate_stages

    program = FIXTURES / "p1.py"
    record = harness.capture_launches(program, "cpu")[0]
    generic = (FIXTURES / "p1.ttir.generic").read_text()
    report = validate_stages(
        record, [("Before A: a", generic), ("Before B: b", generic)], None, "p1"
    )
    assert [r.verdict for r in report.passes] == ["match", "match"]
    assert report.first_bad_pass is None and report.culprit is None


@_NEEDS_TRITON
def test_validate_stages_hands_the_llvm_stages_to_the_continuation() -> None:
    from ttsem.validate import PassResult, validate_stages

    program = FIXTURES / "p1.py"
    record = harness.capture_launches(program, "cpu")[0]
    generic = (FIXTURES / "p1.ttir.generic").read_text()
    llvm = "module { llvm.func @k() { llvm.return } }"
    seen: list[list[str]] = []

    def below(rec, stages):
        assert rec is record
        seen.append([name for name, _ in stages])
        return [
            PassResult(name, "match" if i else "mismatch", 0, [])
            for i, (name, _) in enumerate(stages)
        ]

    report = validate_stages(
        record,
        [("Before A: a", generic), ("Before L: llvm", llvm), ("Before P: ptx", llvm)],
        None,
        "p1",
        below_llvm=below,
    )
    assert seen == [["Before L: llvm", "Before P: ptx"]]
    assert [r.verdict for r in report.passes] == ["match", "mismatch", "match"]
    assert report.first_bad_pass == "Before L: llvm" and report.culprit == "Before L: llvm"
    plain = validate_stages(
        record, [("Before A: a", generic), ("Before L: llvm", llvm)], None, "p1"
    )
    assert [r.verdict for r in plain.passes] == ["match", "unsupported"]


def test_defuse_resolves_every_operand_of_a_fixture() -> None:
    from ttsem import mlir
    from ttsem.defuse import definer, defs, users

    module = mlir.parse((FIXTURES / "p3.ttir.generic").read_text())
    table = defs(module)
    unresolved = []

    def visit(ops):
        for op in ops:
            for name in op.operands:
                if definer(table, op, name) is None:
                    unresolved.append((op.name, name))
            for r in op.regions:
                for b in r.blocks:
                    visit(b.ops)

    for op in module.ops:
        for r in op.regions:
            for b in r.blocks:
                visit(b.ops)
    assert unresolved == []
    used = users(module)
    assert any(len(v) > 1 for v in used.values())  # a splat or range used more than once


def test_parents_and_ancestors_follow_the_region_nesting() -> None:
    from ttsem import mlir
    from ttsem.defuse import ancestors, parents

    module = mlir.parse((FIXTURES / "p3.ttir.generic").read_text())
    table = parents(module)
    assert any(o is not None and o.name == "scf.for" for o, _ in table.values())
    body_op = next(
        op
        for op in _all_ops(module)
        if table[id(op)][0] is not None and table[id(op)][0].name == "scf.for"
    )
    chain = [o.name for o in ancestors(table, body_op)]
    assert chain[0] == "scf.for" and chain[-1] == "tt.func"


def _all_ops(module):
    out = []

    def visit(ops):
        for op in ops:
            out.append(op)
            for r in op.regions:
                for b in r.blocks:
                    visit(b.ops)

    for op in module.ops:
        for r in op.regions:
            for b in r.blocks:
                visit(b.ops)
    return out


def test_memory_read_trace_records_byte_addresses() -> None:
    from ttsem.memory import Memory

    memory = Memory()
    memory.trace_reads = True
    memory.register(1000, np.arange(8, dtype=np.int32))
    memory.load(np.array([1000, 1008]), None, None, np.dtype(np.int32))
    assert memory.read_bytes().tolist() == [1000, 1001, 1002, 1003, 1008, 1009, 1010, 1011]


@_NEEDS_TRITON
def test_validate_stages_reports_extra_reads_against_the_first_stage() -> None:
    from ttsem.validate import validate_stages

    record = harness.capture_launches(FIXTURES / "p1.py", "cpu")[0]
    generic = (FIXTURES / "p1.ttir.generic").read_text()
    report = validate_stages(record, [("a", generic), ("b", generic)], None, "p1")
    assert [r.extra_reads for r in report.passes] == [0, 0]
    assert report.passes[0].verdict == "match"


def test_split_dump_separates_generic_intermediate_modules() -> None:
    from ttsem.validate import split_dump

    dump = """// -----// IR Dump Before P: p ('builtin.module' operation) //----- //
"builtin.module"() ({
  "tt.func"() : () -> ()
}) : () -> ()
"builtin.module"() ({
  "tt.func"() : () -> ()
}) : () -> ()
"""
    stages = split_dump(dump)
    assert [name for name, _ in stages] == ["Before P: p", "Before P: p (step 1)"]


def test_split_modules_leaves_no_step_header_on_the_previous_module() -> None:
    from ttsem.validate import split_dump

    dump = "\n".join(
        [
            "// -----// IR Dump Before TritonGPUPipeline (tritongpu-pipeline) "
            "('builtin.module' operation) //----- //",
            "module {",
            "  tt.func @k() { tt.return }",
            "}",
            "",
            "// -----// intermediate step 1 //----- //",
            "#loc = loc(unknown)",
            "module {",
            "  tt.func @k() { tt.return }",
            "}",
            "",
        ]
    )
    stages = split_dump(dump)
    assert [name for name, _ in stages] == [
        "Before TritonGPUPipeline (tritongpu-pipeline)",
        "Before TritonGPUPipeline (tritongpu-pipeline) (step 1)",
    ]
    first, second = (text for _, text in stages)
    assert "//" not in first and first.endswith("}")
    assert second.startswith("#loc") and second.endswith("}")


def test_a_multiply_feeding_an_add_is_the_contraction_class() -> None:
    from ttsem import mlir

    ty = "tensor<64xf32>"
    text = f"""
"builtin.module"() ({{
  "tt.func"() <{{sym_name = "k", function_type = () -> ()}}> ({{
    %a = "arith.constant"() <{{value = dense<1.5> : {ty}}}> : () -> {ty}
    %m = "arith.mulf"(%a, %a) <{{fastmath = #arith.fastmath<none>}}> : ({ty}, {ty}) -> {ty}
    %s = "arith.addf"(%m, %a) <{{fastmath = #arith.fastmath<none>}}> : ({ty}, {ty}) -> {ty}
    "tt.return"() : () -> ()
  }}) : () -> ()
}}) : () -> ()
"""
    scan = harness.scan_inexact(mlir.parse(text))
    assert scan.classes == frozenset({"contract"}) and "f32" in scan.elems
    policy = harness.float_policy("f32", scan)
    assert 0 < policy.rtol <= harness.APPROX_ULP * harness.ULP_AS_RTOL["f32"]
    plain = text.replace('"arith.mulf"', '"arith.addf"')
    assert not harness.scan_inexact(mlir.parse(plain)).classes


def test_a_buffer_that_differs_only_where_exchanged_is_device_ordered() -> None:
    from ttsem.memory import Memory

    mem = Memory()
    mem.register(1000, np.zeros(8, np.int64))
    mem.atomic("xchg", np.array([1000 + 8 * 3, 1000 + 8 * 5]), np.array([7, 9], np.int64), None)
    assert [list(a) for a in mem.exchanged] == [[1024, 1040]]
    record = harness.LaunchRecord(
        0,
        None,
        "k",
        "",
        (1,),
        (),
        {},
        {},
        {},
        {},
        {"ws": 1000},
        {"ws": 1000},
        {"ws": "torch.int64"},
    )
    want = np.arange(8, dtype=np.int64)
    got = want.copy()
    got[3] = 100
    got[5] = 200
    assert harness._only_exchanged_differ(mem, record, "ws", want, got)
    got[0] = 5  # the whole buffer is protocol state once an exchange landed in it
    assert harness._only_exchanged_differ(mem, record, "ws", want, got)
    assert not harness._only_exchanged_differ(Memory(), record, "ws", want, got)
    other = harness.LaunchRecord(
        0,
        None,
        "k",
        "",
        (1,),
        (),
        {},
        {},
        {},
        {},
        {"out": 5000},
        {"out": 5000},
        {"out": "torch.int64"},
    )
    assert not harness._only_exchanged_differ(mem, other, "out", want, got)


def test_unwrap_reaches_the_tensor_behind_a_wrapper_with_a_torch_dtype() -> None:
    torch = pytest.importorskip("torch")
    base = torch.zeros(4, dtype=torch.int8)

    class Wrapper:  # triton.reinterpret(t, torch.float8_e5m2): dtype is a torch dtype
        def __init__(self, base: object, dtype: object) -> None:
            self.base, self.dtype = base, dtype

    assert harness._unwrap(Wrapper(base, torch.float8_e5m2)) is base
    assert harness._unwrap(Wrapper(Wrapper(base, torch.float8_e5m2), torch.int8)) is base


def test_scalar_value_encodes_a_narrow_float_scalar_as_its_bits() -> None:
    from conftest import ty

    assert harness._scalar_value(42.0, ty("bf16")).tolist() == 0x4228
    assert harness._scalar_value(42.0, ty("f16")).dtype == np.float16
    assert harness._scalar_value(3, ty("i32")).dtype == np.int32


def test_flat_args_names_the_leaves_of_a_tuple_argument() -> None:
    from collections import namedtuple

    pair = namedtuple("pair", "a b")
    assert harness._flat_args("shape", (4, 8)) == [("shape.0", 4), ("shape.1", 8)]
    assert harness._flat_args("p", pair(1, (2, 3))) == [("p.0", 1), ("p.1.0", 2), ("p.1.1", 3)]
    assert harness._flat_args("x", 7) == [("x", 7)]
    typed = harness._flat_typed("shape", (4, 8), ("i32", "constexpr"))
    assert typed == [("shape.0", 4, "i32"), ("shape.1", 8, "constexpr")]


def test_torch_elem_covers_the_unsigned_integers() -> None:
    assert harness._TORCH_ELEM["torch.uint32"] == ("int", 32, "i32")
    assert harness._TORCH_ELEM["torch.uint16"] == ("int", 16, "i16")


def test_elem_type_reads_a_triton_dtype_spelling() -> None:
    assert harness._elem_type("uint32").name == "i32"
    assert harness._elem_type("tl.float8e5").name == "f8E5M2"
    assert harness._elem_type("torch.bfloat16").name == "bf16"


def test_only_pad_writes_differ_excuses_the_padded_granule_and_nothing_else() -> None:
    class Mem:
        pad_writes = {4096 + 3 * 4: np.float32(4.0).tobytes()}

    record = harness.LaunchRecord(
        launch_id=0,
        fn=None,
        fn_name="k",
        fn_file="",
        grid=(1,),
        args=(),
        kwargs={},
        bound={},
        pre={},
        post={},
        ptrs={"out": 4096},
        bases={"out": 4096},
        elems={"out": "torch.float32"},
    )
    got = np.zeros(8, np.float32)
    want = got.copy()
    want[3] = 4.0  # the device wrote the block's value into the padding
    assert harness._only_pad_writes_differ(Mem(), record, "out", want, got)
    want[5] = 1.0  # any other difference is a mismatch
    assert not harness._only_pad_writes_differ(Mem(), record, "out", want, got)
    want[5] = 0.0
    want[3] = 5.0  # a value the block did not have there is a mismatch too
    assert not harness._only_pad_writes_differ(Mem(), record, "out", want, got)
