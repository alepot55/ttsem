"""The minimiser: spans, well-formedness, and a reduction against a fake pass."""

from __future__ import annotations

from pathlib import Path

import pytest

from ttsem import harness, minimize

_NEEDS_TRITON = pytest.mark.skipif(
    not harness.TRITON_AVAILABLE, reason="needs a Triton install to record a launch"
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_op_spans_cover_region_ops_as_one_span() -> None:
    text = (FIXTURES / "p3.ttir.generic").read_text()
    lines = text.split("\n")
    spans = minimize.op_spans(lines)
    assert spans, "no ops found"
    multi = [s for s in spans if s[1] > s[0]]
    assert multi, "expected at least one op with a region"
    for first, last, _ in multi:
        assert lines[first].rstrip().endswith("{") or "({" in lines[first]
        assert "}" in lines[last]


def test_well_formed_rejects_a_dangling_use() -> None:
    text = (FIXTURES / "p1.ttir.generic").read_text()
    assert minimize.well_formed(text)
    lines = text.split("\n")
    # drop the first op that defines a value: something downstream uses it
    idx = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith("%"))
    assert not minimize.well_formed("\n".join(lines[:idx] + lines[idx + 1 :]))


@_NEEDS_TRITON
def test_pass_argument_and_binding_names() -> None:
    assert (
        minimize.pass_argument("Before TritonGPUFuseNestedLoops (tritongpu-fuse-nested-loops)")
        == "tritongpu-fuse-nested-loops"
    )
    assert minimize.pass_argument("Before Canonicalizer (canonicalize)") == "canonicalize"
    add = minimize.binding_for("tritongpu-fuse-nested-loops")
    assert add.__name__ == "add_fuse_nested_loops"
    assert minimize.binding_for("canonicalize").__name__ == "add_canonicalizer"


@_NEEDS_TRITON
def test_reduce_deletes_the_dead_op_and_keeps_the_one_the_fake_pass_rewrites() -> None:
    program = FIXTURES / "p1.py"
    record = harness.capture_launches(program, "cpu")[0]
    generic = (FIXTURES / "p1.ttir.generic").read_text()
    assert generic.count('"arith.remsi"') >= 2
    # p1 is one straight chain into its store, so nothing in it can go; plant a dead op
    lines = generic.split("\n")
    idx = next(i for i, ln in enumerate(lines) if '"tt.make_range"' in ln)
    dead = lines[idx].replace(lines[idx].split("=")[0].strip(), "%dead", 1)
    planted = "\n".join(lines[: idx + 1] + [dead] + lines[idx + 1 :])
    assert minimize.well_formed(planted)

    def fake_pass(text: str) -> str:
        # the last remainder feeds the stored value; turning it into a sum changes the output
        head, _, tail = text.rpartition('"arith.remsi"')
        return head + '"arith.addi"' + tail

    red = minimize.reduce(record, planted, fake_pass)
    # the dead op goes, and the load with the address chain behind it becomes a constant
    assert red.lines_to < red.lines_from - 1
    assert "%dead" not in red.before and '"tt.load"' not in red.before
    assert '"arith.remsi"' in red.before and '"arith.addi"' in red.after
    assert minimize.meaning_changed(record, red.before, red.after)
    assert not minimize.meaning_changed(record, red.before, red.before)


def test_strip_locs_keeps_the_program() -> None:
    generic = (FIXTURES / "p3.ttir.generic").read_text()
    assert minimize.well_formed(minimize.strip_locs(generic))
    nested = 'x loc(callsite("/a/b.py":1:2 at "acc"("acc"("/c/d.py":3:4)))) loc(unknown) tail'
    assert minimize.strip_locs(nested) == "x tail"
    aliased = '#loc = loc(unknown)\n#loc1 = loc("f.py":3:4)\nmodule {\n  op loc(#loc1)\n}'
    assert minimize.strip_locs(aliased) == "module {\n  op\n}"


def test_zero_constants_by_type() -> None:
    assert minimize._zero_of("i32") == "0 : i32"
    assert minimize._zero_of("i1") == "false : i1"
    assert minimize._zero_of("f32") == "0.000000e+00 : f32"
    assert minimize._zero_of("tensor<64xi1>") == "dense<false> : tensor<64xi1>"
    enc = "tensor<64x32xbf16, #ttg.blocked<{sizePerThread = [1, 1]}>>"
    assert minimize._zero_of(enc) == f"dense<0.000000e+00> : {enc}"
    assert minimize._zero_of("!tt.ptr<f32>") is None
    assert minimize._zero_of("tensor<8x!tt.ptr<f32>>") is None
    ptr, res = "tensor<8x!tt.ptr<f32>>", "tensor<8xf32>"
    line = f'  %v = "tt.load"(%p) <{{cache = 1 : i32}}> : ({ptr}) -> {res} loc(unknown)'
    want = f'  %v = "arith.constant"() <{{value = dense<0.000000e+00> : {res}}}> : () -> {res}'
    assert minimize.constant_for([line], 0, 0) == want
    assert minimize.constant_for(['    %a:2 = "tt.x"() : () -> (i32, i32)'], 0, 0) is None


@_NEEDS_TRITON
def test_reads_changed_sees_a_load_the_source_never_did(tmp_path: Path) -> None:
    program = FIXTURES / "p1.py"
    record = harness.capture_launches(program, "cpu")[0]
    generic = (FIXTURES / "p1.ttir.generic").read_text()
    assert not minimize.reads_changed(record, generic, generic)
    # drop the mask of the load: the after-module reads the odd addresses the source skipped
    lines = generic.split("\n")
    idx = next(i for i, ln in enumerate(lines) if '"tt.load"' in ln)
    assert lines[idx].count("%") >= 3, lines[idx]
    unmasked = lines[idx]
    head, args = unmasked.split('"tt.load"(', 1)
    operands, rest = args.split(")", 1)
    first = operands.split(",")[0]
    sig_head, sig = rest.split(" : (", 1)
    ptr_ty = sig.split(",")[0]
    result = sig.split(") -> ", 1)[1]
    sig_head = sig_head.replace(
        "operandSegmentSizes = array<i32: 1, 1, 1>", "operandSegmentSizes = array<i32: 1, 0, 0>"
    )
    lines[idx] = f'{head}"tt.load"({first}){sig_head} : ({ptr_ty}) -> {result}'
    after = "\n".join(lines)
    assert minimize.well_formed(after)
    assert minimize.reads_changed(record, generic, after)
    assert not minimize.reads_changed(record, after, generic)


def test_crash_mode_keeps_what_makes_the_pass_fail() -> None:
    generic = (FIXTURES / "p1.ttir.generic").read_text()

    def brittle_pass(text: str) -> str:
        if '"tt.load"' in text:
            raise RuntimeError("the pass asserts on a load")
        return text

    red = minimize.reduce(None, generic, brittle_pass, crash=True)
    assert '"tt.load"' in red.before and red.lines_to < red.lines_from
    assert minimize.well_formed(red.before)
    assert minimize.pass_argument("tritongpu-fuse-nested-loops") == "tritongpu-fuse-nested-loops"


def test_crash_mode_keeps_the_same_failure_not_any_failure() -> None:
    """A candidate that fails for another reason (a module the verifier rejects, say) is not
    the crash being reduced: the first failure's signature is the one every survivor has."""
    generic = (FIXTURES / "p1.ttir.generic").read_text()

    def brittle_pass(text: str) -> str:
        if '"tt.store"' not in text:
            raise RuntimeError("<stage>:4:5: error: the verifier rejects a kernel with no store")
        if '"tt.load"' in text:
            raise RuntimeError("PLEASE submit a bug report: the pass segfaults on a load")
        return text

    red = minimize.reduce(None, generic, brittle_pass, crash=True)
    assert '"tt.load"' in red.before and '"tt.store"' in red.before
    assert red.lines_to < red.lines_from


def test_crash_mode_asks_the_verifier_about_every_candidate() -> None:
    generic = (FIXTURES / "p1.ttir.generic").read_text()
    asked: list[str] = []

    def brittle_pass(text: str) -> str:
        if '"tt.load"' in text:
            raise RuntimeError("the pass asserts on a load")
        return text

    def verifier(text: str) -> None:
        asked.append(text)
        if '"tt.store"' not in text:
            raise RuntimeError("no store")

    red = minimize.reduce(None, generic, brittle_pass, crash=True, verify=verifier)
    assert asked and '"tt.store"' in red.before and '"tt.load"' in red.before


def test_failure_signatures_ignore_positions_and_numbers() -> None:
    a = minimize.failure_signature("<stage>:12:7: error: 'tt.load' op operand #0 must be ptr")
    b = minimize.failure_signature("<stage>:4:31: error: 'tt.load' op operand #1 must be ptr")
    assert a == b
    assert a != minimize.failure_signature("PLEASE submit a bug report to https://github.com/")
