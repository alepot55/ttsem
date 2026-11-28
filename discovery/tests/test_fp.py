"""Certified floating-point bounds: the grammar of B, the generated statements, the judge."""

from __future__ import annotations

import math
import shutil

import numpy as np
import pytest

from discovery import adversary, bound, fp_export, fpbound, fptable, fsem, problems, prove


def _lib(module: str):
    return {m: prog for m, _, prog in fptable.library()}[module]


# --------------------------------------------------------------------------- the grammar


def test_a_bound_parses_evaluates_and_prints_in_lean():
    b = bound.parse("3*u*(d + D) + 2*u*R*(d+1) + ue^2 + 1/2*etae")
    env = {"u": 1.0, "ue": 2.0, "etae": 4.0, "d": 3.0, "D": 5.0, "R": 10.0}
    assert b.value(env) == pytest.approx(3 * 8 + 2 * 10 * 4 + 4 + 2)
    assert b.atoms() == {"u", "d", "D", "R", "ue", "etae"}
    lean = b.lean()
    assert "F.u" in lean and "(t.depth : ℝ)" in lean and "(hi - lo)" in lean
    assert "((1 : ℝ) / 2)" in lean and "(F.ue ^ 2)" in lean


@pytest.mark.parametrize(
    "text",
    ["u - ue", "u / d", "foo * u", "u ^ 9", "", "u +", "(u", "2 ** 1.5", "u * 1e5"],
)
def test_a_bound_outside_the_grammar_is_refused(text):
    with pytest.raises(bound.BoundError):
        bound.parse(text)


def test_the_bound_is_read_from_the_last_line_of_the_reply():
    reply = "Reasoning...\nB = u\nthen better:\nB = 2*u*d"
    assert fpbound.parse_bound_reply(reply).text == "2*u*d"
    with pytest.raises(bound.BoundError):
        fpbound.parse_bound_reply("no bound here")


# --------------------------------------------------------------------------- the statements


def test_the_generated_relations_follow_the_program():
    fp = fp_export.export(_lib("SoftmaxOnline"), problems.SOFTMAX, "SoftmaxOnline")
    d = fp.definitions
    assert "r1 = lmax A" in d
    assert "Discovery.fsum F D (fun x y => ∃ q1 q2, F.add (x - v_m) q1 ∧ F.exp q1 q2" in d
    assert "F.mul (L.d * r2) r3" in d and "F.add (r3 + r6) r7" in d
    assert "znorm (st : State_a) : ℝ := 1 / ((Real.exp ((0 : ℝ) - st.m)) / st.d)" in d
    lse = fp_export.export(_lib("SoftmaxLse"), problems.SOFTMAX, "SoftmaxLse").definitions
    assert (
        "F.log r2 r3" in lse and "F.add (v_m + r3) r4" in lse
    )  # a reduction inside (+ m (log ...))


def test_only_programs_with_an_exact_constant_have_a_statement():
    prog = problems.SOFTMAX.parse(
        "(program p (input x) (pass a (state d) (block (d (sum (* 0.1 (exp x))))) "
        "(merge (d (+ L.d R.d)))) (output elementwise (/ (exp x) a.d)))"
    )
    with pytest.raises(fp_export.FPExportError):
        fp_export.export(prog, problems.SOFTMAX, "X")


def test_the_prover_sees_the_tools_but_not_the_classical_bound():
    text = fpbound.framework_text()
    for tool in ("depth_law", "fsum_bounds", "exp_bounds", "mul_err", "lmax_append"):
        assert tool in text
    for hidden in ("sum_tree_bound", "sum_tree_error", "posBlock"):
        assert hidden not in text


def test_the_bound_prompt_states_no_bound():
    fp = fp_export.export(_lib("SoftmaxOnline"), problems.SOFTMAX, "SoftmaxOnline")
    prompt = fpbound.bound_prompt(fp, problems.SOFTMAX, [])
    assert "depth_law" not in prompt and "fsum_bounds" not in prompt
    assert "(1 + F.u) ^" not in prompt and "γ" not in prompt
    assert prompt.rstrip().endswith("B = <expression>")


# --------------------------------------------------------------------------- measuring


def test_the_log_normalizer_error_matches_the_relative_one():
    prog = _lib("SoftmaxOnline")
    x = 100 + np.random.default_rng(3).standard_normal(1024)
    err, env = fpbound.log_normalizer_error(prog, x, 128, "left")
    rel = fsem.measure("softmax", prog, {"x": x[:, None]}, "float32", exact=False, block=128)
    assert err / fsem.FORMATS["float32"].u == pytest.approx(rel["normalizer"], rel=1e-3, abs=1e-3)
    assert env["d"] == 7 and env["D"] == 7 and env["n"] == 1024
    assert env["R"] == pytest.approx(float(np.ptp(np.float32(x))))


def test_a_bound_too_small_is_broken_by_the_adversary():
    prog = _lib("SoftmaxOnline")
    found = fpbound.attack(prog, bound.parse("u"), steps=5, seeds=(0,))
    assert found["ratio"] > 1 and math.isfinite(found["error"])


def test_the_attention_is_measured_one_component_at_a_time():
    prog = _lib("AttentionOnline")
    el = adversary.generate("attention", "normal0", np.random.default_rng(0), 512, 64)
    runs = fpbound.measure("attention", prog, el, 64, "left")
    assert len(runs) == problems.ATT_DIM
    for err, env in runs:
        assert (
            err >= 0 and env["S"] > 0 and env["V"] == pytest.approx(np.max(np.abs(el["v"])), 1e-6)
        )
    b = bound.parse("2*u*(d+D)*S + n*etae*V", "attention")
    q, info = fpbound.ratio("attention", prog, b, el)
    assert 0 <= q < 1 and info["S"] > 0


def test_the_variance_bound_reads_the_exact_variance():
    prog = _lib("VarianceSumsq")
    el = adversary.generate("variance", "offset", np.random.default_rng(0), 1024, 128)
    err, env = fpbound.measure("variance", prog, el, 128, "left")
    assert env["S2"] == pytest.approx(0.01, rel=0.2) and env["Q"] == pytest.approx(1e8, rel=1e-3)
    assert err > 1e-4  # the sum of squares cancels at a mean of 1e4


def test_statement_version_3_assumes_small_depth_and_small_absolute_errors():
    for mod, prob in (("SoftmaxOnline", "softmax"), ("AttentionOnline", "attention")):
        fp = fp_export.export(_lib(mod), problems.PROBLEMS[prob], mod)
        assert fp_export.DEEP in fp.statement_template and fp_export.TINY in fp.statement_template
    att = fp_export.export(_lib("AttentionOnline"), problems.ATTENTION, "AttentionOnline")
    assert "(∀ e ∈ X, |e.2| ≤ Vmax)" in att.statement_template
    lean_b = bound.parse("S*u + V*n*etae", "attention").lean()
    assert lean_b == "(((attnScale X) * F.u) + (Vmax * (X.length : ℝ) * F.ηe))"
    with pytest.raises(bound.BoundError):
        bound.parse("S*u")  # S belongs to the attention only


# --------------------------------------------------------------------------- the judge

lean = pytest.mark.skipif(
    shutil.which("lake") is None and not prove.ELAN_DEFAULT.exists(), reason="no Lean toolchain"
)


def test_the_judge_refuses_a_definition_before_lean_runs(tmp_path):
    fp = fp_export.export(_lib("SoftmaxOnline"), problems.SOFTMAX, "SoftmaxOnline")
    v = fpbound.judge(fp, bound.parse("u"), "def C := 1", "sorry", tmp_path)
    assert not v.ok and v.stage == "deny" and v.seconds == 0.0


@lean
def test_the_judge_compiles_the_statements_and_rejects_a_wrong_proof(tmp_path):
    fp = fp_export.export(_lib("SoftmaxLse"), problems.SOFTMAX, "SoftmaxLse")
    v = fpbound.judge(fp, bound.parse("u"), "-- none", "intro F\n  exact le_refl _", tmp_path)
    assert not v.ok and v.stage == "compile" and v.errors


def test_the_deny_list_reads_unicode_identifiers_as_lean_does():
    """An ASCII-only word pattern split `hφdef` into `h` and `def` and refused a correct proof."""
    assert prove.banned("set φ := a - b with hφdef", helpers=True) == []
    assert prove.banned("have h₀def := 1") == []
    assert prove.banned("def C := 1", helpers=True) == ["def"]
    assert prove.banned("exact x def y") == ["def"]


def test_a_word_glued_to_a_greek_letter_is_one_identifier_only_where_lean_says_so():
    """The review (diary, 12:35 UTC): treating every non-ASCII letter as part of an identifier let
    `λdef` hide a `def`. Lean's identifiers take most Greek letters but not λ, Π or Σ."""
    assert prove.banned("hφdef") == [] and prove.banned("αsorry") == []
    assert prove.banned("λdef", helpers=True) == ["def"]
    assert prove.banned("Πdef", helpers=True) == ["def"]
    assert prove.banned("x·def", helpers=True) == ["def"]
    assert "sorry" in prove.banned("éaxiom sorry")


def test_a_run_that_overflows_is_skipped_when_a_bound_is_attacked(monkeypatch):
    env = {**fpbound.ENV32, "D": 8.0, "d": 4.0, "n": 4096.0, "R": 1.0, "M": 1.0}
    runs = iter([(math.inf, env)] + [(1e-7, env)] * 100)
    monkeypatch.setattr(fpbound, "measure", lambda *a, **k: next(runs))
    q, info = fpbound.ratio("softmax", None, bound.parse("u"), {})
    assert math.isfinite(q) and info["error"] == 1e-7


def test_an_attention_run_that_overflows_still_reports_its_atoms():
    prog = _lib("AttentionOnline")
    el = adversary.generate("attention", "normal0", np.random.default_rng(0), 256, 64)
    el["v"] = np.full_like(el["v"], 3e38)  # the weighted sum overflows float32
    got = fpbound.measure("attention", prog, el, 64, "left")
    err, env = got if isinstance(got, tuple) else got[0]
    assert not math.isfinite(err) and env["S"] > 0 and env["V"] > 0


def test_the_random_merge_tree_is_drawn_from_the_climb():
    prog = _lib("SoftmaxOnline")
    el = adversary.generate("softmax", "normal0", np.random.default_rng(0), 4096, 64)
    rng = np.random.default_rng(5)
    depths = {fpbound.measure("softmax", prog, el, 64, "random", rng)[1]["d"] for _ in range(6)}
    assert len(depths) > 1
