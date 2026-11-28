"""Rescaling policies (M4): the IR's selection and parameters, the right depth, the stale lag."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from discovery import bound, fp_export, fptable, fsem, ir, policies, policyrun, problems, semantics


def test_a_program_with_a_parameter_and_a_selection_round_trips():
    prog = policies.program("conditional")
    assert prog.params == {"tau": 8.0}
    assert "tau" not in prog.types
    again = problems.SOFTMAX.parse(prog.text())
    assert again.digest() == prog.digest() and again.params == prog.params


@pytest.mark.parametrize(
    "text",
    [
        "(program p (input x) (param tau) (pass a (state d) (block (d (sum x))) "
        "(merge (d (+ L.d R.d)))) (output elementwise x))",
        "(program p (input x) (param x 1) (pass a (state d) (block (d (sum x))) "
        "(merge (d (+ L.d R.d)))) (output elementwise x))",
        "(program p (input x) (pass a (state d) (block (d (sum x))) "
        "(merge (d (ifgt L.d R.d L.d)))) (output elementwise x))",
    ],
)
def test_bad_parameters_and_selections_are_refused(text):
    with pytest.raises(ir.IRError):
        problems.SOFTMAX.parse(text)


def test_the_conditional_policy_is_exact_over_the_reals():
    prog = policies.program("conditional")
    x = 3 * np.random.default_rng(0).standard_normal((2000, 1))
    ref = problems.SOFTMAX.reference({"x": x})["y"]
    for seed in range(3):
        y = semantics.run(
            prog,
            {"x": x},
            semantics.Arith("float64"),
            semantics.Schedule("random", "tree", 64),
            np.random.default_rng(seed),
        )["y"]
        assert np.max(np.abs(y - ref) / ref) < 1e-12


@pytest.mark.parametrize(("order", "rdepth"), [("left", 1), ("right", 9), ("balanced", 3)])
def test_the_right_depth_of_the_merge_tree(order, rdepth):
    prog = policies.program("always")
    x = np.random.default_rng(0).standard_normal((640, 1)).astype(np.float32)
    r = fsem.run(prog, {"x": x}, "float32", block=64, order=order)
    assert r.blocks == 10 and r.rdepth == rdepth


def test_the_reference_lags_by_tau_per_right_step_and_overflows():
    """The kernel's left fold keeps the lag under τ; a right comb lets it grow with the depth."""
    prog = policies.program("conditional")
    x = policyrun._lag_family(np.random.default_rng(0), 4096, 256, 0.98 * 4)
    left_err, left = policyrun.measure(prog, x, 256, "left", 4.0)
    right_err, right = policyrun.measure(prog, x, 256, "right", 4.0)
    assert left["lag"] <= 4.0 and np.isfinite(left_err)
    assert right["rd"] == 15 and right["lag"] > 4.0 * 10 and np.isfinite(right_err)  # ρ = 15
    x8 = policyrun._lag_family(np.random.default_rng(0), 4096, 256, 0.98 * 8)
    over_err, _ = policyrun.measure(prog, x8, 256, "right", 8.0)
    assert not np.isfinite(over_err)  # e^(8·14.7) overflows float32


def test_policy_statements_are_generated_for_the_three_kinds():
    prog = policies.program("conditional")
    texts = {
        kind: fp_export.export_policy(prog, problems.SOFTMAX, "P", kind).statement_template
        for kind in fp_export.POLICY_KINDS
    }
    assert "znorm (t.eval (block_a tau) (merge_a tau))" in texts["exact"]
    assert "Real.exp (tau * (rdepth t : ℝ))" in texts["overflow"]
    assert "(fblock_a F D tau) (fmerge_a F tau)" in texts["fp"] and "0 ≤ tau →" in texts["fp"]
    fp = fp_export.export_policy(prog, problems.SOFTMAX, "P", "fp")
    assert "(if (R.m - L.m) > tau then R.m else L.m)" in fp.definitions  # the exact merge
    assert (
        "(if r1 > tau then R.m else L.m)" in fp.definitions
    )  # the float merge compares fl(R.m - L.m)


def test_the_canonical_digest_ignores_the_names_a_program_chooses():
    lib = {m: p for m, _, p in fptable.library()}
    assert lib["SoftmaxLseAbs"].canonical_digest() == lib["SoftmaxLseCompact"].canonical_digest()
    assert lib["SoftmaxLseAbs"].digest() != lib["SoftmaxLseCompact"].digest()
    assert lib["SoftmaxOnline"].canonical_digest() != lib["SoftmaxLse"].canonical_digest()
    online = problems.SOFTMAX.known["online"]
    renamed = online.replace("(state m d)", "(state mx sm)").replace(" m)", " mx)")
    renamed = renamed.replace("L.m", "L.mx").replace("R.m", "R.mx").replace("L.d", "L.sm")
    renamed = renamed.replace("R.d", "R.sm").replace("(m (", "(mx (").replace("(d (", "(sm (")
    renamed = renamed.replace("a.m", "a.mx").replace("a.d", "a.sm").replace("(- x m)", "(- x mx)")
    a, b = problems.SOFTMAX.parse(online), problems.SOFTMAX.parse(renamed)
    assert a.digest() != b.digest() and a.canonical_digest() == b.canonical_digest()


def test_the_archive_counts_a_renamed_program_as_a_duplicate():
    from discovery import archive, evaluate

    lib = {m: p for m, _, p in fptable.library()}
    arch = archive.Archive()
    first = arch.insert(evaluate.evaluate(lib["SoftmaxLseAbs"].text(), problems.SOFTMAX, 0))
    again = arch.insert(evaluate.evaluate(lib["SoftmaxLseCompact"].text(), problems.SOFTMAX, 0))
    assert first.status == "new-cell" and again.status == "duplicate"


def test_names_that_could_break_the_statements_are_refused():
    """The review (diary, 12:35 UTC): a name holding `-/` closed the doc comment of the trusted
    statements, and a parameter named like a binder of the statement shadowed it."""
    prog = policies.program("conditional")
    lo = problems.SOFTMAX.parse(policies.POLICIES["conditional"].replace("tau", "lo"))
    with pytest.raises(fp_export.FPExportError, match="binders"):
        fp_export.export_policy(lo, problems.SOFTMAX, "P", "fp")
    bad = dataclasses.replace(prog, name="p -/ #eval IO.println 1 /-")
    text = fp_export.export_policy(bad, problems.SOFTMAX, "P", "exact").statements(bound.parse("u"))
    assert "-/ #eval" not in text and text.count("/-") == text.count("-/")


@pytest.mark.parametrize(
    "out", ["(* (exp (- x a.m)) x)", "(exp (- x (* 2 x)))", "(/ a.d (exp (- x a.m)))"]
)
def test_a_softmax_output_not_of_the_form_exp_x_minus_c_has_no_statement(out):
    """The statements read the normalizer at the element 0, the adversary at the row's maximum:
    the two agree only for outputs exp(x − c)·g with c and g free of x."""
    text = policies.POLICIES["conditional"].replace("(/ (exp (- x a.m)) a.d)", out)
    with pytest.raises(fp_export.FPExportError, match="softmax output"):
        fp_export.export_policy(problems.SOFTMAX.parse(text), problems.SOFTMAX, "P", "exact")


def test_the_policy_is_run_with_its_own_parameter_name():
    text = policies.POLICIES["conditional"].replace("tau", "thr")
    prog = problems.SOFTMAX.parse(text)
    x = policyrun._lag_family(np.random.default_rng(0), 1024, 256, 0.98 * 2)
    err, env = policyrun.measure(prog, x, 256, "left", 2.0)
    assert np.isfinite(err) and env["lag"] <= 2.0


def test_no_statement_docstring_carries_an_invariant_or_a_metric():
    prog = policies.program("conditional")
    for kind in fp_export.POLICY_KINDS:
        text = fp_export.export_policy(prog, problems.SOFTMAX, "P", kind).statements(
            bound.parse("u")
        )
        assert "lag" not in text and "stale" not in text
