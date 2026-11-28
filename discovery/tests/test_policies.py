"""Rescaling policies (M4): the IR's selection and parameters, the right depth, the stale lag."""

from __future__ import annotations

import numpy as np
import pytest

from discovery import fp_export, fsem, ir, policies, policyrun, problems, semantics


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
