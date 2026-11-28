"""The IR, its numpy semantics and the check cascade, on the problems' seeds and known forms."""

from __future__ import annotations

import numpy as np
import pytest

from discovery import ir, semantics
from discovery.evaluate import evaluate
from discovery.problems import PROBLEMS, SOFTMAX, VARIANCE

ALL = [
    (p.name, name, text)
    for p in PROBLEMS.values()
    for name, text in {"seed": p.seed, **p.known}.items()
]


@pytest.mark.parametrize(("problem", "name", "text"), ALL, ids=[f"{a}-{b}" for a, b, _ in ALL])
def test_parses_and_round_trips(problem, name, text):
    prog = PROBLEMS[problem].parse(text)
    again = PROBLEMS[problem].parse(prog.text())
    assert again.text() == prog.text()
    assert again.digest() == prog.digest()


@pytest.mark.parametrize(
    ("problem", "name", "stage", "cell"),
    [
        ("softmax", "seed", "passed", (3, 2, "stable")),
        ("softmax", "online", "passed", (2, 2, "stable")),
        ("softmax", "unshifted", "passed", (2, 1, "nonfinite")),
        ("softmax", "no-rescale", "exact", None),
        ("softmax", "log-sum-exp", "passed", (2, 1, "stable")),
        ("attention", "seed", "passed", (3, 3, "stable")),
        ("attention", "online", "passed", (1, 3, "stable")),
        ("variance", "seed", "passed", (2, 3, "stable")),
        ("variance", "welford", "passed", (1, 3, "stable")),
        ("variance", "sum-of-squares", "passed", (1, 3, "inaccurate")),
    ],
)
def test_cascade(problem, name, stage, cell):
    p = PROBLEMS[problem]
    v = evaluate(p.seed if name == "seed" else p.known[name], p, seed=3)
    assert v.stage == stage, v.message
    if cell is not None:
        assert v.cell() == cell


def test_fewer_passes_cost_less():
    seed = evaluate(SOFTMAX.seed, SOFTMAX)
    online = evaluate(SOFTMAX.known["online"], SOFTMAX)
    assert online.cost["time_us"] < seed.cost["time_us"]
    assert online.cost["bytes_per_elem"] == 12 and seed.cost["bytes_per_elem"] == 16


def test_counterexample_is_small_and_readable():
    v = evaluate(SOFTMAX.known["no-rescale"], SOFTMAX)
    assert "wrong result for N=" in v.message and "expected y=" in v.message


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        (
            "(program p (input x) (pass a (state m) (block (m x)) (merge (m L.m))) "
            "(output elementwise x))",
            "only inside",
        ),
        (
            "(program p (input x) (pass a (state m) (block (m (rmax x))) "
            "(merge (m (max L.m R.m)))) (output elementwise (/ x b.d)))",
            "unknown name `b.d`",
        ),
        (
            "(program p (input x) (pass a (state m d) (block (m (rmax x))) (merge (m L.m))) "
            "(output elementwise x))",
            "must bind every state field",
        ),
        (
            "(program p (input x) (pass a (state m) (block (m (sum (rmax x)))) (merge (m L.m))) "
            "(output elementwise x))",
            "not inside another reduction",
        ),
        (
            "(program p (input x) (pass a (state m) (block (m (frob x))) (merge (m L.m))) "
            "(output elementwise x))",
            "unknown operator",
        ),
        ("(program p (input x) (pass a (state m) (block (m (rmax x)))", "unclosed"),
        (
            "(program p (input x) (pass a (state m) (block (m (rmax x)) (m (rmin x))) "
            "(merge (m L.m))) (output elementwise x))",
            "binds a name twice",
        ),
        (
            "(program p (input x) (pass a (state m) (block (x (rmax x)) (m x)) "
            "(merge (m L.m))) (output elementwise x))",
            "cannot be the name of a temporary",
        ),
        (
            "(program p (input x) (pass a (state d) (block (m (rmax x)) (d (sum (exp (- x m))))) "
            "(merge (d (+ L.d (* R.d (exp L.m)))))) (output elementwise x))",
            "unknown name `L.m`",
        ),
    ],
)
def test_errors_name_the_problem(text, fragment):
    with pytest.raises(ir.IRError, match=None) as err:
        SOFTMAX.parse(text)
    assert fragment in str(err.value)


def test_wrong_outputs_are_rejected():
    with pytest.raises(ir.IRError, match="output final"):
        VARIANCE.parse(SOFTMAX.known["online"])


def test_any_blocking_and_tree_agree_for_an_associative_merge():
    prog = SOFTMAX.parse(SOFTMAX.known["online"])
    rng = np.random.default_rng(0)
    el = SOFTMAX.generate(rng, 50, "normal")
    ref = SOFTMAX.reference(el)["y"]
    for kind in ("one", "single", "fixed", "random"):
        for order in ("left", "right", "tree"):
            out = semantics.run(prog, el, schedule=semantics.Schedule(kind, order, 7), rng=rng)["y"]
            np.testing.assert_allclose(out, ref, rtol=1e-12)


def test_bfloat16_rounding():
    a = np.array([1.0, 1.0 + 2**-8, 1.0 + 3 * 2**-9, 3.0e38, np.inf], dtype=np.float32)
    r = semantics.round_bf16(a)
    assert r[0] == 1.0 and r[1] == 1.0  # ties to even
    assert r[2] == np.float32(1.0 + 2**-7)
    assert np.isinf(r[4])


# --------------------------------------------------------------------------- the review's cases

BIG = VARIANCE.known["welford"].replace(
    "(q (+ L.q R.q (* (- R.mu L.mu) (- R.mu L.mu) (/ (* L.n R.n) n))))",
    "(q (+ L.q R.q (* (- R.mu L.mu) (- R.mu L.mu) (/ (* L.n R.n) n)) "
    "(max 0 (- (min L.n R.n) 128))))",
)
CLAMP = """(program clamp (input x)
  (pass a (state d) (block (d (sum (exp (max (min x 88) -87))))) (merge (d (+ L.d R.d))))
  (output elementwise (/ (exp (max (min x 88) -87)) a.d)))"""
TIE = """(program tie (input x)
  (pass a
    (state n mu q)
    (block (n (sum 1)) (mu (/ (sum x) n)) (q (sum (* (- x mu) (- x mu)))))
    (merge (n (+ L.n R.n))
           (dm (- R.mu L.mu))
           (mu (+ L.mu (/ (* dm dm R.n) (* n dm))))
           (q (+ L.q R.q (* dm dm (/ (* L.n R.n) n))))))
  (output final (mean a.mu) (var (/ a.q a.n))))"""
LEFT_SHIFT = """(program left-shift (input x)
  (pass a
    (state m d)
    (block (m (rmax x)) (d (sum (exp (- x m)))))
    (merge (m L.m) (d (+ L.d (* R.d (exp (- R.m L.m)))))))
  (output elementwise (/ (exp (- x a.m)) a.d)))"""


def test_a_merge_wrong_only_on_large_blocks_is_caught():
    assert BIG != VARIANCE.known["welford"]
    v = evaluate(BIG, VARIANCE)
    assert v.stage == "exact"
    assert "blocks of 2048" in v.message or "large blocks" in v.message


def test_wrong_over_the_reals_on_large_inputs_is_caught():
    v = evaluate(CLAMP, SOFTMAX)
    assert v.stage == "exact" and "inputs" in v.message


def test_repeated_values_catch_a_merge_that_divides_by_zero():
    v = evaluate(TIE, VARIANCE)
    assert v.stage == "exact"
    assert any(f in v.message for f in ("zeros", "threes", "small-int"))


def test_exact_but_ill_conditioned_forms_still_pass():
    v = evaluate(VARIANCE.known["sum-of-squares"], VARIANCE)
    assert v.stage == "passed" and v.fp_class == "inaccurate"


def test_homomorphism_flag():
    assert evaluate(SOFTMAX.known["online"], SOFTMAX).homomorphic
    v = evaluate(LEFT_SHIFT, SOFTMAX)  # right end to end over the reals, not a homomorphism
    assert v.stage == "passed" and not v.homomorphic and "cannot be proved" in v.message


@pytest.mark.parametrize("literal", ["inf", "-inf", "+inf", "nan", "1e999"])
def test_non_finite_literals_are_rejected(literal):
    text = SOFTMAX.seed.replace("(m (rmax x))", f"(m (max (rmax x) {literal}))")
    v = evaluate(text, SOFTMAX)
    assert v.stage == "parse"


def test_a_pass_that_reads_no_element_is_not_a_pass_over_memory():
    counted = """(program c (input x)
  (pass a (state n) (block (n (sum 1))) (merge (n (+ L.n R.n))))
  (pass b (state t) (block (t (sum x))) (merge (t (+ L.t R.t))))
  (output final (mean (/ b.t a.n)) (var (* 0 b.t))))"""
    v = evaluate(counted, VARIANCE)
    assert v.passes == 1  # pass a only counts; set before the exact tests run


def test_floating_point_rows_do_not_depend_on_the_program():
    a = evaluate(SOFTMAX.seed, SOFTMAX, seed=5).fp
    b = evaluate(SOFTMAX.known["online"], SOFTMAX, seed=5).fp
    # the same rows: the spread family's float32 errors agree within rounding noise
    assert abs(a["spread"]["float32"] - b["spread"]["float32"]) < 1e-6
