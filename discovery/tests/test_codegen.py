"""The kernel tier: every seed and known form as a Triton kernel on CUDA, against the reference.

The exact forms match ``problem.reference`` (float64) on one element, a row shorter than a block,
one element either side of a block, and several blocks with a tail. On every input family the
kernel lands in the floating-point class the semantics measured there, so the forms classed
non-finite or inaccurate fail on the GPU where the semantics saw them fail. A merge that is
wrong over the reals is right on one block and wrong from the second, so the kernel does merge
the blocks with the program's merge. Programs that use every operator and every mix of shapes
agree with the numpy semantics folded over the same blocks, names Python cannot take are
renamed, the header of every kernel names the float32 instructions its PTX uses, the GPU check
lists the shapes it ran, and ttsem's check reports itself unavailable on Windows.

The timings: every do_bench call comes after a warm-up, inside the GPU's lock; a timing is the
median of several, with their range and run counts; a call that timed too few runs is redone.
"""

from __future__ import annotations

import functools
import math
import re
import sys
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

from discovery import codegen_triton, gpu, semantics, ttsem_check  # noqa: E402
from discovery.evaluate import FP_TOL, Verdict, evaluate  # noqa: E402
from discovery.problems import PROBLEMS, SOFTMAX, Problem  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

FORMS = [(p.name, name) for p in PROBLEMS.values() for name in ("seed", *p.known)]
IDS = [f"{p}-{n}" for p, n in FORMS]
TOL = {"softmax": 1e-4, "attention": 1e-4, "variance": 1e-3}  # normwise, normwise, relative
ROWS, DIM, LONG = 3, 64, 5000

# programs that are not any operator, for the translation: every operator, a constant state
# field, constants to fold, a negative number, and a row vector meeting element scalars
UNUSUAL = {
    "softmax": """(program ops
  (input x)
  (pass a
    (state lo hi c)
    (block (lo (rmin x)) (hi (rmax (abs x))) (c 2))
    (merge (lo (min L.lo R.lo)) (hi (max L.hi R.hi)) (c (* 0.5 (+ L.c R.c)))))
  (pass b (state z) (block (z (sum (sqrt (- x (- a.lo 1)))))) (merge (z (+ L.z R.z))))
  (output elementwise
    (+ (neg (log (/ (- x (- a.lo 1)) a.hi))) (* (exp 0) (/ 1 4) a.c b.z) -2)))""",
    "variance": """(program fin
  (input x)
  (pass a
    (state n lo)
    (block (n (sum 1)) (lo (rmin (neg x))))
    (merge (n (+ L.n R.n)) (lo (min L.lo R.lo))))
  (output final (mean (- a.lo (neg a.n))) (var (sqrt (abs a.lo)))))""",
    "attention": """(program mix
  (input s v)
  (pass a
    (state m o)
    (block (m (rmax s)) (o (sum (* (exp (- s m)) v))))
    (merge (m (max L.m R.m)) (o (+ (* L.o (exp (- L.m m))) (* R.o (exp (- R.m m)))))))
  (pass b
    (state w)
    (block (w (sum (* (+ s (neg a.o)) (exp (- s a.m))))))
    (merge (w (+ L.w R.w))))
  (output final (o (/ b.w (+ (abs a.m) 1)))))""",
}


def _text(problem: str, name: str) -> str:
    p = PROBLEMS[problem]
    return p.seed if name == "seed" else p.known[name]


@functools.cache
def _verdict(problem: str, name: str) -> Verdict:
    return evaluate(_text(problem, name), PROBLEMS[problem], seed=3)


def _sizes(p: Problem) -> list[int]:
    block = codegen_triton.block_for(p)
    return [1, 7, block - 1, block, block + 1, LONG]


def _run(p: Problem, text: str, n: int, family: str = "normal") -> list[tuple[dict, dict]]:
    """The kernel of ``text`` on ROWS rows of ``n`` elements: for every row, its outputs and its
    elements (as stored in float32), in float64."""
    module = gpu.kernel(p.parse(text), p)
    shape = (ROWS, n) if codegen_triton.layout(p) == "rows" else (ROWS, n, DIM)
    inputs = gpu.make_inputs(p, shape, family, seed=n)
    out = {k: v.double().cpu().numpy() for k, v in module.run(inputs).items()}
    if codegen_triton.layout(p) == "rows":
        x = {f: t.double().cpu().numpy() for f, t in inputs.items()}
        elems = [{f: a[i][:, None] for f, a in x.items()} for i in range(ROWS)]
    else:
        q, k, v = (inputs[name].double().cpu().numpy() for name in "qkv")
        s = q @ k.T / math.sqrt(q.shape[1])
        elems = [{"s": s[i][:, None], "v": v} for i in range(ROWS)]
    return [({k: v[i] for k, v in out.items()}, el) for i, el in enumerate(elems)]


def _errors(problem: str, text: str, n: int, family: str = "normal") -> list[float]:
    """The kernel's error on each row against ``problem.reference``, by the problem's measure."""
    p = PROBLEMS[problem]
    return [p.error(out, p.reference(el), el) for out, el in _run(p, text, n, family)]


def _fp_class(err: float) -> str:
    return "nonfinite" if not math.isfinite(err) else "inaccurate" if err > FP_TOL else "stable"


# every program the tests compile, for the header's claims about rounding
ROUNDING_CASES = [(p, _text(p, n)) for p, n in FORMS] + sorted(UNUSUAL.items())
ROUNDING_IDS = IDS + [f"{p}-unusual" for p in sorted(UNUSUAL)]
# the float32 arithmetic of a kernel's PTX, and the instructions the header names
F32_OPS = re.compile(r"\b(?:add|sub|mul|fma|div|rcp|sqrt|rsqrt|ex2|lg2)\.[a-z.]*f32\b")
NAMED_F32_OPS = {
    "add.rn.f32",
    "sub.rn.f32",
    "mul.rn.f32",
    "div.full.f32",
    "ex2.approx.f32",
    "sqrt.approx.ftz.f32",
    "fma.rn.ftz.f32",
}


def _ptx(module) -> list[str]:
    """The PTX of every variant of the module's kernel compiled so far (Triton 3.8's caches)."""
    jit = next(v for v in vars(module).values() if isinstance(v, triton.runtime.jit.JITFunction))
    return [k.asm["ptx"] for cache in jit.device_caches.values() for k in cache[0].values()]


@cuda
@pytest.mark.parametrize(("problem", "name"), FORMS, ids=IDS)
def test_exact_forms_match_the_reference(problem, name):
    if _verdict(problem, name).stage != "passed":
        pytest.skip("not exact over the reals (see test_a_wrong_merge_is_wrong_from_block_two)")
    text = _text(problem, name)
    worst = {n: max(_errors(problem, text, n)) for n in _sizes(PROBLEMS[problem])}
    assert all(e <= TOL[problem] for e in worst.values()), worst


@cuda
@pytest.mark.parametrize(("problem", "name"), FORMS, ids=IDS)
def test_the_floating_point_class_carries_over(problem, name):
    v = _verdict(problem, name)
    if v.stage != "passed":
        pytest.skip("no floating-point class: not exact over the reals")
    text = _text(problem, name)
    semantics_class = {fam: _fp_class(r["float32"]) for fam, r in v.fp.items()}
    kernel_class = {fam: _fp_class(max(_errors(problem, text, LONG, fam))) for fam in v.fp}
    assert kernel_class == semantics_class


@cuda
def test_a_wrong_merge_is_wrong_from_block_two():
    block = codegen_triton.block_for(SOFTMAX)
    text = SOFTMAX.known["no-rescale"]
    worst = {n: max(_errors("softmax", text, n)) for n in _sizes(SOFTMAX)}
    assert all(e <= TOL["softmax"] for n, e in worst.items() if n <= block), worst
    assert all(e > 1e-2 for n, e in worst.items() if n > block), worst


@cuda
@pytest.mark.parametrize("problem", sorted(UNUSUAL))
def test_every_operator_and_shape_translates(problem):
    p = PROBLEMS[problem]
    prog = p.parse(UNUSUAL[problem])
    blocks = semantics.Schedule("fixed", "left", codegen_triton.block_for(p))
    worst = {}
    for n in _sizes(p):
        errs = []
        for out, el in _run(p, UNUSUAL[problem], n):
            want = semantics.run(prog, el, schedule=blocks)
            for k, w in want.items():
                a, w = np.asarray(out[k], dtype=np.float64), np.asarray(w, dtype=np.float64)
                errs.append(float(np.max(np.abs(a - w)) / np.max(np.abs(w))))
        worst[n] = max(errs)
    assert all(e <= 1e-5 for e in worst.values()), worst


@cuda
def test_names_python_cannot_take_are_renamed():
    # the online softmax, with names that are keywords, the kernel's own names, or not
    # identifiers at all
    text = """(program row
  (input x)
  (pass tl
    (state mask if)
    (block (mask (rmax x)) (if (sum (exp (- x mask)))))
    (merge (mask (max L.mask R.mask))
           (if (+ (* L.if (exp (- L.mask mask))) (* R.if (exp (- R.mask mask)))))))
  (output elementwise (/ (exp (- x tl.mask)) tl.if)))"""
    source = codegen_triton.generate(SOFTMAX.parse(text), SOFTMAX)
    assert "def row_2(" in source and "tl_if = " in source
    worst = {n: max(_errors("softmax", text, n)) for n in _sizes(SOFTMAX)}
    assert all(e <= TOL["softmax"] for e in worst.values()), worst


def test_ttsem_check_reports_unavailable_on_windows():
    result = ttsem_check.check(SOFTMAX.parse(SOFTMAX.known["online"]), SOFTMAX)
    if sys.platform == "win32":
        assert result["verdict"] == "unavailable", result
        assert "generic form" in result["message"]
    else:  # where ttsem runs, the masked kernel is clean
        assert result["verdict"] == "ok", result


@cuda
@pytest.mark.parametrize(("problem", "text"), ROUNDING_CASES, ids=ROUNDING_IDS)
def test_the_header_says_how_the_kernel_rounds(problem, text):
    p = PROBLEMS[problem]
    source = codegen_triton.generate(p.parse(text), p)
    header = " ".join(line[2:] for line in source.splitlines() if line.startswith("# "))
    for claim in (
        "no FMA contraction",
        "div.full.f32",
        "ex2.approx.f32",
        "sqrt.approx.ftz.f32",
        "log is a polynomial evaluated with FMAs",
    ):
        assert claim in header, claim
    assert "every operation is rounded" not in header
    _run(p, text, LONG)  # compiled as run() launches it
    found = [set(F32_OPS.findall(ptx)) for ptx in _ptx(gpu.kernel(p.parse(text), p))]
    assert found
    for ops in found:
        assert ops <= NAMED_F32_OPS, ops - NAMED_F32_OPS
        assert "fma.rn.ftz.f32" not in ops or "(log " in text, ops  # log's polynomial only
    if "(log " in text:
        assert any("fma.rn.ftz.f32" in ops for ops in found), found


@cuda
@pytest.mark.parametrize("problem", ["softmax", "attention"])
def test_the_check_says_which_shapes_it_ran(problem):
    p = PROBLEMS[problem]
    result = gpu.check(p.parse(p.known["online"]), p)
    shapes = gpu.held_out(p, codegen_triton.block_for(p))
    lengths = ", ".join(str(s[1]) for s in shapes if s[2:] in ((), (gpu.ATTENTION_DIM,)))
    assert result["ok"] and f"3 rows of N = {lengths}" in result["message"], result["message"]
    assert "never ran" not in result["message"]
    if problem == "attention":
        assert f"keys, D = {gpu.ATTENTION_DIM}" in result["message"]
        assert "keys, D = 16" in result["message"]


def _fake_do_bench(monkeypatch, answers: list[list[float]]) -> list[tuple[float, float]]:
    """do_bench replaced by one that returns ``answers`` in turn and records when it was called
    and its ``rep``; no GPU needed, and 50 ms of warm-up."""
    calls: list[tuple[float, float]] = []

    def do_bench(fn, rep, return_mode):
        assert return_mode == "all"
        calls.append((time.perf_counter(), rep))
        return answers[len(calls) - 1]

    monkeypatch.setattr(triton.testing, "do_bench", do_bench)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(gpu, "WARM_S", 0.05)
    return calls


def test_every_timing_is_warmed_and_repeated_inside_the_lock(monkeypatch):
    calls = _fake_do_bench(monkeypatch, [[3.0] * 30, [1.0] * 40, [2.0] * 50])
    runs: list[float] = []

    def fn():
        runs.append(time.perf_counter())

    with pytest.raises(RuntimeError, match="exclusive"):
        gpu._time(fn)
    assert not calls and not runs
    with gpu.exclusive():
        t = gpu._time(fn)
    want = {"ms": 2.0, "ms_min": 1.0, "ms_max": 3.0, "medians": [3.0, 1.0, 2.0]}
    assert t == {**want, "runs": [30, 40, 50]}
    # before every do_bench call, the function ran back to back for WARM_S
    starts = [0.0] + [at for at, _ in calls[:-1]]
    for start, (at, rep) in zip(starts, calls, strict=True):
        warm = [r for r in runs if start < r < at]
        assert warm and at - warm[0] >= 0.9 * gpu.WARM_S
        assert rep == 200.0  # a run this short gets do_bench's 200 ms


def test_a_do_bench_call_that_timed_too_few_runs_is_redone(monkeypatch):
    # an estimate made at idle clocks: do_bench times 4 runs instead of 40 and reads them slow
    few, enough = [9.0] * 4, [1.0] * gpu.MIN_RUNS
    calls = _fake_do_bench(monkeypatch, [few, enough, few, few, few])

    def fn():  # a run of 10 ms or more
        time.sleep(0.012)

    with gpu.exclusive():
        t = gpu._time(fn, repeats=2)
    # the first repeat is redone once; the second three times, and its last answer is kept
    assert len(calls) == 5
    assert t["runs"] == [gpu.MIN_RUNS, 4] and (t["ms_min"], t["ms_max"]) == (1.0, 9.0)
    # time for 2 MIN_RUNS runs of 10 ms: MIN_RUNS of them with a flush up to 10 ms each
    assert all(rep >= 2 * gpu.MIN_RUNS * 10 for _, rep in calls)


@cuda
def test_bench_reports_the_median_and_range_of_its_repeats():
    prog = SOFTMAX.parse(SOFTMAX.known["online"])
    b = gpu.bench(prog, SOFTMAX, [(256, 4096)], repeats=2)
    entry = b["shapes"][0]
    for t in [entry["kernel"], *entry["torch"].values()]:
        assert 0 < t["ms_min"] <= t["ms"] <= t["ms_max"], t
        assert (t["ms_min"], t["ms_max"]) == (min(t["medians"]), max(t["medians"])), t
        assert len(t["runs"]) == 2 and min(t["runs"]) >= gpu.MIN_RUNS, t
    report = gpu._report(gpu.check(prog, SOFTMAX), b)
    k = entry["kernel"]
    assert "the median of 2 medians" in report
    assert f"{k['ms']:.3f} ms ({k['ms_min']:.3f}-{k['ms_max']:.3f})" in report


@cuda
def test_fitness_times_a_correct_kernel_and_refuses_a_wrong_one():
    ok = gpu.fitness(SOFTMAX.parse(SOFTMAX.known["online"]), SOFTMAX)
    assert ok["ok"] and 0 < ok["ms_min"] <= ok["ms"] <= ok["ms_max"], ok
    assert len(ok["medians"]) == len(ok["runs"]) == gpu.REPEATS, ok
    assert min(ok["runs"]) >= gpu.MIN_RUNS, ok
    wrong = gpu.fitness(SOFTMAX.parse(SOFTMAX.known["no-rescale"]), SOFTMAX)
    assert not wrong["ok"]  # right up to 2,048 elements, wrong on the rows longer than a block


@cuda
def test_loop_uses_the_measured_time_as_fitness(tmp_path):
    from discovery import llm, loop

    answer = "```lisp\n" + SOFTMAX.known["online"] + "\n```"
    summary = loop.run(
        SOFTMAX,
        llm.Scripted([answer]),
        iterations=1,
        workers=1,
        out=tmp_path,
        memory=tmp_path / "memory.json",
        gpu=True,
    )
    assert all(t["ok"] for t in summary["gpu"].values()) and len(summary["gpu"]) == 2
    assert all(c["measured"] for c in summary["cells"])
