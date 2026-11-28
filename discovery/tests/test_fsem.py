"""The floating-point semantics (fsem) and the adversary."""

from __future__ import annotations

import math

import numpy as np
import pytest

from discovery import adversary, fptable, fsem, problems

f32 = np.float32


def _tree(a: np.ndarray) -> np.float32:
    while len(a) > 1:
        h = len(a) // 2
        s = (a[0 : 2 * h : 2] + a[1 : 2 * h : 2]).astype(f32)
        a = np.concatenate([s, a[2 * h :]]) if len(a) % 2 else s
    return a[0]


def test_online_softmax_rounds_like_a_hand_written_float32_kernel():
    prog = problems.SOFTMAX.parse(problems.SOFTMAX.known["online"])
    x = (100 + np.random.default_rng(5).standard_normal(1000)).astype(f32)

    def block(b):
        m = b.max()
        return m, _tree(np.exp(b - m))

    def merge(a, b):
        m = max(a[0], b[0])
        da = f32(a[1] * f32(np.exp(f32(a[0] - m))))
        db = f32(b[1] * f32(np.exp(f32(b[0] - m))))
        return m, f32(da + db)

    acc = block(x[:128])
    for i in range(128, 1000, 128):
        acc = merge(acc, block(x[i : i + 128]))
    r = fsem.run(prog, {"x": x[:, None]}, "float32", block=128)
    assert r.finished["a.m"] == acc[0]
    assert np.asarray(r.finished["a.d"]).view(np.uint32) == np.asarray(acc[1]).view(np.uint32)
    y = (np.exp(x - acc[0]).astype(f32) / acc[1]).astype(f32)
    assert np.array_equal(r.outputs["y"].astype(f32), y)
    assert r.depth == 7 and r.blocks == 8


@pytest.mark.parametrize("problem", ["softmax", "attention", "variance"])
def test_the_float64_reference_agrees_with_the_exact_one(problem):
    rng = np.random.default_rng(1)
    fam = {"softmax": "normal100", "attention": "spread", "variance": "offset"}[problem]
    el = adversary.generate(problem, fam, rng, 512, 64)
    el = {k: np.asarray(fsem.FORMATS["float32"].value(v), dtype=np.float64) for k, v in el.items()}
    fast, exact = fsem.reference(problem, el, False), fsem.reference(problem, el, True)
    if problem == "softmax":
        assert abs(fast["D"] / float(exact["D"]) - 1) < 1e-13
    elif problem == "attention":
        assert np.max(np.abs(fast["o"] - exact["o"]) / exact["scale"]) < 1e-13
    else:
        assert abs(fast["var"] / exact["var"] - 1) < 1e-12
        assert abs(fast["mean"] - exact["mean"]) <= 1e-15 * abs(exact["mean"])


def test_a_float64_sum_of_squares_is_not_an_exact_reference():
    """Why the variance reference is rational: float64 cancels too, at a mean of 1e4."""
    x = (1e4 + 0.1 * np.random.default_rng(2).standard_normal(4096)).astype(f32).astype(float)
    exact = fsem.reference("variance", {"x": x[:, None]}, True)["var"]
    naive = float(np.mean(x * x) - np.mean(x) ** 2)
    assert abs(naive / exact - 1) > 1e-9  # far above float64's 1e-16


def test_bfloat16_rounds_float64_once_to_nearest_even():
    F = fsem.FORMATS["bfloat16"]
    vals = [1 + 2**-8, 1 + 3 * 2**-8, 1 + 2**-8 + 2**-30, -(1 + 2**-8), 3.0e-40, 1e30]
    got = F.value(np.array(vals))
    assert got[0] == 1.0  # a tie goes to even
    assert got[1] == 1 + 2**-6  # 1 + 3*2^-8 is a tie between 1+2^-7 and 1+2^-6
    assert got[2] == 1 + 2**-7  # just above the tie: up, not to 1 (no double rounding)
    assert got[3] == -1.0
    assert np.all(np.asarray(got, dtype=f32).view(np.uint32) & 0xFFFF == 0)
    assert abs(float(got[5]) / 1e30 - 1) <= 2**-8


@pytest.mark.parametrize(("order", "depth"), [("left", 9), ("right", 9), ("balanced", 4)])
def test_merge_tree_depth(order, depth):
    prog = problems.SOFTMAX.parse(problems.SOFTMAX.known["online"])
    x = np.random.default_rng(0).standard_normal((640, 1))
    r = fsem.run(prog, {"x": x.astype(f32)}, "float32", block=64, order=order)
    assert r.blocks == 10 and r.depth == depth


def test_the_three_one_field_programs_of_the_library_round_alike():
    """softmax-lse, -lse-abs and -lse-compact differ in the IR, not in floating point: exp(0) is
    1 and fl(a - b) = -fl(b - a), so every rounding is the same."""
    lse = [p for m, _, p in fptable.library() if m.startswith("SoftmaxLse")]
    assert len(lse) == 3
    for seed in range(3):
        x = (1e4 + np.random.default_rng(seed).standard_normal(2048)).astype(f32)[:, None]
        outs = [fsem.run(p, {"x": x}, "float32", block=256, order="random") for p in lse]
        s = [np.asarray(next(iter(o.finished.values()))).view(np.uint32) for o in outs]
        assert s[0] == s[1] == s[2]


def test_sum_of_squares_cancels_where_welford_does_not():
    P = problems.VARIANCE
    el = adversary.generate("variance", "offset", np.random.default_rng(0), 4096, 256)
    sumsq = fsem.measure("variance", P.parse(P.known["sum-of-squares"]), el, "float32")
    welford = fsem.measure("variance", P.parse(P.known["welford"]), el, "float32")
    assert sumsq["var"] > 1000 * welford["var"]
    assert welford["var"] < 1e4


@pytest.mark.parametrize("fmt", ["float16", "bfloat16"])
def test_low_precision_formats_run(fmt):
    prog = problems.SOFTMAX.parse(problems.SOFTMAX.known["online"])
    el = adversary.generate("softmax", "normal0", np.random.default_rng(0), 1024, 128)
    e = fsem.measure("softmax", prog, el, fmt, exact=False, block=128)
    assert math.isfinite(e["normalizer"]) and e["normalizer"] < 64


def test_the_adversary_only_climbs():
    prog = problems.SOFTMAX.parse(problems.SOFTMAX.known["online"])
    start = adversary.generate("softmax", "normal0", np.random.default_rng(0), 1024, 128)
    e0 = fsem.measure("softmax", prog, start, "float32", exact=False, block=128)["normalizer"]
    _, best, runs = adversary.climb(
        "softmax", prog, start, "float32", 60, np.random.default_rng(1), block=128
    )
    assert runs == 61 and best >= e0


def test_the_two_online_attentions_of_the_library_round_alike():
    """attention-online and attention-1pass-shared differ only in shared temporaries."""
    lib = {m: p for m, _, p in fptable.library()}
    el = adversary.generate("attention", "spread", np.random.default_rng(4), 1024, 128)
    el = {k: np.asarray(fsem.FORMATS["float32"].value(v)) for k, v in el.items()}
    a = fsem.run(lib["AttentionOnline"], el, "float32", block=128, order="random")
    b = fsem.run(lib["Attention1passShared"], el, "float32", block=128, order="random")
    assert np.array_equal(a.outputs["o"], b.outputs["o"])
