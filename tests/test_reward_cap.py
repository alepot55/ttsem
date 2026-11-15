"""`ttsem.reward.cap`: which launches the reward's cap recognises, and what it rewrites. The cap
turns a grid of cdiv(n, BLOCK) into one with more program instances; an answer it does not
recognise is judged as written. Parsing only: nothing here runs a kernel."""

from __future__ import annotations

from pathlib import Path

import pytest

from ttsem.reward import cap
from ttsem.reward.core import capped_source

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "reward"

HEAD = "import math\nimport triton\nimport triton.language as tl\n\n\n"


def kernel(params: str = "x_ptr, n, BLOCK: tl.constexpr", decorators: str = "") -> str:
    return f"{decorators}@triton.jit\ndef k({params}):\n    pid = tl.program_id(0)\n\n\n"


def host(body: str) -> str:
    return "class ModelNew:\n    def forward(self, x):\n        n = x.numel()\n" + body + "\n"


# (name, source, a text the capped source must hold)
QUALIFYING = [
    (
        "cdiv over a literal block",
        HEAD + kernel() + host("        k[(triton.cdiv(n, 1024),)](x, n, BLOCK=1024)"),
        ("triton.cdiv(n, min(1024, 16))", "BLOCK=min(1024, 16)"),
    ),
    (
        "cdiv over a name",
        HEAD + kernel() + host("        B = 256\n        k[(triton.cdiv(n, B),)](x, n, BLOCK=B)"),
        ("BLOCK=min(B, 16)", "triton.cdiv(n, min(B, 16))"),
    ),
    (
        "(n + B - 1) // B",
        HEAD + kernel() + host("        k[((n + 128 - 1) // 128,)](x, n, BLOCK=128)"),
        ("min(128, 16)",),
    ),
    (
        "(n - 1) // B + 1",
        HEAD + kernel() + host("        k[((n - 1) // 64 + 1,)](x, n, BLOCK=64)"),
        ("min(64, 16)",),
    ),
    (
        "-(-n // B)",
        HEAD + kernel() + host("        k[(-(-n // 64),)](x, n, BLOCK=64)"),
        ("min(64, 16)",),
    ),
    (
        "math.ceil(n / B)",
        HEAD + kernel() + host("        k[(math.ceil(n / 64),)](x, n, BLOCK=64)"),
        ("min(64, 16)",),
    ),
    (
        "a callable grid over meta",
        HEAD
        + kernel()
        + host(
            "        grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)\n"
            "        k[grid](x, n, BLOCK=256)"
        ),
        ("BLOCK=min(256, 16)",),
    ),
    (
        "a product of cdivs, a 2-D tiling on one axis",
        HEAD
        + kernel("x_ptr, n, BM: tl.constexpr, BN: tl.constexpr")
        + host("        k[(triton.cdiv(n, 32) * triton.cdiv(n, 64),)](x, n, BM=32, BN=64)"),
        ("BM=min(32, 16)", "BN=min(64, 16)"),
    ),
    (
        "the parameter's default",
        HEAD
        + kernel("x_ptr, n, BLOCK: tl.constexpr = 128")
        + host(
            "        grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)\n        k[grid](x, n)"
        ),
        ("BLOCK: tl.constexpr = min(128, 16)",),
    ),
    (
        "an autotune config below the cap",
        HEAD
        + kernel(decorators='@triton.autotune(configs=[triton.Config({"BLOCK": 8})], key=["n"])\n')
        + host(
            "        grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)\n        k[grid](x, n)"
        ),
        ("min(8, 16)",),
    ),
]

NOT_QUALIFYING = [
    ("one instance per row", HEAD + kernel() + host("        k[(x.shape[0],)](x, n, BLOCK=1024)")),
    (
        "a divisor that is no constexpr's value",
        HEAD + kernel() + host("        k[(triton.cdiv(n, 128),)](x, n, BLOCK=64)"),
    ),
    (
        "a kernel defined elsewhere",
        HEAD + host("        other[(triton.cdiv(n, 64),)](x, n, BLOCK=64)"),
    ),
    (
        "starred arguments",
        HEAD + kernel() + host("        args = (x, n)\n        k[(triton.cdiv(n, 64),)](*args)"),
    ),
    ("no kernel at all", "class ModelNew:\n    pass\n"),
    ("code that does not parse", "class ModelNew(:\n"),
]


@pytest.mark.parametrize(("name", "source", "holds"), QUALIFYING, ids=[q[0] for q in QUALIFYING])
def test_a_qualifying_launch_is_capped_on_the_same_lines(
    name: str, source: str, holds: tuple[str, ...]
) -> None:
    capped = capped_source(source, 16)
    assert capped is not None, name
    assert all(text in capped for text in holds), capped
    assert capped.count("\n") == source.count("\n")
    compile(capped, "capped.py", "exec")


@pytest.mark.parametrize(("name", "source"), NOT_QUALIFYING, ids=[q[0] for q in NOT_QUALIFYING])
def test_an_answer_the_rule_does_not_accept_is_left_as_written(name: str, source: str) -> None:
    assert capped_source(source, 16) is None, name
    launches, why = cap.analyse(source)
    assert launches == [] and why


def test_analyse_names_the_launch_and_where_its_block_comes_from() -> None:
    source = HEAD + kernel() + host("        k[(triton.cdiv(n, 1024),)](x, n, BLOCK=1024)")
    launches, why = cap.analyse(source)
    assert why == "" and len(launches) == 1
    (launch,) = launches
    assert (launch.kernel, launch.grid_kind, launch.pattern) == ("k", "tuple", "value")
    assert launch.params == ["BLOCK"] and launch.sources == ["arg"]


def test_a_rebound_min_gets_a_conditional_instead() -> None:
    source = HEAD + "min = max\n\n\n" + kernel()
    source += host("        k[(triton.cdiv(n, 1024),)](x, n, BLOCK=1024)")
    assert cap.shadows_min(source)
    capped = capped_source(source, 16)
    assert capped is not None and "(1024 if 1024 <= 16 else 16)" in capped
    assert "min(1024, 16)" not in capped


def test_a_launch_whose_autotune_blocks_all_exceed_the_cap_is_left_alone() -> None:
    tuner = '@triton.autotune(configs=[triton.Config({"BLOCK": 64})], key=["n"])\n'
    source = HEAD + kernel(decorators=tuner)
    source += host(
        "        grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)\n        k[grid](x, n)"
    )
    assert cap.analyse(source)[0], "the rule recognises the launch"
    assert capped_source(source, 16) is None, "and the reward does not cap it"


def test_the_tritonbench_test_after_the_separator_is_not_touched() -> None:
    answer = (FIXTURES / "shift_right_answer.py").read_text()
    assert cap.SEP in answer
    capped = capped_source(answer, 16)
    assert capped is not None
    code = cap.operator_part(answer)
    assert capped.endswith(answer[len(code) :])
    assert "BLOCK_SIZE=min(BLOCK_SIZE, 16)" in capped
    assert capped.count("\n") == answer.count("\n")
