"""The bot's diff: new bad launches, flagged nondeterminism, transitions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ttsem import bot, harness

_NEEDS_TRITON = pytest.mark.skipif(
    not harness.TRITON_AVAILABLE, reason="needs a Triton install to record a launch"
)


def _run(tmp: Path, name: str, rows: list[dict]) -> Path:
    d = tmp / name
    d.mkdir()
    (d / "w0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def test_diff_reports_new_bad_and_flags_the_nondeterministic_tests(tmp_path: Path) -> None:
    base = {"launch": "1", "fn": "k", "stage": "ttgir", "n_diff": 0, "message": ""}
    prev = _run(
        tmp_path,
        "prev",
        [
            {**base, "nodeid": "t.py::test_a[0]", "verdict": "match"},
            {**base, "nodeid": "t.py::test_b[0]", "verdict": "mismatch", "n_diff": 3},
            {**base, "nodeid": "t.py::test_atomic_rmw[add]", "verdict": "match"},
        ],
    )
    cur = _run(
        tmp_path,
        "cur",
        [
            {**base, "nodeid": "t.py::test_a[0]", "verdict": "mismatch", "n_diff": 8},
            {**base, "nodeid": "t.py::test_b[0]", "verdict": "match"},
            {**base, "nodeid": "t.py::test_atomic_rmw[add]", "verdict": "mismatch", "n_diff": 1},
            {**base, "nodeid": "t.py::test_c[0]", "verdict": "poison"},
        ],
    )
    d = bot.diff(bot.load(prev), bot.load(cur))
    assert d.previous == {"match": 2, "mismatch": 1}
    assert d.current == {"match": 1, "poison": 1, "mismatch": 2}
    assert sorted(r["nodeid"] for r in d.new_bad) == [
        "t.py::test_a[0]",
        "t.py::test_atomic_rmw[add]",
        "t.py::test_c[0]",
    ]
    assert [r["nodeid"] for r in d.new_unflagged] == ["t.py::test_a[0]", "t.py::test_c[0]"]
    assert [r["nodeid"] for r in d.gone_bad] == ["t.py::test_b[0]"]
    assert d.transitions[("match", "mismatch")] == 2
    assert d.transitions[("absent", "poison")] == 1
    text = bot.render(d)
    assert "nondeterministic on the device" in text
    assert "no longer bad (1)" in text


def test_dump_module_collapses_identical_modules(tmp_path: Path) -> None:
    from ttsem import pytest_ttsem

    a = pytest_ttsem.dump_module(str(tmp_path / "ir"), "module {}", "ttgir")
    b = pytest_ttsem.dump_module(str(tmp_path / "ir"), "module {}", "ttgir")
    c = pytest_ttsem.dump_module(str(tmp_path / "ir"), "module { x }", "ttgir")
    assert a == b and a != c and a is not None
    assert sorted(p.name for p in (tmp_path / "ir").iterdir()) == sorted([a.name, c.name])
    assert a.name.endswith(".ttgir.generic")
    assert pytest_ttsem.dump_module(None, "module {}", "ttgir") is None


def test_every_atomic_test_is_flagged() -> None:
    assert bot.is_flagged(
        "t.py::test_tensor_atomic_add_access_patterns[shape162-decrease-3-1-bfloat16]"
    )
    assert bot.is_flagged("t.py::test_atomic_rmw[add]")
    assert not bot.is_flagged("t.py::test_dot[1-64]")


@_NEEDS_TRITON
def test_below_llvm_line_hands_the_llvm_stages_to_the_continuation(monkeypatch) -> None:
    from pathlib import Path as _P

    from ttsem import harness
    from ttsem import pytest_ttsem

    record = harness.capture_launches(
        _P(__file__).resolve().parent / "fixtures" / "p1.py", "cpu"
    )[0]
    monkeypatch.setattr(pytest_ttsem, "_target", lambda: harness.GPUTarget("cuda", 90, 32))
    seen: list[list[str]] = []

    class Result:
        def __init__(self, name: str) -> None:
            self.pass_name, self.verdict, self.n_diff = name, "match", 0
            self.unsupported, self.message = [], ""

    def below(rec, stages):
        seen.append([n for n, _ in stages])
        return [Result(n) for n, _ in stages]

    line = pytest_ttsem.below_llvm_line(record, below)
    assert line["below_verdict"] == "match"
    assert len(line["below"]) == len(seen[0]) >= 8
    # the JSONL row truncates the pass name at 80 characters, and some pipeline spellings
    # (`initialize-ws-cluster-barriers{compute-capability=.. ptx-version=..}`) are longer
    assert [name[:80] for name in seen[0]] == [row["pass"] for row in line["below"]]
