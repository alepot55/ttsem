"""The reward against the real worker and judge (every answer in `bwrap`, no GPU) on the judge's
own fixtures and the cap's positive controls. The worker runs in the interpreter of
`TTSEM_REWARD_PYTHON`, by default this one: skipped where that interpreter has no torch and Triton
or the sandbox cannot be created. About half a minute on 4 vCPUs."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ttsem import judge as the_judge
from ttsem.reward import Judge

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "judge"
TRITONBENCH = ROOT / "tests" / "fixtures" / "tritonbench"
CONTROL = ROOT / "tests" / "fixtures" / "reward"
PYTHON = Path(os.environ.get("TTSEM_REWARD_PYTHON", sys.executable))


def judge_python_ready() -> bool:
    if "TTSEM_REWARD_PYTHON" in os.environ:
        return PYTHON.exists()
    return importlib.util.find_spec("torch") is not None and (
        importlib.util.find_spec("triton") is not None
    )


pytestmark = pytest.mark.skipif(
    not judge_python_ready() or the_judge.sandbox_problem(True) is not None,
    reason="needs torch and Triton in the judge's interpreter (TTSEM_REWARD_PYTHON) and bwrap",
)


def make(tmp_path: Path, **kw: Any) -> Judge:
    return Judge(cache_dir=tmp_path, python=PYTHON, ttsem_root=str(ROOT), workers=4, **kw)


def test_the_fixtures_and_the_cap_control(tmp_path: Path) -> None:
    judge = make(tmp_path, mode="gate")
    task = (FIXTURES / "task_relu.py").read_text()
    names = ("ok", "unsafe", "wrong", "no_kernel", "tampers")
    items = [(task, (FIXTURES / f"answer_{n}.py").read_text()) for n in names]
    items.append(
        ((CONTROL / "kb_shift_task.py").read_text(), (CONTROL / "kb_shift_answer.py").read_text())
    )
    try:
        got = [(v.score, v.verdict, v.kind, v.tier) for v in judge.score_many(items)]
    finally:
        judge.close()
    assert judge.info["triton"] and judge.info["torch"] and judge.info["sandbox_problem"] is None
    assert got == [
        (1.0, "verified", None, "scale"),
        (0.0, "unsafe", "oob_read", "scale"),
        (0.0, "wrong", "values", "scale"),
        (0.0, "no_kernel", None, "scale"),
        (0.0, "not_judged", "tampered", "scale"),
        # uncapped one instance covers the 1,000 elements; capped at 16, instances race
        (0.0, "unsafe", "race", "scale_cap"),
    ]


def test_a_verdict_costs_cpu_seconds_and_the_judges_line_is_the_feedback(tmp_path: Path) -> None:
    judge = make(tmp_path, mode="cpu")
    task = (FIXTURES / "task_relu.py").read_text()
    try:
        v = judge.score(task, (FIXTURES / "answer_unsafe.py").read_text())
    finally:
        judge.close()
    assert v.seconds > 0 and v.record["passes"][0]["answer_cpu_s"] > 0
    # the judge's content-addressed copy of the answer is shown as `answer.py`
    assert "`answer.py:" in v.feedback and not re.search(r"a[0-9a-f]{16}", v.feedback)
    assert v.record["passes"][0]["record"]["verdict"] == "unsafe"


METER_PROBE = """
import json, os, sys, tempfile
from pathlib import Path
os.environ["TTSEM_REWARD_TTSEM_ROOT"] = sys.argv[1]
from ttsem.reward import worker
judge, _ = worker._import_judge()  # the worker's own installation of the meter
cpu = []
inner = judge.run_sandboxed
def spy(cmd, env, cwd, timeout):
    ran = inner(cmd, env, cwd, timeout)
    cpu.append([e["cpu_s"] for e in ran.events if e.get("event") == "cpu"])
    return ran
judge.run_sandboxed = spy
task, answer = Path(sys.argv[2]), Path(sys.argv[3])
with tempfile.TemporaryDirectory() as tmp:
    worker._LOCAL.flags = ("--trials", "1", "--budget", "8192")
    record = judge.judge_one(task, answer, Path(tmp) / "r.json", 120, "auto", {}, True, True)
print(json.dumps({"verdict": record["verdict"], "cpu": cpu}))
"""


def test_the_meter_reaches_the_judges_pipe_for_the_reference_and_the_answer() -> None:
    """A change in the command the judge builds for its runner would leave the worker's meter
    unfound, and `seconds` would silently be wall time: the meter's event must arrive, once for
    the reference's run and once for the answer's."""
    done = subprocess.run(
        [str(PYTHON), "-c", METER_PROBE, str(ROOT), str(FIXTURES / "task_relu.py")]
        + [str(FIXTURES / "answer_ok.py")],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-800:]
    got = json.loads(done.stdout.strip().splitlines()[-1])
    assert got["verdict"] == "verified"
    assert len(got["cpu"]) == 2 and all(len(c) == 1 and c[0] > 0 for c in got["cpu"]), got


def test_without_bwrap_nothing_is_judged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The worker starts, finds no `bwrap` on its PATH, and the first call raises instead of
    running the answer anywhere else."""
    bare = tmp_path / "empty"
    bare.mkdir()
    monkeypatch.setenv("PATH", str(bare))
    judge = make(tmp_path / "cache", mode="gate")
    task = (FIXTURES / "task_relu.py").read_text()
    try:
        with pytest.raises(RuntimeError, match="cannot run here: bwrap is not on PATH"):
            judge.score(task, (FIXTURES / "answer_ok.py").read_text())
    finally:
        judge.close()
    assert not (tmp_path / "cache" / "kb").exists(), "no pass was started"


def test_tritonbench_fixtures_and_the_cap_control(tmp_path: Path) -> None:
    judge = make(tmp_path, mode="gate")
    task = (TRITONBENCH / "task_relu.py").read_text()
    names = ("ok", "unsafe", "wrong", "no_kernel")
    items = [(task, (TRITONBENCH / f"answer_{n}.py").read_text()) for n in names]
    items.append(
        (
            (CONTROL / "shift_right_task.py").read_text(),
            (CONTROL / "shift_right_answer.py").read_text(),
        )
    )
    try:
        got = [
            (v.score, v.verdict, v.kind, v.tier)
            for v in judge.score_many(items, format="tritonbench")
        ]
    finally:
        judge.close()
    assert got == [
        (1.0, "verified", None, "test"),
        (0.0, "unsafe", "oob_read", "test"),
        (0.0, "wrong", "values", "test"),
        (0.0, "no_kernel", None, "test"),
        (0.0, "unsafe", "race", "test_cap"),
    ]


def test_a_dtype_the_reference_does_not_take_abstains_under_v3_and_is_zero_under_kernelbench(
    tmp_path: Path,
) -> None:
    """ttsem's `compare` calls `torch.allclose` under every rule, which raises where the dtypes
    differ. The KernelBench-v3 harness casts both outputs to fp32 first, so under `v3` the reward
    abstains (`compare_dtype`) instead of scoring the judge's own error against the answer; under
    `kernelbench`, where KernelBench's own `allclose` raises too, the error stands. If the judge's
    `compare` stops raising under `v3`, this expectation moves with it."""
    task = (FIXTURES / "task_relu.py").read_text()
    answer = (FIXTURES / "answer_ok.py").read_text().replace("return out\n", "return out.half()\n")
    assert "out.half()" in answer
    judge = make(tmp_path, mode="gate")
    try:
        v3 = judge.score(task, answer, rule="v3", precision="keep")
        kb = judge.score(task, answer)
    finally:
        judge.close()
    assert (v3.score, v3.verdict, v3.kind) == (None, "error", "compare_dtype")
    assert v3.feedback.startswith("(the judge abstains)")
    assert (kb.score, kb.verdict, kb.kind) == (0.0, "error", "raised")
