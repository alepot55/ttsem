"""`ttsem.reward.core` against a fake worker: the verdict-to-reward table, the tiers, the cap, the
shape contract, the gate's monotonicity and the answer's extraction, then the worker pool and the
worker's CPU meter on stand-in processes. Nothing here runs a kernel or needs torch, Triton or
a sandbox: `test_reward_integration.py` runs the real worker."""

from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ttsem.reward import core, worker
from ttsem.reward.core import (
    Judge,
    Verdict,
    WorkerPool,
    capped_source,
    extract_answer,
    kind_of,
    scores,
)

TASK = "import torch\nclass Model(torch.nn.Module):\n    pass\n"

# a launch over cdiv(n, BLOCK) with BLOCK a constexpr: the cap applies
QUALIFIES = """import triton
import triton.language as tl


@triton.jit
def k(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)


class ModelNew:
    def forward(self, x):
        n = x.numel()
        k[(triton.cdiv(n, 1024),)](x, n, BLOCK=1024)
        return x
"""

# one program instance per row: no cdiv grid over a constexpr, the cap does not apply
ROWS = """import triton
import triton.language as tl


@triton.jit
def k(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)


class ModelNew:
    def forward(self, x):
        k[(x.shape[0],)](x, x.shape[1], BLOCK=1024)
        return x
"""

SCALED = {"factor": 32, "footprint_full": 1 << 18, "params_full": 0}
HUGE = {"factor": 512, "footprint_full": 1 << 24, "params_full": 0}


def rec(verdict: str, scaling: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    return {"verdict": verdict, "answer": "x.py", "scaling": scaling or SCALED, **extra}


VERIFIED = rec("verified", trials=[{"max_abs_diff": 0.0, "pass": True}])
WRONG = rec("wrong", trials=[{"max_abs_diff": 0.5, "ref_nan": False, "pass": False}])
UNSAFE = rec("unsafe", kind="race", message="kernel `k`: program instances race")
UNCHECKED = rec("not_judged", why="races_unchecked", message="1 launch too large")


class Fake:
    """A worker that answers each tier with a canned record and remembers what it was asked."""

    def __init__(self, **by_tier: dict[str, Any]) -> None:
        self.by_tier = by_tier
        self.asked: list[str] = []

    def __call__(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        if payload["op"] == "hello":
            return {"ttsem": "0.2.0", "sandbox_problem": None}
        tier = tier_of(payload)
        self.asked.append(tier)
        if tier not in self.by_tier:
            raise AssertionError(f"tier {tier} was not expected")
        record = self.by_tier[tier]
        if "error" in record:
            return {"error": record["error"]}
        return {"record": record, "line": f"{record['verdict']} at {tier}", "answer_s": 1.5,
                "reference_s": 2.0, "answer_cpu_s": 1.25, "reference_cpu_s": 1.75}  # fmt: skip


def tier_of(payload: dict[str, Any]) -> str:
    if payload["op"] == "tritonbench":
        return "test_cap" if ".c16" in payload["report"] else "test"
    sub = Path(payload["out"]).name
    if sub == "b13_t1":
        return "scale_cap" if ".c16" in payload["report"] else "scale"
    return {"b17_t1": "medium", "real_t1": "real"}[sub]


def judge(tmp_path: Path, fake: Fake, **kw: Any) -> Judge:
    return Judge(cache_dir=tmp_path, runner=fake, workers=2, **kw)


# ---------------------------------------------------------------- the table


@pytest.mark.parametrize(
    ("record", "kind"),
    [
        (UNSAFE, "race"),
        (WRONG, "values"),
        (rec("wrong", trials=[{"max_abs_diff": math.nan, "ref_nan": False}]), "unwritten"),
        (rec("wrong", trials=[{"max_abs_diff": math.nan, "ref_nan": True}]), "values"),
        (rec("wrong", configs={"failed": {"index": 2}}), "autotune"),
        (rec("wrong", not_copied=["w"], trials=[{"max_abs_diff": 5.0}]), "weights"),
        (rec("wrong", trials=[{"shape": [[4], [2]]}]), "shape"),
        (rec("error", stage="forward", message="CompilationError: at 12:4"), "version"),
        (rec("error", stage="load", message="ModuleNotFoundError: No module"), "version"),
        (rec("error", stage="forward", message="AssertionError: D == 1024"), "raised"),
        (rec("error", stage="configs", message="PTXASError: config 0"), "configs"),
        (
            rec(
                "error",
                stage="forward",
                message="RuntimeError: Float did not match Half",
                settings={"rule": "v3", "precision": "keep", "gpu": "somegpu"},
            ),
            "compare_dtype",
        ),  # fmt: skip
        (
            rec(
                "error",
                stage="forward",
                message="RuntimeError: Float did not match Half",
                settings={"rule": "kernelbench", "precision": "fp32", "gpu": ""},
            ),
            "raised",
        ),  # fmt: skip
        (
            rec(
                "error",
                stage="forward",
                message="RuntimeError: Float did not match Half",
                where="answer.py:40",
                settings={"rule": "v3"},
            ),
            "raised",
        ),  # fmt: skip
        (rec("error", why="crash", message="died"), "crash"),
        (rec("error", why="shared_memory", message="every config skipped"), "shared_memory"),
        (UNCHECKED, "races_unchecked"),
        (rec("not_judged", why="tampered"), "tampered"),
        (
            rec("not_judged", why="tampered", message="it imports `cpp_extension` (line 3)"),
            "native_code",
        ),  # fmt: skip
        (rec("too_slow"), "timeout"),
        (rec("no_kernel"), None),
        (VERIFIED, None),
    ],
)
def test_kind_of(record: dict[str, Any], kind: str | None) -> None:
    assert kind_of(record) == kind


def test_kind_of_tritonbench() -> None:
    assert kind_of({"verdict": "unsafe", "class": "oob_read"}, "tritonbench") == "oob_read"
    assert kind_of({"verdict": "wrong", "class": "wrong_result"}, "tritonbench") == "values"
    reduced = {"verdict": "verified", "class": "close_reduced_precision", "tolerance": "reduced"}
    assert kind_of(reduced, "tritonbench") == "reduced_precision"
    error = {"verdict": "error", "class": "call_error", "message": "CompilationError: x"}
    assert kind_of(error, "tritonbench") == "version"
    assert (
        kind_of({"verdict": "not_judged", "class": "environment"}, "tritonbench") == "environment"
    )


@pytest.mark.parametrize(
    ("verdict", "kind", "cpu", "gate"),
    [
        ("verified", None, 1.0, 1.0),
        ("wrong", "unwritten", 0.0, 0.0),
        ("unsafe", "oob_read", 0.0, 0.0),
        ("no_kernel", None, 0.0, 0.0),
        ("not_judged", "tampered", 0.0, 0.0),
        ("not_judged", "forged", 0.0, 0.0),
        ("not_judged", "native_code", 0.0, None),
        ("error", "raised", 0.0, 0.0),
        ("error", "version", 0.0, None),
        ("error", "shared_memory", 0.0, None),
        ("error", "crash", None, None),
        ("error", "configs", None, None),
        ("error", "compare_dtype", None, None),
        ("too_slow", "timeout", None, None),
        ("not_judged", "races_unchecked", None, None),
        ("not_judged", "tool_error", None, None),
    ],
)
def test_scores(verdict: str, kind: str | None, cpu: float | None, gate: float | None) -> None:
    assert scores(verdict, kind) == (cpu, gate)


@pytest.mark.parametrize("base", [-1.0, 0.0, 0.3, 1.0, 7.5])
@pytest.mark.parametrize("score", [1.0, 0.0, None])
@pytest.mark.parametrize("on_abstain", ["keep", "zero"])
def test_gate_is_monotone(
    tmp_path: Path, base: float, score: float | None, on_abstain: str
) -> None:
    j = judge(tmp_path, Fake(), on_abstain=on_abstain)
    v = Verdict(score, "verified", None, "", "scale", 0.0, {})
    gated = j.gate(base, v)
    assert gated <= base
    if score == 1.0 or (score is None and on_abstain == "keep"):
        assert gated == base
    else:
        assert gated == min(base, 0.0)


# ---------------------------------------------------------------- the tiers


def test_verified_at_both_tiers(tmp_path: Path) -> None:
    fake = Fake(scale=VERIFIED, medium=VERIFIED)
    v = judge(tmp_path, fake).score(TASK, ROWS)
    assert (v.score, v.verdict, v.tier) == (1.0, "verified", "medium")
    assert fake.asked == ["scale", "medium"]
    assert v.seconds == 2.5 and v.record["wall_s"] == 3.0
    assert v.record["reference_s"] == 4.0 and v.record["reference_cpu_s"] == 3.5


def test_a_rejection_at_scale_stops_there(tmp_path: Path) -> None:
    fake = Fake(scale=WRONG)
    v = judge(tmp_path, fake).score(TASK, QUALIFIES)
    assert (v.score, v.verdict, v.kind, v.tier) == (0.0, "wrong", "values", "scale")
    assert fake.asked == ["scale"]
    assert "divided by 32" in v.feedback and "any size of the task" in v.feedback


def test_medium_catches_what_grows_with_the_shape(tmp_path: Path) -> None:
    fake = Fake(scale=VERIFIED, medium=WRONG)
    v = judge(tmp_path, fake, cap=None).score(TASK, QUALIFIES)
    assert (v.score, v.tier) == (0.0, "medium")


def test_medium_is_skipped_when_scale_ran_the_real_sizes(tmp_path: Path) -> None:
    small = rec("verified", {"factor": 1, "footprint_full": 4000, "params_full": 0})
    fake = Fake(scale=small)
    v = judge(tmp_path, fake).score(TASK, ROWS)
    assert (v.score, v.tier) == (1.0, "scale") and fake.asked == ["scale"]


def test_races_unchecked_at_medium_only_is_accepted(tmp_path: Path) -> None:
    v = judge(tmp_path, Fake(scale=VERIFIED, medium=UNCHECKED)).score(TASK, ROWS)
    assert (v.score, v.verdict, v.tier) == (1.0, "verified", "scale")


def test_races_unchecked_at_every_tier_abstains(tmp_path: Path) -> None:
    v = judge(tmp_path, Fake(scale=UNCHECKED, medium=UNCHECKED)).score(TASK, ROWS)
    assert v.score is None and v.kind == "races_unchecked"
    assert v.feedback.startswith("(the judge abstains)")


def test_a_timeout_at_medium_abstains(tmp_path: Path) -> None:
    v = judge(tmp_path, Fake(scale=VERIFIED, medium=rec("too_slow"))).score(TASK, ROWS)
    assert v.score is None and (v.verdict, v.tier) == ("too_slow", "medium")


def test_the_cap_runs_on_a_qualifying_answer_and_counts_only_unsafe(tmp_path: Path) -> None:
    fake = Fake(scale=VERIFIED, scale_cap=UNSAFE)
    v = judge(tmp_path, fake).score(TASK, QUALIFIES)
    assert (v.score, v.verdict, v.kind, v.tier) == (0.0, "unsafe", "race", "scale_cap")
    assert "capped at 16" in v.feedback
    fake = Fake(scale=VERIFIED, scale_cap=WRONG, medium=VERIFIED)
    v = judge(tmp_path, fake).score(TASK, QUALIFIES)
    assert v.score == 1.0 and fake.asked == ["scale", "scale_cap", "medium"]


def test_the_cap_skips_an_answer_that_does_not_qualify(tmp_path: Path) -> None:
    fake = Fake(scale=VERIFIED, medium=VERIFIED)
    judge(tmp_path, fake).score(TASK, ROWS)
    assert fake.asked == ["scale", "medium"]


# autotune configs that all list blocks above the cap: the host may size a buffer for the smallest
AUTOTUNED_FLOOR = """import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({"BLOCK": 1024}), triton.Config({"BLOCK": 2048})], key=["n"]
)
@triton.jit
def k(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)


class ModelNew:
    def forward(self, x):
        n = x.numel()
        k[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](x, n)
        return x
"""


def test_the_cap_stays_within_the_answers_own_autotune_blocks() -> None:
    assert capped_source(AUTOTUNED_FLOOR, 16) is None
    low = AUTOTUNED_FLOOR.replace('{"BLOCK": 1024}', '{"BLOCK": 8}')
    capped = capped_source(low, 16)
    assert capped is not None and "min(8, 16)" in capped and "min(2048, 16)" in capped


def test_no_kernel_at_the_shrunken_shape_is_decided_at_medium(tmp_path: Path) -> None:
    # a PyTorch fallback below a size threshold that only the scaled shape crosses
    fake = Fake(scale=rec("no_kernel"), medium=VERIFIED)
    v = judge(tmp_path, fake).score(TASK, QUALIFIES)
    assert (v.score, v.verdict, v.tier) == (1.0, "verified", "medium")
    assert fake.asked == ["scale", "medium"]
    fake = Fake(scale=rec("no_kernel"), medium=rec("no_kernel"))
    v = judge(tmp_path, fake).score(TASK, QUALIFIES)
    assert (v.score, v.verdict, v.tier) == (0.0, "no_kernel", "medium")


def test_capped_source_keeps_the_lines() -> None:
    capped = capped_source(QUALIFIES, 16)
    assert capped is not None and capped.count("\n") == QUALIFIES.count("\n")
    assert "min(1024, 16)" in capped
    assert capped_source(ROWS, 16) is None


def test_the_task_contract_rejudges_at_the_real_shape(tmp_path: Path) -> None:
    fake = Fake(scale=WRONG, real=VERIFIED)
    v = judge(tmp_path, fake, contract="task").score(TASK, ROWS)
    assert (v.score, v.tier) == (1.0, "real") and fake.asked == ["scale", "real"]
    fake = Fake(scale=rec("wrong", HUGE, trials=[{"max_abs_diff": 1.0}]))
    v = judge(tmp_path, fake, contract="task").score(TASK, ROWS)
    assert v.score is None and v.kind == "not_at_task_shape"
    fake = Fake(scale=rec("not_judged", why="tampered"))
    v = judge(tmp_path, fake, contract="task").score(TASK, ROWS)
    assert (v.score, v.kind) == (0.0, "tampered") and v.hack


def test_compile_errors_abstain_in_gate_mode_only(tmp_path: Path) -> None:
    error = rec("error", stage="forward", message="CompilationError: at 3:4: unsupported")
    gate = judge(tmp_path, Fake(scale=error), mode="gate").score(TASK, ROWS)
    cpu = judge(tmp_path, Fake(scale=error), mode="cpu").score(TASK, ROWS)
    assert gate.score is None and cpu.score == 0.0


def test_a_dtype_clash_in_the_judges_v3_compare_abstains(tmp_path: Path) -> None:
    error = rec("error", stage="forward", message="RuntimeError: Float did not match Half",
                settings={"rule": "v3", "precision": "keep", "gpu": "RTX3090"})  # fmt: skip
    for mode in ("gate", "cpu"):
        v = judge(tmp_path, Fake(scale=error), mode=mode).score(TASK, ROWS, rule="v3")
        assert (v.score, v.kind) == (None, "compare_dtype")
        assert "Float did not match Half" in v.feedback and "any size" not in v.feedback


def test_a_worker_failure_abstains(tmp_path: Path) -> None:
    v = judge(tmp_path, Fake(scale={"error": "boom"})).score(TASK, ROWS)
    assert v.score is None and (v.verdict, v.kind) == ("not_judged", "tool_error")


def test_no_code_is_zero_without_a_run(tmp_path: Path) -> None:
    fake = Fake()
    v = judge(tmp_path, fake).score(TASK, None)
    assert (v.score, v.kind) == (0.0, "no_code") and fake.asked == []


def test_a_missing_sandbox_refuses_to_judge(tmp_path: Path) -> None:
    def runner(payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        return {"sandbox_problem": "bwrap is not on PATH"}

    with pytest.raises(RuntimeError, match="bwrap"):
        Judge(cache_dir=tmp_path, runner=runner).score(TASK, ROWS)


def test_unknown_settings_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown settings"):
        judge(tmp_path, Fake()).score(TASK, ROWS, atol=1e-4)


def test_tritonbench_one_tier_and_the_cap(tmp_path: Path) -> None:
    tb_ok = {"verdict": "verified", "class": "verified", "answer": "x.py"}
    tb_race = {"verdict": "unsafe", "class": "race", "kind": "race", "answer": "x.py"}
    fake = Fake(test=tb_ok, test_cap=tb_race)
    v = judge(tmp_path, fake).score(TASK, QUALIFIES, format="tritonbench")
    assert (v.score, v.kind, v.tier) == (0.0, "race", "test_cap")
    fake = Fake(test={"verdict": "not_judged", "class": "ran_random_task", "answer": "x.py"})
    v = judge(tmp_path, fake).score(TASK, ROWS, format="tritonbench")
    assert v.score is None and v.kind == "ran_random_task"


def test_files_are_content_addressed(tmp_path: Path) -> None:
    judge(tmp_path, Fake(scale=VERIFIED, medium=VERIFIED)).score(TASK, ROWS)
    names = sorted(p.name for p in (tmp_path / "src").iterdir())
    assert len(names) == 2 and all(n.endswith(".py") for n in names)


# ---------------------------------------------------------------- extraction


def test_extract_answer_takes_the_last_block_with_the_answer_class() -> None:
    text = (
        "Plan:\n```python\nclass ModelNew: pass  # draft\n```\nThen\n"
        "```python\nclass ModelNew:\n    final = 1\n```\n```bash\nls\n```\n"
        "```python\nprint('test')\n```\n"
    )
    assert extract_answer(text) == "class ModelNew:\n    final = 1\n"


def test_extract_answer_fallbacks() -> None:
    assert extract_answer("```py\nx = 1\n```") == "x = 1\n"
    raw = "import torch\nclass ModelNew(torch.nn.Module):\n    pass\n"
    assert extract_answer(raw) == raw
    assert extract_answer("class ModelNew( broken") is None
    assert extract_answer("no code here") is None


def test_feedback_names_the_answer_and_the_sizes(tmp_path: Path) -> None:
    unsafe = rec(
        "unsafe",
        {"factor": 32, "footprint_full": 1 << 18},
        kind="oob_read",
        answer_overrides={"M": 32, "#2048": 64},
        message="`a1234567890abcdef_answer.py:41: a = tl.load(a_ptrs)` and "
        "`t1234567890abcdef.a1234567890abcdef.c16.tritonbench.py:9`",
    )
    v = judge(tmp_path, Fake(scale=unsafe)).score(TASK, ROWS)
    assert "`answer.py:41: a = tl.load(a_ptrs)`" in v.feedback and "`answer.py:9`" in v.feedback
    assert "divided by 32: M=32, the literal 2048 as 64" in v.feedback


# ---------------------------------------------------------------- batches and the environment


def test_score_many_and_ascore_judge_each_item(tmp_path: Path) -> None:
    j = judge(tmp_path, Fake(scale=VERIFIED, medium=VERIFIED), cap=None)
    got = j.score_many([(TASK, ROWS), (TASK, QUALIFIES), (TASK, None)])
    assert [(v.score, v.kind) for v in got] == [(1.0, None), (1.0, None), (0.0, "no_code")]
    v = asyncio.run(j.ascore(TASK, ROWS))
    assert (v.score, v.tier) == (1.0, "medium")


def test_from_env_reads_the_variables_and_the_overrides_win(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = {
        "TTSEM_REWARD_CACHE": str(tmp_path / "cache"),
        "TTSEM_REWARD_MODE": "cpu",
        "TTSEM_REWARD_CONTRACT": "task",
        "TTSEM_REWARD_RULE": "v3",
        "TTSEM_REWARD_PRECISION": "keep",
        "TTSEM_REWARD_GPU": "somegpu",
        "TTSEM_REWARD_WORKERS": "3",
        "TTSEM_REWARD_CAP": "0",
        "TTSEM_REWARD_TIERS": "scale,real",
        "TTSEM_REWARD_TIMEOUT": "45",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    j = Judge.from_env(runner=Fake(), workers=5)
    assert (j.mode, j.contract, j.workers, j.cap) == ("cpu", "task", 5, None)
    assert j.tiers == ("scale", "real") and j.timeout_s == 45
    assert j.settings == {"rule": "v3", "precision": "keep", "gpu": "somegpu"}
    assert j.cache_dir == (tmp_path / "cache").resolve()


@pytest.mark.parametrize(
    "bad",
    [{"mode": "x"}, {"contract": "x"}, {"tiers": ("medium",)}, {"on_abstain": "x"}],
)
def test_bad_options_are_refused(tmp_path: Path, bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="one of|scale|keep or zero"):
        Judge(cache_dir=tmp_path, runner=Fake(), **bad)


# ---------------------------------------------------------------- the worker pool

# a stand-in worker: one JSON line back for each one in, and an exit on request
STAND_IN = """\
import json, os, sys
for line in sys.stdin:
    request = json.loads(line)
    if request["op"] == "die":
        sys.exit(3)
    print(json.dumps({"id": request["id"], "op": request["op"], "pid": os.getpid()}), flush=True)
"""


@pytest.fixture
def pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    script = tmp_path / "stand_in_worker.py"
    script.write_text(STAND_IN)
    monkeypatch.setattr(core, "WORKER", script)
    made = WorkerPool(Path(sys.executable), jobs=2)
    yield made
    made.close()


def test_the_pool_starts_one_worker_and_talks_json_lines(pool: WorkerPool) -> None:
    first = pool.request({"op": "hello"}, 60.0)
    second = pool.request({"op": "again"}, 60.0)
    assert (first["op"], second["op"]) == ("hello", "again") and first["pid"] == second["pid"]
    assert first["id"] != second["id"]


def test_a_worker_that_exits_fails_the_request_and_is_started_again(pool: WorkerPool) -> None:
    first = pool.request({"op": "hello"}, 60.0)
    proc = pool._proc
    with pytest.raises(RuntimeError, match="the reward worker exited"):
        pool.request({"op": "die"}, 60.0)
    assert proc is not None
    proc.wait(timeout=30)
    again = pool.request({"op": "hello"}, 60.0)
    assert again["pid"] != first["pid"]


def test_a_forked_child_starts_a_worker_of_its_own(pool: WorkerPool) -> None:
    first = pool.request({"op": "hello"}, 60.0)
    parents_worker = pool._proc
    pool._pid = -1  # as the child of a fork sees it: the pool was started by another process
    second = pool.request({"op": "hello"}, 60.0)
    assert second["pid"] != first["pid"]
    assert parents_worker is not None
    if parents_worker.stdin is not None:
        parents_worker.stdin.close()
    parents_worker.wait(timeout=30)


# ---------------------------------------------------------------- the worker's CPU meter


def test_metered_starts_the_runner_through_the_meter_whichever_form_python_gives_it() -> None:
    py = sys.executable
    around = ["bwrap", "--ro-bind", py, py, "--"]
    for runner in (
        [py, "-P", "-m", "ttsem._judge_runner"],  # Python 3.11+
        [py, "-c", "import runpy; runpy.run_module('ttsem._judge_runner')"],  # Python 3.10
    ):
        cmd = [*around, *runner, "task.py", "work"]
        wrapped = worker.metered(cmd, [runner, [py, "-m", "another"]])
        assert wrapped == [*around, py, "-I", "-c", worker.METER, *runner, "task.py", "work"]
        assert worker.metered([*around, "true"], [runner]) is None


def test_metered_finds_the_judges_own_command_for_this_python() -> None:
    from ttsem._compat import python_module

    runner = python_module(sys.executable, "ttsem._judge_runner")
    wrapped = worker.metered(["bwrap", *runner, "task.py"], [runner])
    assert wrapped is not None and wrapped[-len(runner) - 1 :] == [*runner, "task.py"]


def _metered(code: str) -> tuple[int, list[dict[str, Any]]]:
    """`code` run as a child of the meter, as the sandbox runs the runner: its exit status and
    the events the meter wrote on the pipe."""
    read_end, write_end = os.pipe()
    try:
        cmd = [sys.executable, "-I", "-c", worker.METER, sys.executable, "-c", code]
        done = subprocess.run(
            [*cmd, "--events", str(write_end)], pass_fds=(write_end,), check=False, timeout=120
        )
    finally:
        os.close(write_end)
    data = os.read(read_end, 1 << 16)
    os.close(read_end)
    return done.returncode, [json.loads(line) for line in data.decode().splitlines()]


def test_the_meter_sends_the_cpu_seconds_of_the_runner_and_keeps_its_exit_status() -> None:
    spin = "import time\nt = time.process_time()\nwhile time.process_time() - t < 0.3: pass\n"
    code, events = _metered(spin)
    assert code == 0 and [e["event"] for e in events] == ["cpu"]
    assert 0.25 <= events[0]["cpu_s"] < 30
    code, events = _metered("import sys; sys.exit(3)")
    assert code == 3 and len(events) == 1


def test_the_meter_passes_a_signal_death_on() -> None:
    code, events = _metered("import os, signal; os.kill(os.getpid(), signal.SIGKILL)")
    assert code == -9 and len(events) == 1


# ---------------------------------------------------------------- what the package needs


def test_importing_the_reward_imports_neither_torch_nor_triton() -> None:
    root = Path(__file__).resolve().parent.parent
    probe = (
        "import sys\n"
        "import ttsem.reward, ttsem.reward.verl_adapter, ttsem.reward.worker\n"
        "assert 'torch' not in sys.modules and 'triton' not in sys.modules, sorted(sys.modules)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], cwd=root, capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr[-800:]


# ---------------------------------------------------------------- the worker's own interpreter


class _Judge:
    """What `Worker.hello` asks of the judge module: a root, a sandbox verdict and a prefix."""

    ROOT = Path("/nowhere")

    def __init__(self, prefix: list[str]) -> None:
        self.prefix = prefix

    def sandbox_problem(self, strict: bool) -> str | None:
        return None

    def sandbox_prefix(self, *args: Any, **kwargs: Any) -> list[str]:
        return list(self.prefix)


def test_hello_reports_a_judge_python_that_cannot_start_in_the_sandbox() -> None:
    """The sandbox itself works but the interpreter does not start in it: `hello` says so, which
    makes `Judge._check` raise instead of every pass abstaining."""
    problem = worker.Worker(_Judge(["false"]), None).hello({})["sandbox_problem"]
    assert problem is not None and "cannot start in the sandbox" in problem


def test_hello_is_clean_when_the_judge_python_starts() -> None:
    assert worker.Worker(_Judge([]), None).hello({})["sandbox_problem"] is None
