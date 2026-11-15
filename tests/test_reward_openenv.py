"""The OpenEnv adapter with a fake worker: reset and step, the rubric, attempts with feedback,
abstention, task discovery, and a whole episode through OpenEnv's own app over a WebSocket."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_reward import ROWS, TASK, VERIFIED, WRONG, Fake

pytest.importorskip("openenv")
from fastapi.testclient import TestClient  # noqa: E402

from ttsem.reward.core import Judge  # noqa: E402
from ttsem.reward.openenv_adapter import (  # noqa: E402
    KernelAction,
    KernelEnv,
    create_kernel_app,
)

COMPLETION = f"Here is the kernel.\n```python\n{ROWS}```\n"
SECOND = f"Another try.\n```python\n{ROWS}# second\n```\n"


def judge(tmp_path: Path, fake: Fake) -> Judge:
    return Judge(cache_dir=tmp_path, runner=fake, mode="cpu")


def test_reset_hands_out_the_task_and_step_judges_the_completion(tmp_path: Path) -> None:
    env = KernelEnv({"train": [TASK]}, judge(tmp_path, Fake(scale=VERIFIED, medium=VERIFIED)))
    first = env.reset(index=0, episode_id="e1")
    assert first.task_id == "train-0" and TASK.strip() in first.prompt
    assert first.reward is None and first.done is False
    obs = env.step(KernelAction(completion=COMPLETION))
    assert obs.reward == 1.0 and obs.done is True and obs.verdict == "verified"
    assert obs.tier == "medium" and obs.abstain is False and obs.hack is False
    assert env.state.episode_id == "e1" and env.state.step_count == 1 and env.state.best == 1.0
    assert env.rubric is not None and env.rubric.last_score == 1.0


def test_a_rejected_kernel_scores_zero_and_the_feedback_is_for_the_next_attempt(
    tmp_path: Path,
) -> None:
    fake = Fake(scale=WRONG)
    env = KernelEnv([TASK], judge(tmp_path, fake), max_attempts=2)
    env.reset(index=0)
    first = env.step(KernelAction(completion=COMPLETION))
    assert first.reward == 0.0 and first.done is False and first.verdict == "wrong"
    assert first.feedback and first.attempt == 1
    fake.by_tier = {"scale": VERIFIED, "medium": VERIFIED}
    second = env.step(KernelAction(completion=SECOND))
    assert second.reward == 1.0 and second.done is True and second.attempt == 2
    with pytest.raises(RuntimeError, match="over"):
        env.step(KernelAction(completion=SECOND))


def test_the_attempts_run_out(tmp_path: Path) -> None:
    env = KernelEnv([TASK], judge(tmp_path, Fake(scale=WRONG)), max_attempts=1)
    env.reset(index=0)
    assert env.step(KernelAction(completion=COMPLETION)).done is True


def test_an_abstention_is_the_abstain_value_and_says_so(tmp_path: Path) -> None:
    fake = Fake(scale={"verdict": "too_slow", "answer": "x.py"})
    env = KernelEnv([TASK], judge(tmp_path, fake), abstain_value=0.25)
    env.reset(index=0)
    obs = env.step(KernelAction(completion=COMPLETION))
    assert obs.score is None and obs.abstain is True and obs.reward == 0.25


def test_a_completion_without_code_is_judged_as_no_code(tmp_path: Path) -> None:
    env = KernelEnv([TASK], judge(tmp_path, Fake()))
    env.reset(index=0)
    obs = env.step(KernelAction(completion="I cannot do that."))
    assert obs.reward == 0.0 and obs.verdict == "error" and obs.kind == "no_code"


def test_step_needs_a_reset_and_a_known_task(tmp_path: Path) -> None:
    env = KernelEnv({"train": [TASK]}, judge(tmp_path, Fake()))
    with pytest.raises(RuntimeError, match="before reset"):
        env.step(KernelAction(completion=COMPLETION))
    with pytest.raises(ValueError, match="no split"):
        env.reset(split="test")
    with pytest.raises(IndexError):
        env.reset(index=3)
    assert env.reset(task=TASK).task_id == "adhoc"
    assert env.reset(seed=7).task_id == env.reset(seed=7).task_id


def test_task_discovery_needs_no_judge() -> None:
    env = KernelEnv({"train": [TASK, TASK], "test": [TASK]})
    assert env.list_splits() == ["test", "train"] and env.num_tasks("train") == 2
    assert env.get_task("train", 1) == {"id": "train-1", "index": 1}
    assert [t["index"] for t in env.get_task_range("train", 1)] == [1]
    assert len(env.list_tasks("test")) == 1
    with pytest.raises(IndexError):
        env.get_task("test", 1)
    assert env.get_metadata().name == "ttsem-kernel"


def test_an_episode_through_openenvs_app(tmp_path: Path) -> None:
    app = create_kernel_app(
        {"train": [TASK]}, judge(tmp_path, Fake(scale=VERIFIED, medium=VERIFIED))
    )
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "reset", "data": {"seed": 1}})
        reset = ws.receive_json()["data"]
        assert reset["observation"]["task_id"] == "train-0"
        ws.send_json({"type": "step", "data": {"completion": COMPLETION}})
        step = ws.receive_json()["data"]
        assert step["reward"] == 1.0 and step["done"] is True
        assert step["observation"]["verdict"] == "verified"
        ws.send_json({"type": "state"})
        assert ws.receive_json()["data"]["attempts"] == 1
