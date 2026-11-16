"""The OpenEnv adapter with a fake worker: reset and step, the rubric, attempts with feedback,
abstention, the judge it accepts, task discovery, and whole episodes through OpenEnv's own app
over a WebSocket."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from test_reward import ROWS, TASK, VERIFIED, WRONG, Fake, rec

pytest.importorskip("openenv")
from fastapi.testclient import TestClient  # noqa: E402

from ttsem.reward import openenv_adapter  # noqa: E402
from ttsem.reward.core import Judge  # noqa: E402
from ttsem.reward.openenv_adapter import (  # noqa: E402
    KernelAction,
    KernelEnv,
    create_kernel_app,
)

COMPLETION = f"Here is the kernel.\n```python\n{ROWS}```\n"
SECOND = f"Another try.\n```python\n{ROWS}# second\n```\n"
THIRD = f"And a third.\n```python\n{ROWS}# third\n```\n"
TOO_SLOW = {"verdict": "too_slow", "answer": "x.py"}


def judge(tmp_path: Path, fake: Fake, **kw: object) -> Judge:
    return Judge(cache_dir=tmp_path, runner=fake, mode="cpu", **kw)  # type: ignore[arg-type]


def test_reset_hands_out_the_task_and_step_judges_the_completion(tmp_path: Path) -> None:
    env = KernelEnv({"train": [TASK]}, judge(tmp_path, Fake(scale=VERIFIED, medium=VERIFIED)))
    first = env.reset(index=0, episode_id="e1")
    assert first.task_id == "train-0" and TASK.strip() in first.prompt
    assert first.reward is None and first.done is False
    obs = env.step(KernelAction(completion=COMPLETION))
    assert obs.reward == 1.0 and obs.done is True and obs.verdict == "verified"
    assert obs.tier == "medium" and obs.abstain is False and obs.hack is False
    assert env.state.episode_id == "e1" and env.state.step_count == 1 and env.state.verified
    assert env.rubric is not None and env.rubric.last_score == 1.0
    assert KernelEnv.SUPPORTS_CONCURRENT_SESSIONS is True


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
        env.step(KernelAction(completion=THIRD))


def test_a_verified_kernel_ends_the_episode_before_the_attempts_run_out(tmp_path: Path) -> None:
    env = KernelEnv([TASK], judge(tmp_path, Fake(scale=VERIFIED, medium=VERIFIED)), max_attempts=3)
    env.reset(index=0)
    assert env.step(KernelAction(completion=COMPLETION)).done is True
    with pytest.raises(RuntimeError, match="over"):
        env.step(KernelAction(completion=SECOND))


def test_the_attempts_run_out(tmp_path: Path) -> None:
    env = KernelEnv([TASK], judge(tmp_path, Fake(scale=WRONG)), max_attempts=1)
    env.reset(index=0)
    assert env.step(KernelAction(completion=COMPLETION)).done is True


def test_an_abstention_is_the_abstain_value_and_uses_an_attempt(tmp_path: Path) -> None:
    env = KernelEnv([TASK], judge(tmp_path, Fake(scale=TOO_SLOW)), abstain_value=0.25)
    env.reset(index=0)
    obs = env.step(KernelAction(completion=COMPLETION))
    assert obs.score is None and obs.abstain is True and obs.reward == 0.25 and obs.done is True


def test_the_abstain_value_is_the_environments_else_the_judges_else_zero(tmp_path: Path) -> None:
    fake = Fake(scale=TOO_SLOW)
    act = KernelAction(completion=COMPLETION)
    bare = KernelEnv([TASK], judge(tmp_path / "a", fake))
    bare.reset(index=0)
    assert bare.step(act).reward == 0.0
    carried = KernelEnv([TASK], judge(tmp_path / "b", fake, abstain_value=0.5))
    carried.reset(index=0)
    assert carried.step(act).reward == 0.5
    own = KernelEnv([TASK], judge(tmp_path / "c", fake, abstain_value=0.5), abstain_value=0.1)
    own.reset(index=0)
    assert own.step(act).reward == 0.1


def test_only_a_cpu_judge_is_accepted(tmp_path: Path) -> None:
    gate = Judge(cache_dir=tmp_path, runner=Fake(scale=VERIFIED), mode="gate")
    with pytest.raises(ValueError, match="CPU-only"):
        KernelEnv([TASK], gate)


def test_the_shared_judge_is_one_cpu_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    modes: list[str] = []

    def from_env(**kw: str) -> str:
        modes.append(kw["mode"])
        return f"judge{len(modes)}"

    monkeypatch.setattr(openenv_adapter, "_SHARED", {})
    monkeypatch.setattr(Judge, "from_env", staticmethod(from_env))
    assert openenv_adapter.shared_judge() == openenv_adapter.shared_judge() == "judge1"
    assert modes == ["cpu"]


def test_a_hack_is_seen_per_step_and_counted_in_the_state(tmp_path: Path) -> None:
    fake = Fake(scale=rec("not_judged", why="tampered"))
    env = KernelEnv([TASK], judge(tmp_path, fake), max_attempts=3)
    env.reset(index=0)
    first = env.step(KernelAction(completion=COMPLETION))
    assert first.hack is True and first.reward == 0.0 and first.done is False
    fake.by_tier = {"scale": VERIFIED, "medium": VERIFIED}
    last = env.step(KernelAction(completion=SECOND))
    assert last.hack is False and last.done is True and env.state.hacks == 1


def test_a_completion_without_code_is_judged_as_no_code(tmp_path: Path) -> None:
    env = KernelEnv([TASK], judge(tmp_path, Fake()))
    env.reset(index=0)
    obs = env.step(KernelAction(completion="I cannot do that."))
    assert obs.reward == 0.0 and obs.verdict == "error" and obs.kind == "no_code"


def test_the_prompt_closes_its_fence_whatever_the_task_ends_with() -> None:
    env = KernelEnv([TASK.rstrip()])
    prompt = env.reset(index=0).prompt
    assert prompt.endswith("\n```\n") and TASK.strip() in prompt


def test_reset_picks_the_task_and_rejects_what_it_cannot_use(tmp_path: Path) -> None:
    env = KernelEnv({"train": [TASK] * 20}, judge(tmp_path, Fake()))
    with pytest.raises(RuntimeError, match="before reset"):
        env.step(KernelAction(completion=COMPLETION))
    with pytest.raises(ValueError, match="no split"):
        env.reset(split="test")
    with pytest.raises(IndexError):
        env.reset(index=20)
    with pytest.raises(TypeError, match="idx"):
        env.reset(idx=3)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="empty"):
        env.reset(task="  ")
    assert env.reset(index=7).task_id == "train-7"
    assert env.reset(task=TASK).task_id == "adhoc"
    ids = [env.reset(seed=s).task_id for s in range(20)]
    assert len(set(ids)) > 1 and ids == [env.reset(seed=s).task_id for s in range(20)]
    with pytest.raises(TypeError, match="not a string"):
        KernelEnv(TASK)  # type: ignore[arg-type]


def test_task_discovery_needs_no_judge() -> None:
    env = KernelEnv({"train": [TASK, TASK], "test": [TASK]})
    assert env.list_splits() == ["test", "train"] and env.num_tasks("train") == 2
    assert env.get_task("train", 1) == {"id": "train-1", "index": 1}
    assert [t["index"] for t in env.get_task_range("train", 1)] == [1]
    assert len(env.list_tasks("test")) == 1
    with pytest.raises(IndexError):
        env.get_task("test", 1)
    with pytest.raises(IndexError, match="no split"):  # OpenEnv answers an IndexError with a 400
        env.num_tasks("dev")
    assert env.get_metadata().name == openenv_adapter.ENV_NAME == "ttsem-kernel"


def test_an_episode_through_openenvs_app(tmp_path: Path) -> None:
    fake = Fake(scale=VERIFIED, medium=VERIFIED)
    app = create_kernel_app({"train": [TASK, TASK]}, judge(tmp_path, fake), max_concurrent_envs=2)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "reset", "data": {"seed": 1, "index": 1}})
        reset = ws.receive_json()["data"]
        assert reset["observation"]["task_id"] == "train-1"
        ws.send_json({"type": "step", "data": {"completion": COMPLETION}})
        step = ws.receive_json()["data"]
        assert step["reward"] == 1.0 and step["done"] is True
        assert step["observation"]["verdict"] == "verified"
        ws.send_json({"type": "state"})
        state = ws.receive_json()["data"]
        assert state["attempts"] == 1 and state["verified"] is True and state["hacks"] == 0


def test_a_server_takes_one_session_unless_told_otherwise(tmp_path: Path) -> None:
    fake = Fake(scale=VERIFIED, medium=VERIFIED)
    one = create_kernel_app([TASK], judge(tmp_path / "one", fake))
    with TestClient(one) as client, client.websocket_connect("/ws"):
        with client.websocket_connect("/ws") as second:
            refused = second.receive_json()
            assert "CAPACITY" in str(refused).upper()
    two = create_kernel_app([TASK], judge(tmp_path / "two", fake), max_concurrent_envs=2)
    with TestClient(two) as client, client.websocket_connect("/ws") as a:
        with client.websocket_connect("/ws") as b:
            for ws in (a, b):
                ws.send_json({"type": "reset", "data": {}})
                assert ws.receive_json()["type"] == "observation"


def test_the_rest_routes_are_stateless_and_the_schema_shows_reset_arguments(
    tmp_path: Path,
) -> None:
    app = create_kernel_app({"train": [TASK]}, judge(tmp_path, Fake(scale=VERIFIED)))
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.post("/reset", json={"index": 0}).status_code == 200
        step = client.post("/step", json={"action": {"completion": COMPLETION}})
        assert step.status_code == 500  # a fresh environment, nobody called reset on it
        assert {"split", "index", "task"} <= set(
            client.get("/schema").json()["reset"]["properties"]
        )
        assert "attempts" in client.get("/schema").json()["state"]["properties"]
        assert client.get("/ttsem-kernel/splits").status_code == 200
        assert client.post("/ttsem-kernel/num_tasks", json={"split": "train"}).json() == {
            "num_tasks": 1
        }
        assert client.post("/ttsem-kernel/num_tasks", json={"split": "dev"}).status_code == 400


def test_importing_the_adapter_imports_neither_torch_nor_triton() -> None:
    root = Path(__file__).resolve().parent.parent
    probe = (
        "import sys\n"
        "import ttsem.reward.openenv_adapter\n"
        "assert 'torch' not in sys.modules and 'triton' not in sys.modules, sorted(sys.modules)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], cwd=root, capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr[-800:]
