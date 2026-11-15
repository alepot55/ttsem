"""An OpenEnv environment over the exact judge: `reset` hands out a task, `step` judges one answer.

Needs `openenv` (BSD-3, `pip install openenv`; written against 0.13.0), which `ttsem` does not
depend on: nothing else in the package imports this module. The reward is the judge's CPU-only
score (`mode="cpu"`: 1.0 for a verified kernel, 0.0 for a rejected one, no GPU in the loop), taken
through a `Rubric` so that a trainer can read it back (`env.rubric.last_score`). An abstention (a
failure that may be the judge's environment, not the answer's) is `abstain_value` as a reward,
0.0 as in the verl adapter, and says so in `abstain`: drop those samples if the trainer can.

    from ttsem.reward.openenv_adapter import KernelEnv, create_kernel_app

    env = KernelEnv({"train": [reference_src, ...]})   # a KernelBench-style reference per task
    obs = env.reset(split="train", index=0)            # obs.prompt: what the model is shown
    obs = env.step(KernelAction(completion=text))      # obs.reward, obs.verdict, obs.feedback

    app = create_kernel_app({"train": [...]})          # the OpenEnv server, `uvicorn` serves it

With `max_attempts` above 1 an episode lasts until the kernel is verified or the attempts are
spent, and `obs.feedback` (the judge's paragraph, at the line of the defect) is what the next
attempt is shown. The judge is configured by the same environment variables as the verl adapter
(`Judge.from_env`: TTSEM_REWARD_PYTHON, TTSEM_REWARD_CACHE, TTSEM_REWARD_WORKERS, ...) unless one
is passed in. Every answer runs in the judge's `bwrap` sandbox, never in this process.

An episode lives in one WebSocket session (`/ws`, what OpenEnv's `EnvClient` uses): OpenEnv's REST
routes make a fresh environment per request, so a `/step` there answers "step before reset".

Status: tested against a fake worker and through OpenEnv's app over a WebSocket (0.13.0); not
tested in a training run.
"""

from __future__ import annotations

import asyncio
import random
import threading
from collections.abc import Mapping, Sequence
from typing import Any

from fastapi import FastAPI
from openenv.core.env_server.http_server import create_fastapi_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, EnvironmentMetadata, Observation, State
from openenv.core.rubrics import Rubric

from ttsem.reward.core import Judge, Verdict, extract_answer

PROMPT = (
    "Rewrite the PyTorch module below as `ModelNew`, with the same `forward` signature as `Model`, "
    "so that its operators run in Triton kernels that you write. Reply with one ```python block "
    "that defines `ModelNew`.\n\n```python\n{task}```\n"
)


class KernelAction(Action):
    """A model's completion: the answer's code is the last fenced Python block in it."""

    completion: str


class KernelObservation(Observation):
    """What the model sees: the task after a reset, the verdict and the judge's line after a step.
    `score` is the judge's (None: an abstention); `reward` is what the rubric makes of it."""

    task_id: str = ""
    prompt: str = ""
    attempt: int = 0
    score: float | None = None
    verdict: str = ""
    kind: str = ""
    tier: str = ""
    feedback: str = ""
    abstain: bool = False
    hack: bool = False
    seconds: float = 0.0


class KernelState(State):
    task_id: str = ""
    attempts: int = 0
    best: float = 0.0


class VerdictRubric(Rubric):
    """The judge's score as the reward; an abstention is `abstain_value`."""

    def __init__(self, abstain_value: float = 0.0) -> None:
        super().__init__()
        self.abstain_value = abstain_value

    def forward(self, action: Any, observation: Any) -> float:
        score = observation.score
        return self.abstain_value if score is None else float(score)


_SHARED: dict[str, Judge] = {}
_LOCK = threading.Lock()


def shared_judge() -> Judge:
    """The process's CPU-only judge, configured from the environment once: every session of a
    server shares one worker pool."""
    with _LOCK:
        if "cpu" not in _SHARED:
            _SHARED["cpu"] = Judge.from_env(mode="cpu")
        return _SHARED["cpu"]


class KernelEnv(Environment[KernelAction, KernelObservation, KernelState]):
    """Tasks are KernelBench-style references (`Model`, `get_inputs`, `get_init_inputs`), by split
    name or as one list (the split `train`). `reset(split=, index=, task=)` picks one: an index in
    a split, a reference given as text, else a random one of the split (`seed` repeats it)."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        tasks: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
        judge: Judge | None = None,
        *,
        max_attempts: int = 1,
        abstain_value: float = 0.0,
    ) -> None:
        super().__init__(rubric=VerdictRubric(abstain_value))
        if max_attempts < 1:
            raise ValueError("max_attempts is at least 1")
        self._splits: dict[str, list[str]] = {}
        if isinstance(tasks, Mapping):
            self._splits = {name: list(pool) for name, pool in tasks.items()}
        elif tasks is not None:
            self._splits = {"train": list(tasks)}
        self._judge = judge
        self.max_attempts = max_attempts
        self._task = ""
        self._state = KernelState()

    @property
    def judge(self) -> Judge:
        """Built on first use: an environment made only to list its tasks needs no judge."""
        if self._judge is None:
            self._judge = shared_judge()
        return self._judge

    @property
    def state(self) -> KernelState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        from ttsem import __version__

        return EnvironmentMetadata(
            name="ttsem-kernel",
            description="Write a Triton kernel for a PyTorch module; a CPU-only judge checks it "
            "for correctness, out-of-bounds accesses and races.",
            version=__version__,
            author="Alessandro Potenza",
        )

    # ------------------------------------------------------------ the episode

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        split: str = "train",
        index: int | None = None,
        task: str | None = None,
        **kwargs: Any,
    ) -> KernelObservation:
        if task is not None:
            task_id, source = "adhoc", task
        else:
            if split not in self._splits:
                raise ValueError(f"no split {split!r}: {sorted(self._splits)}")
            pool = self._splits[split]
            if not pool:
                raise ValueError(f"split {split!r} has no task")
            i = (
                self._index(split, index)
                if index is not None
                else random.Random(seed).randrange(len(pool))
            )
            task_id, source = f"{split}-{i}", pool[i]
        self._task = source
        self._state = KernelState(episode_id=episode_id, task_id=task_id)
        return KernelObservation(
            task_id=task_id, prompt=PROMPT.format(task=source), done=False, reward=None
        )

    def step(
        self, action: KernelAction, timeout_s: float | None = None, **kwargs: Any
    ) -> KernelObservation:
        """`timeout_s` is not used: the judge's own timeouts (`Judge(timeout_s=)`) apply."""
        if not self._task:
            raise RuntimeError("step before reset")
        if self._state.attempts >= self.max_attempts or self._state.best >= 1.0:
            raise RuntimeError("the episode is over: reset")
        v: Verdict = self.judge.score(self._task, extract_answer(action.completion))
        self._state.attempts += 1
        self._state.step_count += 1
        obs = KernelObservation(
            task_id=self._state.task_id,
            attempt=self._state.attempts,
            score=v.score,
            verdict=v.verdict,
            kind=v.kind or "",
            tier=v.tier,
            feedback=v.feedback,
            abstain=v.abstain,
            hack=v.hack,
            seconds=v.seconds,
        )
        assert self.rubric is not None
        obs.reward = self.rubric(action, obs)
        self._state.best = max(self._state.best, v.score or 0.0)
        obs.done = self._state.best >= 1.0 or self._state.attempts >= self.max_attempts
        return obs

    async def step_async(
        self, action: KernelAction, timeout_s: float | None = None, **kwargs: Any
    ) -> KernelObservation:
        return await asyncio.to_thread(self.step, action, timeout_s, **kwargs)

    # ------------------------------------------------------------ task discovery (TaskProvider)

    def _index(self, split: str, index: int) -> int:
        if not 0 <= index < len(self._splits[split]):
            raise IndexError(f"split {split!r} has {len(self._splits[split])} tasks, not {index}")
        return index

    def list_splits(self) -> list[str]:
        return sorted(self._splits)

    def num_tasks(self, split: str) -> int:
        return len(self._splits[split])

    def get_task(self, split: str, index: int) -> dict[str, Any]:
        return {"id": f"{split}-{self._index(split, index)}", "index": index}

    def list_tasks(self, split: str) -> list[dict[str, Any]]:
        return [self.get_task(split, i) for i in range(self.num_tasks(split))]

    def get_task_range(
        self, split: str, start: int | None = None, stop: int | None = None
    ) -> list[dict[str, Any]]:
        n = self.num_tasks(split)
        return [self.get_task(split, i) for i in range(*slice(start, stop).indices(n))]


def create_kernel_app(
    tasks: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
    judge: Judge | None = None,
    *,
    max_attempts: int = 1,
    abstain_value: float = 0.0,
    max_concurrent_envs: int | None = None,
) -> FastAPI:
    """The OpenEnv server for `KernelEnv`: `uvicorn.run(create_kernel_app(tasks))`."""
    return create_fastapi_app(
        lambda: KernelEnv(tasks, judge, max_attempts=max_attempts, abstain_value=abstain_value),
        KernelAction,
        KernelObservation,
        max_concurrent_envs=max_concurrent_envs,
        env_name="ttsem_kernel",
        state_cls=KernelState,
    )
