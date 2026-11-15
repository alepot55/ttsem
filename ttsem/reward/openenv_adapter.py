"""An OpenEnv environment over the exact judge: `reset` hands out a task, `step` judges one answer.

Needs `openenv` (BSD-3, `pip install "ttsem[openenv]"`; written against 0.13.0), which `ttsem` does
not depend on: nothing else in the package imports this module. The reward is the judge's CPU-only
score (a `Judge` in `mode="cpu"`: 1.0 for a verified kernel, 0.0 for a rejected one, no GPU in the
loop; a judge in another mode is refused), taken through a `Rubric` so that a trainer can read it
back (`env.rubric.last_score`). An abstention (a failure that may be the judge's environment, not
the answer's) is the reward `abstain_value`: the environment's, else the judge's own, else 0.0 as
in the verl adapter. It says so in `abstain`: drop those samples if the trainer can. An abstention
uses up an attempt.

    from ttsem.reward.openenv_adapter import KernelAction, KernelEnv, create_kernel_app

    env = KernelEnv({"train": [reference_src, ...]})   # a KernelBench-style reference per task
    obs = env.reset(split="train", index=0)            # obs.prompt: what the model is shown
    obs = env.step(KernelAction(completion=text))      # obs.reward, obs.verdict, obs.feedback

    app = create_kernel_app({"train": [...]}, max_concurrent_envs=8)  # `uvicorn` serves it

With `max_attempts` above 1 an episode lasts until the kernel is verified or the attempts are
spent, and `obs.feedback` (the judge's paragraph, at the line of the defect) is what the next
attempt is shown; the prompt is in the observation of `reset` only, the trainer keeps it. The judge
is the one passed in, else one built once per process from the environment variables of the verl
adapter (`Judge.from_env`: TTSEM_REWARD_PYTHON, TTSEM_REWARD_CACHE, TTSEM_REWARD_WORKERS, ...).
Every answer runs in the judge's `bwrap` sandbox, never in this process, and so does the reference
that `reset(task=...)` takes from the caller.

An episode lives in one WebSocket session (`/ws`, what OpenEnv's `EnvClient` uses): OpenEnv's REST
routes make a fresh environment per request, so a REST `/step` fails (HTTP 500; the server log says
"step before reset"). A server takes one session at a time unless `max_concurrent_envs` is given:
that is OpenEnv's default, and parallel rollouts need more.

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
from openenv.core.env_server.types import (
    Action,
    EnvironmentMetadata,
    Observation,
    ResetRequest,
    State,
)
from openenv.core.rubrics import Rubric

from ttsem.reward.core import Judge, Verdict, extract_answer

ENV_NAME = "ttsem-kernel"
PROMPT = (
    "Rewrite the PyTorch module below as `ModelNew`, with the same `forward` signature as `Model`, "
    "so that its operators run in Triton kernels that you write. Reply with one ```python block "
    "that defines `ModelNew`.\n\n```python\n{task}```\n"
)


class UnknownSplit(IndexError, ValueError):
    """No such split. An `IndexError` because OpenEnv's task routes answer only that one with a
    400 and every other exception with a 500; a `ValueError` for `reset`."""


class KernelAction(Action):
    """A model's completion: the answer's code is the last fenced Python block in it."""

    completion: str


class KernelReset(ResetRequest):
    """What `reset` takes besides OpenEnv's `seed` and `episode_id`, so that `/schema` shows it."""

    split: str = "train"
    index: int | None = None
    task: str | None = None


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
    """`hacks` counts the attempts of the episode that reached into the checker: the last
    observation shows only its own."""

    task_id: str = ""
    attempts: int = 0
    verified: bool = False
    hacks: int = 0


class VerdictRubric(Rubric):
    """The judge's score as the reward; an abstention is `abstain_value`, 0.0 when it is None."""

    def __init__(self, abstain_value: float | None = None) -> None:
        super().__init__()
        self.abstain_value = abstain_value

    def forward(self, action: Any, observation: Any) -> float:
        score = observation.score
        if score is not None:
            return float(score)
        return 0.0 if self.abstain_value is None else float(self.abstain_value)


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
        abstain_value: float | None = None,
    ) -> None:
        super().__init__(rubric=VerdictRubric(abstain_value))
        if max_attempts < 1:
            raise ValueError("max_attempts is at least 1")
        if judge is not None and judge.mode != "cpu":
            raise ValueError(
                f"the reward is the judge's CPU-only score: Judge(mode='cpu'), not {judge.mode!r}"
            )
        if isinstance(tasks, str):
            raise TypeError("tasks is a list of references or a mapping of lists, not a string")
        self._splits: dict[str, list[str]] = {}
        if isinstance(tasks, Mapping):
            self._splits = {name: list(pool) for name, pool in tasks.items()}
        elif tasks is not None:
            self._splits = {"train": list(tasks)}
        self._judge = judge
        self._abstain_value = abstain_value
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
            name=ENV_NAME,
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
        if kwargs:
            raise TypeError(
                f"reset takes split, index, task, seed and episode_id, not {sorted(kwargs)}"
            )
        if task is not None:
            if not task.strip():
                raise ValueError("task is empty")
            task_id, source = "adhoc", task
        else:
            pool = self._pool(split)
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
            task_id=task_id,
            prompt=PROMPT.format(task=source.rstrip() + "\n"),
            done=False,
            reward=None,
        )

    def step(
        self, action: KernelAction, timeout_s: float | None = None, **kwargs: Any
    ) -> KernelObservation:
        """`timeout_s` is not used: the judge's own timeouts (`Judge(timeout_s=)`) apply."""
        if not self._task:
            raise RuntimeError("step before reset")
        if self._state.verified or self._state.attempts >= self.max_attempts:
            raise RuntimeError("the episode is over: reset")
        judge = self.judge
        v: Verdict = judge.score(self._task, extract_answer(action.completion))
        self._state.attempts += 1
        self._state.step_count += 1
        self._state.hacks += int(v.hack)
        self._state.verified = v.score is not None and v.score >= 1.0
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
        assert isinstance(self.rubric, VerdictRubric)
        self.rubric.abstain_value = (
            self._abstain_value if self._abstain_value is not None else judge.abstain_value
        )
        obs.reward = self.rubric(action, obs)
        obs.done = self._state.verified or self._state.attempts >= self.max_attempts
        return obs

    async def step_async(
        self, action: KernelAction, timeout_s: float | None = None, **kwargs: Any
    ) -> KernelObservation:
        return await asyncio.to_thread(self.step, action, timeout_s, **kwargs)

    # ------------------------------------------------------------ task discovery (TaskProvider)

    def _pool(self, split: str) -> list[str]:
        if split not in self._splits:
            raise UnknownSplit(f"no split {split!r}: {sorted(self._splits)}")
        return self._splits[split]

    def _index(self, split: str, index: int) -> int:
        n = len(self._pool(split))
        if not 0 <= index < n:
            raise IndexError(f"split {split!r} has {n} tasks, not {index}")
        return index

    def list_splits(self) -> list[str]:
        return sorted(self._splits)

    def num_tasks(self, split: str) -> int:
        return len(self._pool(split))

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
    abstain_value: float | None = None,
    max_concurrent_envs: int | None = None,
) -> FastAPI:
    """The OpenEnv server for `KernelEnv`: `uvicorn.run(create_kernel_app(tasks, ...))`.
    `max_concurrent_envs` is OpenEnv's: left out, the server takes one WebSocket session at a time
    and a second one is refused with CAPACITY_REACHED; parallel rollouts need it above 1 (the
    sessions share the judge's worker pool, so `Judge(workers=)` is the other ceiling)."""
    return create_fastapi_app(
        lambda: KernelEnv(tasks, judge, max_attempts=max_attempts, abstain_value=abstain_value),
        KernelAction,
        KernelObservation,
        max_concurrent_envs=max_concurrent_envs,
        env_name=ENV_NAME,
        state_cls=KernelState,
        reset_cls=KernelReset,
    )
