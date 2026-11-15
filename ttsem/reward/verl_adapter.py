"""verl's custom reward function over the exact judge: no trainer code of ours to write.

CPU-only mode (the binary correctness reward, no GPU in the reward loop), in the key verl reads
since 0.7.1 (a top-level `custom_reward_function` in 0.6)::

    reward:
      custom_reward_function:
        path: <this file>
        name: compute_score      # or acompute_score under verl's async reward loop

Gate mode (keep an existing GPU reward for a verified answer, lower it for a rejected one): a
three-line file of the trainer's own, named in `reward.custom_reward_function.path`::

    from ttsem.reward.verl_adapter import gated
    from my_rewards import compute_score as base   # the GPU reward, verl's signature
    compute_score = gated(base)

The judge is configured by environment variables (`Judge.from_env`: TTSEM_REWARD_PYTHON,
TTSEM_REWARD_TTSEM_ROOT, TTSEM_REWARD_CACHE, TTSEM_REWARD_WORKERS, ...). `ground_truth` is the
task's PyTorch reference module, a string. Each call returns a dict: `score` (a float: verl needs
one) and the `ttsem_*` fields verl logs per sample next to it. The four arguments, the dict with a
`score` key and the async function follow the source of verl 0.9.1 (its documentation page gives
only the signature); this module is tested against a fake worker, not against verl itself: verify
it on the verl you pin.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

if __package__ in (None, "") and importlib.util.find_spec("ttsem") is None:
    # loaded by path, as verl loads a custom reward function, from a checkout that is not installed
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ttsem.reward.core import Judge, Verdict, extract_answer  # noqa: E402

BaseScore = Callable[..., Any]

_JUDGES: dict[str, Judge] = {}
_LOCK = threading.Lock()


def judge(mode: str) -> Judge:
    """The process's judge for `mode` (cpu or gate), configured from the environment once."""
    with _LOCK:
        if mode not in _JUDGES:
            _JUDGES[mode] = Judge.from_env(mode=mode)
        return _JUDGES[mode]


def fields(v: Verdict) -> dict[str, Any]:
    """What verl logs per sample next to the score."""
    return {
        "ttsem_verdict": v.verdict,
        "ttsem_kind": v.kind or "",
        "ttsem_tier": v.tier,
        "ttsem_abstain": v.abstain,
        "ttsem_hack": v.hack,
        "ttsem_seconds": v.seconds,
        "ttsem_feedback": v.feedback,
    }


# a sample `gated(..., lazy=True)` did not judge, the base having rejected it
SKIPPED: dict[str, Any] = {
    "ttsem_verdict": "skipped",
    "ttsem_kind": "",
    "ttsem_tier": "",
    "ttsem_abstain": False,
    "ttsem_hack": False,
    "ttsem_seconds": 0.0,
    "ttsem_feedback": "",
}


def compute_score(
    data_source: Any, solution_str: str, ground_truth: str, extra_info: Any = None, **_: Any
) -> dict[str, Any]:
    """CPU-only mode: the reward is the judge's score; an abstention is the judge's
    `abstain_value`, 0.0 when it has none (verl needs a float)."""
    j = judge("cpu")
    v = j.score(ground_truth, extract_answer(solution_str))
    score = v.score if v.score is not None else (j.abstain_value or 0.0)
    return {"score": float(score), **fields(v)}


async def acompute_score(
    data_source: Any, solution_str: str, ground_truth: str, extra_info: Any = None, **kw: Any
) -> dict[str, Any]:
    return await asyncio.to_thread(
        compute_score, data_source, solution_str, ground_truth, extra_info, **kw
    )


def _base(value: Any) -> tuple[float, dict[str, Any]]:
    """A base reward as verl's functions return it: a number, or a dict with `score`."""
    if isinstance(value, dict):
        return float(value["score"]), {k: v for k, v in value.items() if k != "score"}
    return float(value), {}


def gated(base_score: BaseScore, *, lazy: bool = False) -> BaseScore:
    """Gate mode over `base_score` (verl's signature). The judge can only lower the base reward
    (`Judge.gate`). By default the base and the judge run at once; with `lazy` the judge runs only
    when the base is above the judge's `fail_value` (CPU saved, the two latencies added)."""
    pool = ThreadPoolExecutor(max_workers=4)

    def compute(
        data_source: Any, solution_str: str, ground_truth: str, extra_info: Any = None, **kw: Any
    ) -> dict[str, Any]:
        j = judge("gate")
        answer = extract_answer(solution_str)
        if lazy:
            base, extra = _base(
                base_score(data_source, solution_str, ground_truth, extra_info, **kw)
            )
            if base <= j.fail_value:  # the same keys as a judged sample: verl stacks them
                return {**extra, "score": base, "base_score": base, **SKIPPED}
            v = j.score(ground_truth, answer)
        else:
            later = pool.submit(
                base_score, data_source, solution_str, ground_truth, extra_info, **kw
            )
            v = j.score(ground_truth, answer)
            base, extra = _base(later.result())
        return {**extra, "score": j.gate(base, v), "base_score": base, **fields(v)}

    return compute
