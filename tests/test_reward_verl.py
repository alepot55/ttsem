"""The verl adapter with a fake worker: the dict verl reads, the gate over a base reward, the
same logged keys for every sample, and loading the file by path as verl does."""

from __future__ import annotations

import asyncio
import importlib.util
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from test_reward import QUALIFIES, ROWS, TASK, UNSAFE, VERIFIED, WRONG, Fake

from ttsem.reward import verl_adapter
from ttsem.reward.core import Judge

COMPLETION = f"Here is the kernel.\n```python\n{ROWS}```\n"


@pytest.fixture
def judges(tmp_path: Path) -> Iterator[dict[str, Fake]]:
    """Both modes' judges, each over its own fake worker."""
    fakes = {"cpu": Fake(scale=VERIFIED, medium=VERIFIED), "gate": Fake(scale=WRONG)}
    verl_adapter._JUDGES.clear()
    for mode, fake in fakes.items():
        verl_adapter._JUDGES[mode] = Judge(cache_dir=tmp_path / mode, runner=fake, mode=mode)
    yield fakes
    verl_adapter._JUDGES.clear()


def test_compute_score_returns_a_float_score_and_the_fields(judges: dict[str, Fake]) -> None:
    out = verl_adapter.compute_score("kernelbench", COMPLETION, TASK, {"uuid": "1"})
    assert out["score"] == 1.0 and isinstance(out["score"], float)
    assert out["ttsem_verdict"] == "verified" and out["ttsem_tier"] == "medium"
    assert set(out) == {"score", *verl_adapter.SKIPPED}


def test_acompute_score(judges: dict[str, Fake]) -> None:
    out = asyncio.run(verl_adapter.acompute_score("kb", COMPLETION, TASK))
    assert out["score"] == 1.0


def test_an_abstention_is_a_float_for_verl(judges: dict[str, Fake]) -> None:
    judges["cpu"].by_tier = {"scale": {"verdict": "too_slow", "answer": "x.py"}}
    out = verl_adapter.compute_score("kb", COMPLETION, TASK)
    assert out["score"] == 0.0 and out["ttsem_abstain"] is True


def test_gated_only_lowers_the_base(judges: dict[str, Fake]) -> None:
    calls: list[str] = []

    def base(data_source: Any, solution_str: str, ground_truth: str, extra_info: Any) -> Any:
        calls.append(data_source)
        return {"score": 0.8, "speedup": 1.6}

    out = verl_adapter.gated(base)("kb", COMPLETION, TASK, None)
    assert out["score"] == 0.0 and out["base_score"] == 0.8 and out["speedup"] == 1.6
    assert out["ttsem_verdict"] == "wrong" and calls == ["kb"]
    judges["gate"].by_tier = {"scale": VERIFIED, "medium": VERIFIED}
    assert verl_adapter.gated(base)("kb", COMPLETION, TASK, None)["score"] == 0.8


def test_gated_lazy_skips_what_the_base_rejected(judges: dict[str, Fake]) -> None:
    judges["gate"].by_tier = {"scale": VERIFIED, "scale_cap": UNSAFE}
    reward = verl_adapter.gated(lambda *a: 0.0, lazy=True)
    skipped = reward("kb", f"```python\n{QUALIFIES}```", TASK, None)
    assert skipped["score"] == 0.0 and skipped["ttsem_verdict"] == "skipped"
    assert judges["gate"].asked == []
    judged = verl_adapter.gated(lambda *a: 1.0, lazy=True)("kb", f"```python\n{QUALIFIES}```", TASK)
    assert judged["score"] == 0.0 and judged["ttsem_kind"] == "race"
    assert set(skipped) == set(judged)  # verl stacks the logged keys across a batch


def test_loads_by_path_as_verl_does(judges: dict[str, Fake]) -> None:
    path = Path(verl_adapter.__file__)
    spec = importlib.util.spec_from_file_location("custom_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.compute_score) and callable(module.gated)
