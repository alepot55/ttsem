"""The `nvws` phase of a warp-specialized loop, run in program order with a slot cursor per side.

`nvws-insert-aref` turns the multi-buffered descriptor loads into `nvws.aref.*` while the loop
is still one body; every stage from there to `nvws-lower-aref` must agree with the reference.
The stages after `tritongpu-partition-loops` (a `nvws.warp_group`, then partitions that wait on
barriers) are the scheduler's business and stay `unsupported` here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ttsem import harness, validate

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def ws_stages():
    pytest.importorskip("triton")
    record = harness.capture_launches(FIXTURES / "ws_desc.py", "cpu", cc=100)[0]
    stages = validate.split_dump(harness.dump_for_launch(record, harness.target_for("cpu", 100)))
    return record, [(n, t) for n, t in stages if "llvm.func" not in t]


def test_the_aref_stages_of_a_warp_specialized_matmul_match(ws_stages) -> None:
    record, tile = ws_stages
    first_split = next(
        i
        for i, (_, text) in enumerate(tile)
        if "nvws.warp_group" in text or "warp_specialize.partitions" in text
    )
    before = tile[:first_split]
    aref_stages = [i for i, (_, text) in enumerate(before) if "nvws.aref.create" in text]
    assert len(aref_stages) >= 5, "the fixture should spend several stages in the aref form"
    report = validate.validate_stages(record, before, None, "ws_desc")
    verdicts = {i: r.verdict for i, r in enumerate(report.passes)}
    assert all(verdicts[i] == "match" for i in aref_stages), verdicts
    assert report.first_bad_pass is None


def test_the_partitioned_stages_match_under_the_scheduler(ws_stages) -> None:
    """After `tritongpu-partition-loops` the producer and the consumer are partitions that
    hand over on mbarrier phases (and, one stage earlier, on aref slots): every tile stage
    down to the LLVM conversion must agree with the reference."""
    record, tile = ws_stages
    first_split = next(
        i
        for i, (_, text) in enumerate(tile)
        if "nvws.warp_group" in text or "warp_specialize.partitions" in text
    )
    report = validate.validate_stages(record, tile[first_split:], None, "ws")
    verdicts = [r.verdict for r in report.passes]
    assert verdicts and set(verdicts) == {"match"}, list(
        zip(verdicts, [n[:50] for n, _ in tile[first_split:]], strict=True)
    )
    assert report.first_bad_pass is None
