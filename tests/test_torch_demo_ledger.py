"""What `python -m ttsem demo --torch` says about upstream agrees with LEDGER.md.

Each case of the demo prints a status line (`upstream  fixed on main by pytorch#N (D Mon YYYY)`).
The ledger is where the state of an issue is kept up to date, so a status the demo prints must
say the same thing: the same fix, the same day. No torch is needed to read the cases.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pytest

from ttsem import torch_demo

LEDGER = Path(__file__).resolve().parent.parent / "LEDGER.md"

FIXED = re.compile(r"fixed on main by pytorch#(\d+) \((\d{1,2} \w{3} \d{4})\)")
PROPOSED = re.compile(r"fix proposed in pytorch#(\d+)")


def _ledger_row(issue: str) -> tuple[str, str]:
    """The state and fix cells of the ledger row of `pytorch#N`."""
    number = issue.removeprefix("pytorch#")
    link = f"(https://github.com/pytorch/pytorch/issues/{number})"
    rows = [
        line for line in LEDGER.read_text().splitlines() if line.startswith("| [") and link in line
    ]
    assert len(rows) == 1, f"{issue}: {len(rows)} ledger rows link the issue"
    cells = [cell.strip() for cell in rows[0].strip().strip("|").split("|")]
    return cells[-2], cells[-1]


def _dates(text: str) -> set[dt.date]:
    """Every date written as `2026-09-22`, `22 Sep 2026` or `1 Oct 2026`."""
    found = {dt.date.fromisoformat(d) for d in re.findall(r"\d{4}-\d{2}-\d{2}", text)}
    for day in re.findall(r"\d{1,2} [A-Z][a-z]{2} \d{4}", text):
        found.add(dt.datetime.strptime(day, "%d %b %Y").date())
    return found


@pytest.mark.parametrize("case", torch_demo.CASES, ids=lambda case: case.issue)
def test_the_status_the_demo_prints_is_the_one_in_the_ledger(case: torch_demo.Case) -> None:
    state, fix = _ledger_row(case.issue)
    if m := FIXED.fullmatch(case.status):
        pr, day = m.group(1), dt.datetime.strptime(m.group(2), "%d %b %Y").date()
        assert state.startswith("fixed"), (case.issue, state)
        assert f"/pull/{pr})" in fix and "merged" in fix, (case.issue, fix)
        assert day in _dates(f"{state} {fix}"), (case.issue, day, state, fix)
    elif m := PROPOSED.fullmatch(case.status):
        assert not state.startswith("fixed"), (case.issue, state)
        assert f"/pull/{m.group(1)})" in fix and "open" in fix, (case.issue, fix)
    else:
        pytest.fail(f"{case.issue}: a status this test does not know how to check: {case.status!r}")
