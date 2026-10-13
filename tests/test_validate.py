"""The per-pass validator: stages it cannot read back, and the summary under a launch."""

from __future__ import annotations

from ttsem import validate


def test_an_unreadable_stage_is_classified_by_its_diagnostic() -> None:
    printer = validate._unreadable_stage(
        "triton-opt cannot re-read the module: <stage>:90:5: error: cannot name an operation "
        "with no results"
    )
    assert printer is not None and "11752" in printer
    nested = validate._unreadable_stage(
        "<stage>:37:5: error: 'ttg.warp_specialize.partitions' op region #0 ('partitionRegions')"
        " failed to verify constraint: region with at least 1 blocks"
    )
    assert nested is not None and "nested" in nested
    assert validate._unreadable_stage("something else entirely") is None


def _report(*verdicts: str) -> validate.Report:
    """A report over stages `Before P0: p0`, `Before P1: p1`, ... with these verdicts, read as
    `validate_stages` reads them."""
    results = [
        validate.PassResult(f"Before P{i}: p{i}", v, 3 if v == "mismatch" else 0, [])
        for i, v in enumerate(verdicts)
    ]
    first_bad = next((r.pass_name for r in results if r.verdict == "mismatch"), None)
    idx = validate.changed_by_index(results)
    changed_by = results[idx].pass_name if idx is not None else None
    culprit = validate.culprit_of(results)
    return validate.Report("p.py", results, first_bad, culprit, "k", changed_by)


def test_the_summary_leads_with_the_pass_that_changes_the_meaning() -> None:
    """The source is the reference: P1's output is the first to disagree, so P1 leads; the first
    stage that differs is P2's input, named after it and not as the pass to blame."""
    lines = validate.summary_lines(_report("match", "match", "mismatch", "mismatch"))
    assert lines == [
        "first pass that changes the meaning: Before P1: p1",
        "first stage whose input differs: Before P2: p2",
    ]


def test_the_summary_names_the_culprit_when_the_device_is_the_wrong_one() -> None:
    """A device that ran a miscompiled kernel agrees with the IR only after the pass that broke
    it: that pass leads, and the first stage that differs is the very first one."""
    lines = validate.summary_lines(_report("mismatch", "mismatch", "match", "match"))
    assert lines[0] == "first pass that changes the meaning: Before P1: p1"
    assert lines[1] == "first stage whose input differs: Before P0: p0"
    assert lines[2].startswith("culprit: Before P1: p1")


def test_the_summary_when_no_pass_is_named() -> None:
    agree = validate.summary_lines(_report("match", "unsupported", "match"))
    assert agree == [
        "no pass changes the meaning: every stage agrees with the reference",
        "first stage whose input differs: none",
    ]
    differ = validate.summary_lines(_report("mismatch", "mismatch"))
    assert differ[0].startswith("no pass is named")
