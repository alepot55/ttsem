"""The per-pass validator's classification of stages it cannot read back."""

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
