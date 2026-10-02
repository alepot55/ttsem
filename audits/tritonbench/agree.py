"""How often the GPU-free verdict and the device's say the same thing, answer by answer.

    python audits/tritonbench/agree.py JUDGE_OUT/rows.jsonl DEVICE_OUT/rows.jsonl [--list]

The first file holds the rows of `python -m ttsem judge --tritonbench --manifest ... --out
JUDGE_OUT`, the second those of `device.py`, both for the same manifest. Each side's class is put
in a coarse class on purpose: did not run (the file raised), right (verified at any tolerance),
wrong (other values), fault (out of bounds, race, use of memory nobody wrote: the semantics only;
a GPU runs past them), not comparable, not judged. The agreement is counted over the answers both
sides decide: did not run, right or wrong. `--list` prints the answers where those two disagree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

RIGHT = ("verified", "verified_lenient", "close_reduced_precision")
FAULTS = ("oob_read", "oob_write", "race", "poison")
NOT_JUDGED = (
    "unsupported", "timeout", "memory", "tool_error", "crash", "error", "environment", "bad_task",
)  # fmt: skip
CLASSES = ("did_not_run", "right", "wrong", "fault", "not_comparable", "not_judged")
DECIDED = ("did_not_run", "right", "wrong")


def coarse(name: str) -> str:
    if name in RIGHT:
        return "right"
    if name == "wrong_result":
        return "wrong"
    if name == "call_error":
        return "did_not_run"
    if name in FAULTS:
        return "fault"
    if name in NOT_JUDGED:
        return "not_judged"
    return "not_comparable"


def classes(path: Path, key: str) -> dict[str, str]:
    """The class of each answer's record (`key`: `tritonbench` or `device`), by id."""
    found: dict[str, str] = {}
    for text in path.read_text(encoding="utf-8").splitlines():
        if text.strip():
            row: dict[str, Any] = json.loads(text)
            found[str(row["id"])] = str(row[key].get("class"))
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("judge", type=Path, help="the judge's rows.jsonl")
    ap.add_argument("device", type=Path, help="device.py's rows.jsonl")
    ap.add_argument("--list", action="store_true", help="print the answers the sides disagree on")
    args = ap.parse_args(argv)
    ours, theirs = classes(args.judge, "tritonbench"), classes(args.device, "device")
    matrix: dict[tuple[str, str], int] = {}
    differ: list[str] = []
    for key in sorted(ours.keys() & theirs.keys()):
        pair = coarse(ours[key]), coarse(theirs[key])
        matrix[pair] = matrix.get(pair, 0) + 1
        if pair[0] != pair[1] and pair[0] in DECIDED and pair[1] in DECIDED:
            differ.append(f"{key}: ttsem {ours[key]}, device {theirs[key]}")
    print(f"{len(ours.keys() & theirs.keys()):,} answers on both sides\n")
    print("| ttsem \\ device | " + " | ".join(CLASSES) + " |")
    print("|---|" + "---|" * len(CLASSES))
    for a in CLASSES:
        print(f"| {a} | " + " | ".join(str(matrix.get((a, b), 0)) for b in CLASSES) + " |")
    both = sum(matrix.get((a, b), 0) for a in DECIDED for b in DECIDED)
    same = sum(matrix.get((a, a), 0) for a in DECIDED)
    if both:
        print(
            f"\nBoth sides decide on {both:,} answers and agree on {same:,} "
            f"({100 * same / both:.1f}%)."
        )
    if args.list:
        print("\n" + "\n".join(differ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
