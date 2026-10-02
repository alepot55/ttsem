"""The summary table of the TritonBench audit, from the judge's rows.

    python audits/tritonbench/table.py OUT/rows.jsonl [MORE/rows.jsonl ...]

Reads the rows `python -m ttsem judge --tritonbench --manifest ... --out OUT` writes (one per
answer: the manifest's fields, `bench` and `model` among them, and the record) and prints, per
benchmark (G, T), one line per model and the total, in the columns of
`ttsem.judge_tritonbench.columns`:

- ran: the file ran to the end, which is all TritonBench's own check asks, plus the answers the
  semantics stopped for a fault that a GPU runs past;
- verified (strict or lenient tolerance), reduced precision, wrong, unsafe;
- not compared: a task or answer that draws random numbers, a reference that cannot be compared
  with, a fault the task's own test causes in the reference too, right values with no kernel;
- not judged: what this machine or the tool lacks, a timeout, a crash;
- wrong or unsafe, out of the answers that ran and could be judged.

An id in several files counts once, from the last file that has it.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from ttsem.judge_tritonbench import RECORD, columns

KEYS = ("answers", "ran", "verified", "reduced", "wrong", "unsafe", "not_compared", "not_judged")
HEAD = (
    "| model | answers | ran | verified | reduced precision | wrong | unsafe | not compared "
    "| not judged | wrong or unsafe |\n" + "|---" * 10 + "|"
)


def bad_of(count: dict[str, int]) -> tuple[int, int]:
    """Wrong or unsafe, and the answers that ran and could be judged."""
    judged = count["verified"] + count["reduced"] + count["wrong"] + count["unsafe"]
    return count["wrong"] + count["unsafe"], judged


def name_of(model: str) -> str:
    """A model as the benchmark's tables name it (its general-purpose files start `output_`)."""
    return model.removeprefix("output_")


def line(name: str, count: dict[str, int], bold: bool = False) -> str:
    bad, judged = bad_of(count)
    cells = [f"{count[k]:,}" for k in KEYS] + [f"{bad} of {judged}"]
    if bold:
        cells = [f"**{c}**" for c in cells]
        name = f"**{name}**"
    return f"| {name} | " + " | ".join(cells) + " |"


def load(paths: list[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        for text in path.read_text(encoding="utf-8").splitlines():
            if text.strip():
                row = json.loads(text)
                rows[str(row["id"])] = row
    return rows


def tables(rows: dict[str, dict[str, Any]]) -> list[str]:
    by_bench: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows.values():
        bench = str(row.get("bench") or "?")
        by_bench[bench][str(row.get("model") or "?")].append(row[RECORD])
    out: list[str] = []
    every: list[dict[str, Any]] = []
    for bench in sorted(by_bench):
        models = by_bench[bench]
        out += ["", f"### TritonBench-{bench}", "", HEAD]
        records: list[dict[str, Any]] = []
        for model in sorted(models, key=lambda m: name_of(m).lower()):
            out.append(line(name_of(model), columns(models[model])))
            records += models[model]
        total = columns(records)
        out.append(line("total", total, bold=True))
        bad, judged = bad_of(total)
        if judged:
            share = f"{100 * bad / judged:.0f}%"
            text = f"of the {judged:,} answers that ran and could be judged, {bad:,} ({share})"
            out += ["", f"{bench}: {text} return wrong values or are unsafe."]
        every += records
    total = columns(every)
    bad, judged = bad_of(total)
    if judged and len(by_bench) > 1:
        share = f"{100 * bad / judged:.0f}%"
        out += [
            "",
            (
                f"All: {total['answers']:,} answers; of the {judged:,} that ran and could be "
                f"judged, {bad:,} ({share}) return wrong values ({total['wrong']:,}) or are "
                f"unsafe ({total['unsafe']:,})."
            ),
        ]
    return out


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0].startswith("-"):
        sys.stderr.write(__doc__ or "")
        return 2
    print("\n".join(tables(load([Path(a) for a in args]))).lstrip("\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
