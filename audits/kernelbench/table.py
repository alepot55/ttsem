"""The judge's verdicts on a KernelBench-style dataset, against the dataset's own labels.

    python audits/kernelbench/table.py OUT_DIR [OUT_DIR ...] --manifest M.jsonl [--manifest ...]
                                       [--by model|problem|gpu]

Reads the reports `python -m ttsem judge --manifest M.jsonl --out OUT_DIR` writes
(OUT_DIR/<id>.scaled.json, and <id>.full.json where the full-shape pass ran; the first directory
that has an id wins) for the answers of the manifests, and prints, per dataset (`source`):

- the verdicts of the scaled pass (every answer), by the dataset's label;
- the same in detail: wrong by values, by shape or type, under another autotune config, with
  weights the v3 harness cannot copy, or with memory nobody wrote; unsafe by kind; error and
  not_judged by stage;
- the verdicts of the full-shape pass, where it ran;
- where the judge decides (verified, wrong, unsafe, no_kernel), how many of each label it verifies;
- with `--by`, the answers labelled correct, by that field.

The verdicts are taken as the judge wrote them. An id that a manifest lists twice with two labels
(the v3 release evaluated some (model, GPU, problem) twice but ships one file) is `ambiguous`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ORDER = ("verified", "wrong", "unsafe", "no_kernel", "error", "too_slow", "not_judged")
DECIDED = ("verified", "wrong", "unsafe", "no_kernel")
AGAINST = ("wrong", "unsafe", "no_kernel")


def load(manifests: list[Path], dirs: list[Path]) -> list[dict[str, Any]]:
    """One row per answer of the manifests that has a report: its manifest fields, `scaled`, and
    `full` where that pass ran."""
    items: dict[str, dict[str, Any]] = {}
    labels: dict[str, set[str]] = defaultdict(set)
    for manifest in manifests:
        for text in manifest.read_text(encoding="utf-8").splitlines():
            if text.strip():
                item = json.loads(text)
                labels[item["id"]].add(str(item.get("label")))
                items.setdefault(item["id"], item)
    rows = []
    for key, item in items.items():
        found = next((d for d in dirs if (d / f"{key}.scaled.json").exists()), None)
        if found is None:
            continue
        row = {**item, "scaled": json.loads((found / f"{key}.scaled.json").read_text())}
        full = next(
            (d / f"{key}.full.json" for d in [found, *dirs] if (d / f"{key}.full.json").exists()),
            None,
        )
        if full is not None:
            row["full"] = json.loads(full.read_text())
        if len(labels[key]) > 1:
            row["label"] = "ambiguous"
        rows.append(row)
    return rows


def detail(report: dict[str, Any]) -> str:
    """The verdict with its kind."""
    verdict = str(report["verdict"])
    if verdict == "unsafe":
        return f"unsafe: {report.get('kind')}"
    if verdict == "wrong":
        trials = report.get("trials") or []
        if any("shape" in t or "why" in t for t in trials):
            return "wrong: shape or type"
        if "failed" in (report.get("configs") or {}):  # right under the first config only
            return "wrong: under another autotune config"
        if report.get("not_copied"):  # the v3 harness copies weights by name: these stayed its own
            return "wrong: holds weights of its own (not copied by name)"
        if any(
            t.get("max_abs_diff") != t.get("max_abs_diff") and not t.get("ref_nan") for t in trials
        ):  # NaN where the reference has none: the poisoned allocation, never written
            return "wrong: depends on memory nobody wrote (NaN)"
        return "wrong: values"
    if verdict in ("error", "not_judged"):
        return f"{verdict}: {report.get('why') or report.get('stage') or ''}".rstrip(": ")
    return verdict


def table(rows: list[dict[str, Any]], key: str = "scaled") -> str:
    labels = sorted({str(r["label"]) for r in rows})
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        if key in r:
            counts[str(r["label"])][str(r[key]["verdict"])] += 1
    lines = ["| verdict | " + " | ".join(f"labelled {lab}" for lab in labels) + " |"]
    lines.append("|---|" + "---|" * len(labels))
    for v in ORDER:
        if any(counts[lab][v] for lab in labels):
            lines.append(f"| {v} | " + " | ".join(str(counts[lab][v]) for lab in labels) + " |")
    lines.append("| total | " + " | ".join(str(sum(counts[lab].values())) for lab in labels) + " |")
    return "\n".join(lines)


def detail_table(rows: list[dict[str, Any]]) -> str:
    labels = sorted({str(r["label"]) for r in rows})
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        counts[detail(r["scaled"])][str(r["label"])] += 1
    lines = ["| verdict, in detail | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for key in sorted(counts):
        lines.append(f"| {key} | " + " | ".join(str(counts[key][lab]) for lab in labels) + " |")
    return "\n".join(lines)


def decided(rows: list[dict[str, Any]]) -> list[str]:
    found = []
    for label in sorted({str(r["label"]) for r in rows}):
        verdicts = Counter(str(r["scaled"]["verdict"]) for r in rows if r["label"] == label)
        judged = sum(verdicts[v] for v in DECIDED)
        if judged:
            share = f"{100 * verdicts['verified'] / judged:.0f}%"
            against = sum(verdicts[v] for v in AGAINST)
            found.append(
                f"labelled {label}: the judge decides on {judged}, verifies {verdicts['verified']} "
                f"({share}), and judges {against} wrong, unsafe or not a kernel"
            )
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("dirs", type=Path, nargs="+", help="the judge's --out directories")
    ap.add_argument("--manifest", type=Path, action="append", required=True)
    ap.add_argument("--by", default="", help="group the answers labelled correct by this field")
    args = ap.parse_args(argv)
    rows = load(args.manifest, args.dirs)
    for source in sorted({str(r.get("source")) for r in rows}):
        mine = [r for r in rows if str(r.get("source")) == source]
        print(f"## {source}: {len(mine)} answers judged\n\n{table(mine)}\n\n{detail_table(mine)}\n")
        full = [r for r in mine if "full" in r]
        if full:
            print(f"full-shape pass, {len(full)} answers:\n\n{table(full, 'full')}\n")
        print("\n".join(decided(mine)) + "\n")
        if args.by:
            groups: dict[str, Counter[str]] = defaultdict(Counter)
            for r in mine:
                if r["label"] == "correct":
                    groups[str(r.get(args.by))][str(r["scaled"]["verdict"])] += 1
            print(f"labelled correct, by {args.by}:")
            for name, count in sorted(groups.items()):
                print(f"  {name}: " + ", ".join(f"{v} {count[v]}" for v in ORDER if count[v]))
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
