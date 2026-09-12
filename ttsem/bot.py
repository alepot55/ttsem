"""The bot's bookkeeping: what changed between two runs of the suite under the validator.

Each run is a directory of JSONL files written by ``ttsem/pytest_ttsem.py`` (one row per launch:
``nodeid``, ``launch``, ``fn``, ``stage``, ``verdict``, ``n_diff``, ``message``). The diff
lists the launches whose verdict moved into or out of the bad set, with the tests known to be
nondeterministic on the device flagged rather than hidden.

    python -m ttsem.bot summary run/
    python -m ttsem.bot diff previous/ current/      # exit 1 when something new is bad
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

BAD = {"mismatch", "poison", "error"}
ORDER = ["match", "approx", "poison", "unsupported", "mismatch", "error"]
# Tests whose verdict moves between two device runs of the same commit (SEMANTICS.md, "Device
# deviations"): the old value an atomic returns, NaN payloads.
NONDETERMINISTIC = ("test_atomic_rmw", "test_atomic_cas", "test_propagate_nan", "test_globaltimer")

Key = tuple[str, str, str]  # (nodeid, launch, stage)


def load(run: Path) -> dict[Key, dict[str, Any]]:
    rows: dict[Key, dict[str, Any]] = {}
    for f in sorted(run.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows[(str(r["nodeid"]), str(r["launch"]), str(r.get("stage", "")))] = r
    return rows


def counts(rows: dict[Key, dict[str, Any]]) -> dict[str, int]:
    c = collections.Counter(str(r["verdict"]) for r in rows.values())
    return {v: c[v] for v in ORDER if c[v]}


def is_flagged(nodeid: str) -> bool:
    test = nodeid.split("::")[-1].split("[")[0]
    return (
        test in NONDETERMINISTIC or "atomic" in test
    )  # every atomic test orders lanes on the device


@dataclasses.dataclass
class Diff:
    previous: dict[str, int]
    current: dict[str, int]
    new_bad: list[dict[str, Any]]  # bad now, not bad (or absent) before
    gone_bad: list[dict[str, Any]]  # bad before, not bad now
    transitions: dict[tuple[str, str], int]

    @property
    def new_unflagged(self) -> list[dict[str, Any]]:
        return [r for r in self.new_bad if not is_flagged(str(r["nodeid"]))]


def diff(prev: dict[Key, dict[str, Any]], cur: dict[Key, dict[str, Any]]) -> Diff:
    new_bad, gone_bad = [], []
    trans: collections.Counter[tuple[str, str]] = collections.Counter()
    for key, r in cur.items():
        before = prev.get(key)
        old = str(before["verdict"]) if before else "absent"
        new = str(r["verdict"])
        if old != new:
            trans[(old, new)] += 1
        if new in BAD and old not in BAD:
            new_bad.append({**r, "was": old})
    for key, r in prev.items():
        after = cur.get(key)
        new = str(after["verdict"]) if after else "absent"
        if str(r["verdict"]) in BAD and new not in BAD:
            gone_bad.append({**r, "now": new})
    return Diff(counts(prev), counts(cur), new_bad, gone_bad, dict(trans))


def _row(r: dict[str, Any], extra: str) -> str:
    flag = " (nondeterministic on the device)" if is_flagged(str(r["nodeid"])) else ""
    msg = str(r.get("message") or "")[:100]
    return f"- `{r['nodeid']}` launch {r['launch']}: {extra}{flag} {msg}".rstrip()


def render(d: Diff) -> str:
    out = [f"previous: {d.previous}", f"current:  {d.current}", ""]
    if d.transitions:
        out.append("transitions:")
        for (a, b), n in sorted(d.transitions.items(), key=lambda kv: -kv[1]):
            out.append(f"- {a} -> {b}: {n}")
        out.append("")
    out.append(f"new bad ({len(d.new_bad)}, {len(d.new_unflagged)} not flagged):")
    out += [_row(r, f"{r['was']} -> {r['verdict']}") for r in d.new_bad] or ["- none"]
    out.append("")
    out.append(f"no longer bad ({len(d.gone_bad)}):")
    out += [_row(r, f"{r['verdict']} -> {r['now']}") for r in d.gone_bad] or ["- none"]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary")
    s.add_argument("run", type=Path)
    d = sub.add_parser("diff")
    d.add_argument("previous", type=Path)
    d.add_argument("current", type=Path)
    args = ap.parse_args()
    if args.cmd == "summary":
        rows = load(args.run)
        print(f"{len(rows)} launches {counts(rows)}")
        return 0
    result = diff(load(args.previous), load(args.current))
    sys.stdout.write(render(result))
    return 1 if result.new_unflagged else 0


if __name__ == "__main__":
    sys.exit(main())
