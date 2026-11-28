"""The spend of the floating-point session: every model call recorded under ``results/``, at list
price as the CLI reports it.

Each call is counted once: the agentic prover's attempts are found wherever a record embeds them
and told apart by their sandbox; the one-shot rounds of ``fpbound.certify`` by their records'
totals; the proposers by ``cost_usd_proposer`` and ``proposer_cost_usd``; the certified search by
the proposer's and the certifier's usage in each run's summary. It is a lower bound: an attempt
that timed out records no cost (the CLI's report never came), the certificates of M5 computed
before its bug are cached without their cost, a run whose process ended without writing a record
is missing, and the session's own reviewing agents are part of the conversation's usage, not of
these files.

    python -m discovery.spend          # writes results/m6/spend.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
OUT = RESULTS / "m6" / "spend.json"


def _dicts(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from _dicts(v)
    elif isinstance(o, list):
        for v in o:
            yield from _dicts(v)


def _milestone(rel: str) -> str:
    head = rel.split("/")[0]
    return {"m6": "M6 (runs again in the closed sandbox)"}.get(head, head.upper())


def collect(results: Path = RESULTS) -> dict:
    attempts: dict[str, dict] = {}
    oneshot: dict[str, float] = {}
    proposers: dict[str, float] = {}
    search: dict[str, float] = {}
    for f in sorted(results.rglob("*.json")):
        rel = f.relative_to(results).as_posix()
        if rel == OUT.relative_to(RESULTS).as_posix():
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        for d in _dicts(data):
            if "sandbox" in d and "cost_usd" in d:
                attempts.setdefault(
                    d["sandbox"],
                    {
                        "cost": float(d["cost_usd"] or 0.0),
                        "file": rel,
                        "model": d.get("model"),
                        "timed_out": "timeout" in str(d.get("error", "")),
                    },
                )
        if not isinstance(data, dict):
            continue
        for key in ("cost_usd_proposer", "proposer_cost_usd"):
            if key in data:
                proposers[rel] = float(data[key] or 0.0)
        if rel.startswith("m3/") and "proposals" in data and "cost_usd" in data:
            if "/agent/" not in rel:
                oneshot[rel] = float(data["cost_usd"] or 0.0)
        if rel.startswith("m5/") and rel.endswith("/summary.json"):
            search[rel] = sum(
                float((data.get(k) or {}).get("cost_usd", 0.0))
                for k in ("usage", "certifier_usage")
            )
    by: dict[str, float] = {}
    for a in attempts.values():
        by[_milestone(a["file"])] = by.get(_milestone(a["file"]), 0.0) + a["cost"]
    for part in (oneshot, proposers, search):
        for rel, c in part.items():
            by[_milestone(rel)] = by.get(_milestone(rel), 0.0) + c
    total = sum(by.values())
    return {
        "total_usd": round(total, 2),
        "by_milestone_usd": {k: round(v, 2) for k, v in sorted(by.items())},
        "agentic_attempts": len(attempts),
        "agentic_usd": round(sum(a["cost"] for a in attempts.values()), 2),
        "timed_out_attempts_without_cost": sum(1 for a in attempts.values() if a["timed_out"]),
        "oneshot_usd": round(sum(oneshot.values()), 2),
        "proposers_usd": round(sum(proposers.values()), 2),
        "certified_search_usd": round(sum(search.values()), 2),
        "lower_bound": True,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    rec = collect()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
