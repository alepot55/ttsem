"""The M3 summary: proved bounds, their tightness and F1 check, and the order they certify (F4).

Reads the agentic prover's records (``results/m3/agent/``), the F1 checks (``results/m3/f1/``) and
the measured errors of M1 (``results/m1/library.json``), and writes ``results/m3/summary.json``:

* per program: B, who proved it, rounds or turns, cost, tightness (B / the worst error the long
  attack found) and whether that attack broke it (F1);
* F4: for each pair the brief names, the certified bounds at reference points against the errors
  measured there; the certified order must match the measured one on every pair whose measured
  errors differ by more than 10 times.

    python -m discovery.fpreport
"""

from __future__ import annotations

import json
from pathlib import Path

from discovery import bound, fpbound

RESULTS = Path(__file__).parent / "results"
U = fpbound.U32
BASE = {**fpbound.ENV32, "D": 8.0, "d": 15.0, "n": 4096.0}

# reference points: the families of M1 (n = 4096, blocks of 256, left fold)
POINTS = {
    "logits around 100": {**BASE, "R": 8.0, "M": 104.0},
    "logits around 1e4": {**BASE, "R": 8.0, "M": 10004.0},
    "variance: mean 1e4, spread 0.1": {**BASE, "R": 0.8, "M": 10000.4, "S2": 0.01, "Q": 1e8 + 0.01},
}

# (program A, program B, point, family of results/m1 that measures it, metric scale)
PAIRS = [
    ("SoftmaxOnline", "SoftmaxLse", "logits around 100", "normal100"),
    ("SoftmaxOnline", "SoftmaxLse", "logits around 1e4", "normal1e4"),
    ("VarianceWelford", "VarianceSumsq", "variance: mean 1e4, spread 0.1", "offset"),
]
PROBLEM = {
    "SoftmaxOnline": "softmax",
    "SoftmaxLse": "softmax",
    "AttentionOnline": "attention",
    "VarianceWelford": "variance",
    "VarianceSumsq": "variance",
}


def _in_u(module: str, value: float, point: dict) -> float:
    """A certified bound in the unit of M1's measurements: u for the softmax (the log of the
    normalizer is its relative error to first order); for the variance, relative to the variance."""
    if PROBLEM[module] == "variance":
        return value / point["S2"] / U
    return value / U


def main() -> int:
    agent = RESULTS / "m3" / "agent"
    f1 = RESULTS / "m3" / "f1"
    measured = {
        (r["program"], r["format"]): r
        for r in json.loads((RESULTS / "m1" / "library.json").read_text(encoding="utf-8"))
    }
    programs = {}
    for module in PROBLEM:
        rec_path = agent / f"{module}-agent.json"
        rejudged = agent / f"{module}-rejudged.json"
        if not rec_path.exists() and not rejudged.exists():
            programs[module] = {"proved": False, "status": "no record"}
            continue
        rec = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.exists() else {}
        item = {
            "B": rec.get("B"),
            "proved": bool(rec.get("proved")),
            "by": (rec.get("proved") or {}).get("model"),
            "attempts": [
                {k: a.get(k) for k in ("model", "stage", "ok", "turns", "seconds", "cost_usd")}
                for a in rec.get("attempts", [])
            ],
            "cost_usd": rec.get("cost_usd"),
        }
        if rejudged.exists():  # a proof refused by the old deny list, judged again after its fix
            again = json.loads(rejudged.read_text(encoding="utf-8"))
            if again.get("ok"):
                item.update(B=again["B"], proved=True, by=again["model"], rejudged=again["note"])
        # a later statement version (the attention's v4) has its own record
        for later in sorted(agent.glob(f"{module}-v*.json")):
            again = json.loads(later.read_text(encoding="utf-8"))
            if again.get("proved"):
                item.update(B=again["proved"]["B"], proved=True, by=again["proved"].get("model"))
                item["version"] = later.stem.split("-v")[-1]
        check = f1 / f"{module}.json"
        if check.exists():
            c = json.loads(check.read_text(encoding="utf-8"))
            item.update(
                F1_violated=c["F1_violated"], tightness=c["tightness"], worst_ratio=c["worst_ratio"]
            )
        programs[module] = item
    pairs = []
    for a, b, point, family in PAIRS:
        env = POINTS[point]
        row = {"pair": [a, b], "point": point, "family": family}
        for m in (a, b):
            p = programs.get(m, {})
            meas = measured.get((m, "float32"), {})
            row[m] = {
                "proved": p.get("proved", False),
                "certified_u": _in_u(m, bound.parse(p["B"], PROBLEM[m]).value(env), env)
                if p.get("B")
                else None,
                "measured_max_u": meas.get(f"{family}_max"),
                "adversary_max_u": meas.get("adversary_max"),
            }
        ma, mb = row[a]["measured_max_u"], row[b]["measured_max_u"]
        ca, cb = row[a]["certified_u"], row[b]["certified_u"]
        row["measured_ratio"] = (mb / ma) if ma and mb else None
        row["certified_ratio"] = (cb / ca) if ca and cb else None
        unproved = [m for m in (a, b) if not row[m]["proved"]]
        if unproved:
            # a proposed bound that is not proved certifies nothing: the pair is not counted
            row["F4"] = f"unresolved: no proved bound for {', '.join(unproved)}"
        elif row["measured_ratio"] and row["certified_ratio"]:
            far = row["measured_ratio"] > 10 or row["measured_ratio"] < 0.1
            same = (row["measured_ratio"] > 1) == (row["certified_ratio"] > 1)
            row["F4"] = "agrees" if same else ("DISAGREES" if far else "differs, within 10x")
        pairs.append(row)
    out = {"programs": programs, "F4": pairs, "points": POINTS}
    path = RESULTS / "m3" / "summary.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
