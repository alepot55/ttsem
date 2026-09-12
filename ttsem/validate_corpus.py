"""Per-pass validation of every program of a directory, in parallel.

    python -m ttsem.validate_corpus --programs DIR --results OUT [--device cuda]
        [--triton-opt PATH] [--jobs N] [--limit N]

For each program: capture its launches, dump the IR before every pass (`MLIR_ENABLE_DUMP`), run
each whole-module dump with the level-1 semantics on the recorded inputs and compare with the
device. One JSON per program with the ordered stage verdicts; a summary at the end counts, per
pass, how many programs matched, mismatched, were unsupported or failed, so a pass that breaks
programs shows up as a column, not as an anecdote.
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_RUNNER = "ttsem.validate"


def _one(program: Path, results: Path, device: str, triton_opt: str | None, cc: int) -> dict:
    out = results / (program.stem + ".json")
    cmd = [
        sys.executable,
        "-m",
        _RUNNER,
        str(program),
        "--device",
        device,
        "--cc",
        str(cc),
        "--json",
        str(out),
    ]
    if triton_opt:
        cmd += ["--triton-opt", triton_opt]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return {"program": program.name, "error": "timeout"}
    if r.returncode != 0 or not out.exists():
        return {"program": program.name, "error": (r.stderr or r.stdout).strip()[-500:]}
    return json.loads(out.read_text())


def summarize(reports: list[dict]) -> dict:
    per_pass: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    first_bad = collections.Counter()
    errors = 0
    for rep in reports:
        if "launches" not in rep:
            errors += 1
            continue
        for launch in rep["launches"]:
            for stage in launch.get("passes", []):
                per_pass[stage["pass_name"]][stage["verdict"]] += 1
            if launch.get("first_bad_pass"):
                first_bad[launch["first_bad_pass"]] += 1
    return {
        "programs": len(reports),
        "errors": errors,
        "first_bad_pass": dict(first_bad.most_common()),
        "per_pass": {k: dict(v) for k, v in per_pass.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--programs", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--triton-opt", default=None)
    ap.add_argument("--cc", type=int, default=120)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.results.mkdir(parents=True, exist_ok=True)
    programs = sorted(a.programs.glob("*.py"))
    if a.limit:
        programs = programs[: a.limit]
    with ProcessPoolExecutor(a.jobs) as pool:
        reports = list(
            pool.map(
                _one,
                programs,
                [a.results] * len(programs),
                [a.device] * len(programs),
                [a.triton_opt] * len(programs),
                [a.cc] * len(programs),
            )
        )
    summary = summarize(reports)
    (a.results / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{summary['programs']} programs, {summary['errors']} errors")
    print("first bad pass:", summary["first_bad_pass"])
    worst = sorted(summary["per_pass"].items(), key=lambda kv: -kv[1].get("mismatch", 0))[:5]
    for name, counts in worst:
        print(f"  {counts} {name[:90]}")


if __name__ == "__main__":
    main()
