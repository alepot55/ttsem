"""The race detector over a directory of final-TTGIR dumps: Membar's barriers in, then out.

For every `*.ttgir.generic` under `corpus`, `triton-opt --allocate-shared-memory
-test-print-membar` gives the module with the barriers Membar inserts; the detector runs on
that module and on the same module with the barriers stripped. A kernel that races with the
barriers in place is a finding to look at (a Membar gap, or a detector gap); one that does
not race without them is a kernel whose warps never exchange shared bytes, where every
barrier Membar put there is conservative.

    python -m ttsem.races_corpus CORPUS_DIR --triton-opt .../triton-opt --json out.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from ttsem import races


def membar(text_path: Path, triton_opt: str) -> str | None:
    cmd = [
        triton_opt,
        str(text_path),
        "--allocate-shared-memory",
        "-test-print-membar",
        "--mlir-print-op-generic",
        "--mlir-print-local-scope",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    return out.stdout if out.returncode == 0 and out.stdout.strip() else None


def ablate(text: str) -> dict:
    """Strip each barrier line on its own and count the ones whose absence races."""
    lines = text.split("\n")
    where = [i for i, ln in enumerate(lines) if any(f'"{b}"' in ln for b in races.STRIPPED)]
    necessary = 0
    for i in where:
        candidate = "\n".join(lines[:i] + lines[i + 1 :])
        try:
            reports = races.detect(candidate)
        except Exception:
            continue
        if any(r.races for r in reports):
            necessary += 1
    return {"barriers": len(where), "necessary": necessary}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--triton-opt", required=True)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--ablate",
        action="store_true",
        help="remove Membar's barriers one at a time: a barrier is necessary when its removal "
        "alone produces a race",
    )
    args = ap.parse_args()
    files = sorted(args.corpus.rglob("*.ttgir.generic"))
    if args.limit:
        files = files[: args.limit]
    rows = []
    t_start = time.time()
    for i, f in enumerate(files):
        row = {"file": str(f.relative_to(args.corpus)), "status": "ok"}
        text = membar(f, args.triton_opt)
        if text is None:
            row["status"] = "triton-opt failed"
            rows.append(row)
            continue
        for label, module_text in (("with", text), ("without", races.strip_barriers(text))):
            try:
                reports = races.detect(module_text)
            except Exception as exc:
                row[label] = {"status": f"parse: {exc}"[:120]}
                continue
            row[label] = {
                "status": ";".join(sorted({r.status for r in reports}))[:160],
                "races": sum(len(r.races) for r in reports),
                "pairs": sum(len(r.pairs) for r in reports),
                "accesses": sum(r.accesses for r in reports),
                "barriers": sum(r.barriers for r in reports),
                "gaps": sorted({g for r in reports for g in r.gaps})[:6],
            }
        if args.ablate and row.get("with", {}).get("status") == "ok":
            row["ablation"] = ablate(text)
        w, wo = row.get("with", {}), row.get("without", {})
        flag = "RACE-WITH" if w.get("races") else ("silent" if not wo.get("races") else "ok")
        counts = f"with={w.get('races', '-'):<4} without={wo.get('races', '-'):<5}"
        if "ablation" in row:
            counts += f" needed={row['ablation']['necessary']}/{row['ablation']['barriers']}"
        more = f"acc={wo.get('accesses', '-'):<5} bar={w.get('barriers', '-'):<3}"
        status = w.get("status", row["status"])[:60]
        print(f"{flag:<9} {row['file']:<34} {counts} {more} {status}", flush=True)
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(
                f"# {i + 1}/{len(files)} in {time.time() - t_start:.0f}s",
                file=sys.stderr,
                flush=True,
            )
    ran = [r for r in rows if r.get("with", {}).get("status") == "ok"]
    n_with = sum(1 for r in ran if r["with"]["races"])
    n_without = sum(1 for r in ran if r["without"].get("races"))
    if args.ablate:
        total = sum(r["ablation"]["barriers"] for r in ran if "ablation" in r)
        needed = sum(r["ablation"]["necessary"] for r in ran if "ablation" in r)
        print(f"barriers: {needed} of {total} necessary on their own")
    print(
        f"{len(rows)} files, {len(ran)} ran, {n_with} race with barriers, {n_without} race without"
    )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
