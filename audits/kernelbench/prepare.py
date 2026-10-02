"""Public KernelBench-style answers as task files, answer files and a manifest for the judge.

    python audits/kernelbench/prepare.py v3 RUNS_DIR PROBLEMS_DIR OUT_DIR
    python audits/kernelbench/prepare.py makora TEST.parquet OUT_DIR [--per-problem 30,10]
                                        [--levels 1,2] [--seed 0] [--skip MANIFEST.jsonl]

The manifest, OUT_DIR/manifest.jsonl, is what `python -m ttsem judge --manifest` reads: one line
per answer with `id`, `task`, `answer` (absolute paths) and the dataset's own label (`correct` or
`failed`), plus what the judge needs to apply the right rule (`rule`, `precision`, `gpu`) and what
the tables group by (`source`, `model`, `problem`, ...).

v3: `Infatoshi/kernelbench-v3-runs` (CC-BY-4.0) holds the winning solution of each (model, GPU,
problem) of an agent sweep of frontier models (Feb 2026), shipped only when it passed the
KernelBench-v3 harness (github.com/Infatoshi/KernelBench-v3, `src/eval/benchmark.py`: five seeds,
`max|diff| < atol + rtol * max|ref|`, 1e-3 for fp32; the dataset card's "allclose 1e-2" is the
agents' own self-check). Only the solutions with a `@triton.jit` kernel are taken. The task is the
problem file of `Infatoshi/kernelbench-v3-problems` (MIT). Each line carries `rule: v3` (that
harness's own rule), `precision: keep` (the tasks set their own dtypes) and the run's GPU.

makora: `makora-ai/triton-gpu-latency` (Apache-2.0), test split. A row is one program: a
KernelBench problem (`Model`, `get_inputs`, `get_init_inputs`) followed by a candidate `ModelNew`
written with Triton, and `y`, its measured latency, null (NaN in the parquet) when the candidate
"did not compile, did not match the reference output within tolerance, errored at runtime, or
otherwise could not be measured"; so `y` present is the dataset's verdict "correct". The task is
the problem part, cut where the candidate's import block starts; the answer is the whole program
as the dataset stores it (it defines `ModelNew`). Sampled per problem: so many rows labelled
correct and so many labelled failed (`--per-problem`), with `--seed`; `--skip` leaves out the rows
an earlier sample already has. The lines are judged by KernelBench's own rule (the judge's
default). Reading the parquet needs `pyarrow`; nothing else beyond the standard library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path


def split_program(x: str) -> tuple[str, str] | None:
    """(problem, candidate) of a makora program, or None when there is no Triton import."""
    lines = x.splitlines()
    first = next(
        (i for i, line in enumerate(lines) if line.startswith(("import triton", "from triton"))),
        None,
    )
    if first is None:
        return None
    start = first
    while start > 0 and (
        lines[start - 1].startswith(("import ", "from ")) or not lines[start - 1].strip()
    ):
        start -= 1
    problem = "\n".join(lines[:start]) + "\n"
    if "def get_inputs" not in problem or "class Model" not in problem:
        return None
    return problem, "\n".join(lines[start:]) + "\n"


def makora(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    table = pq.read_table(args.parquet).to_pydict()
    levels = {int(v) for v in args.levels.split(",")}
    n_ok, n_bad = (int(v) for v in args.per_problem.split(","))
    by_problem: dict[str, tuple[list[int], list[int]]] = {}
    skipped = 0
    taken: set[int] = set()  # rows another sample already has
    for manifest in args.skip:
        taken |= {json.loads(ln)["row"] for ln in manifest.read_text().splitlines() if ln.strip()}
    columns = zip(table["x"], table["y"], table["problem_id"], strict=True)
    for i, (x, y, problem) in enumerate(columns):
        if int(problem.split("_")[0]) not in levels or i in taken:
            continue
        if split_program(x) is None or "class ModelNew" not in x:
            skipped += 1
            continue
        measured = y is not None and not math.isnan(y)
        by_problem.setdefault(problem, ([], []))[0 if measured else 1].append(i)
    rng = random.Random(args.seed)
    (args.out / "tasks").mkdir(parents=True, exist_ok=True)
    (args.out / "answers").mkdir(parents=True, exist_ok=True)
    items = []
    for problem in sorted(by_problem, key=lambda p: tuple(int(v) for v in p.split("_"))):
        good, bad = by_problem[problem]
        picked = [(i, "correct") for i in rng.sample(good, min(n_ok, len(good)))]
        picked += [(i, "failed") for i in rng.sample(bad, min(n_bad, len(bad)))]
        for i, label in picked:
            x = table["x"][i]
            parts = split_program(x)
            assert parts is not None
            digest = hashlib.sha1(parts[0].encode()).hexdigest()[:8]
            task = args.out / "tasks" / f"mk_{problem}_{digest}.py"
            task.write_text(parts[0], encoding="utf-8")
            answer = args.out / "answers" / f"mk_{problem}_{i}.py"
            answer.write_text(x, encoding="utf-8")
            y = table["y"][i]
            items.append(
                {
                    "id": answer.stem,
                    "task": str(task.resolve()),
                    "answer": str(answer.resolve()),
                    "source": "makora",
                    "row": i,
                    "problem": problem,
                    "holdout": table["holdout_group"][i],
                    "label": label,
                    "y": None if y is None or math.isnan(y) else y,
                }
            )
    rng.shuffle(items)  # so a run cut short is still a sample of every problem
    with (args.out / "manifest.jsonl").open("w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    print(f"{len(items)} answers from {len(by_problem)} problems ({skipped} rows not split)")


def safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")


def v3(args: argparse.Namespace) -> None:
    rows = list(csv.DictReader((args.runs / "results.csv").open()))
    elsewhere = {p.name: p for p in sorted(args.problems.rglob("*.py"))}  # tile_specialized/ ...
    (args.out / "answers").mkdir(parents=True, exist_ok=True)
    items, missing = [], 0
    for r in rows:
        if not r["solution_link"]:
            continue
        solution = args.runs / "solutions" / Path(r["solution_link"]).name
        task = args.problems / f"level{r['level']}" / r["problem"]
        if not task.exists():
            task = elsewhere.get(r["problem"], task)
        if not solution.exists() or not task.exists():
            missing += 1
            continue
        code = solution.read_text(encoding="utf-8")
        if "@triton.jit" not in code and "@triton.autotune" not in code:
            continue
        ident = safe(f"v3_{r['model']}_{r['gpu']}_L{r['level']}_{Path(r['problem']).stem}")
        answer = args.out / "answers" / f"{ident}.py"
        answer.write_text(code, encoding="utf-8")
        items.append(
            {
                "id": ident,
                "task": str(task.resolve()),
                "answer": str(answer.resolve()),
                "source": "v3",
                "model": r["model"],
                "gpu": r["gpu"],
                "level": int(r["level"]),
                "suite": task.parent.name,
                "problem": r["problem"],
                "label": "correct" if r["correct"] == "True" else "failed",
                "speedup": float(r["speedup"]) if r["speedup"] else None,
                "precision_used": r["precision_used"],
                "precision": "keep",  # the v3 tasks set their own dtypes; nothing is cast
                "rule": "v3",  # judged by the v3 harness's own rule
                "cuda_inline": "load_inline" in code,
            }
        )
    random.Random(0).shuffle(items)
    with (args.out / "manifest.jsonl").open("w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    print(f"{len(items)} Triton answers ({missing} rows without a solution or task file)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    three = sub.add_parser("v3", help="the KernelBench-v3 release")
    three.add_argument("runs", type=Path, help="Infatoshi/kernelbench-v3-runs")
    three.add_argument("problems", type=Path, help="Infatoshi/kernelbench-v3-problems")
    three.add_argument("out", type=Path)
    mk = sub.add_parser("makora", help="makora-ai/triton-gpu-latency, test split")
    mk.add_argument("parquet", type=Path, help="its data/test.parquet")
    mk.add_argument("out", type=Path)
    mk.add_argument("--per-problem", default="30,10", help="rows labelled correct,failed (30,10)")
    mk.add_argument("--levels", default="1,2", help="KernelBench levels (1,2)")
    mk.add_argument("--seed", type=int, default=0)
    mk.add_argument("--skip", type=Path, action="append", default=[], help="a manifest to skip")
    args = ap.parse_args(argv)
    makora(args) if args.cmd == "makora" else v3(args)


if __name__ == "__main__":
    main()
