"""The rediscovery control: archive search, blind archive search and independent sampling (IID),
on every problem, a few seeds each, with the same model and the same number of calls.

Every run starts from the seed program with an empty memory of its own, so no run reads
another's failures, and by default with no proved lemma in its prompts (``--library none``): a
lemma states the merges of the program it proves, so a lemma about a known form would put the
answer in the prompt. ``--library PATH`` shows the lemmas of that index instead; it is copied
once, before the first run, to ``library.json`` under ``--out``, and every run reads the copy, so
all of them see the same lemmas whatever the prover adds meanwhile. The output is
``results.json`` and a Markdown table, ``results.md``, under ``--out``; each run's log is in its
own directory there.

    python -m discovery.experiments --problems softmax attention variance --seeds 0 1 \
        --iterations 6 --workers 2 --model sonnet --out discovery/runs/rediscovery
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import shutil
from pathlib import Path

from discovery import loop
from discovery.evaluate import evaluate
from discovery.problems import PROBLEMS

MODES = {"archive": ("archive", False), "blind": ("archive", True), "iid": ("iid", False)}


def one(
    problem: str,
    mode: str,
    seed: int,
    iterations: int,
    workers: int,
    model: str,
    out: Path,
    library: Path | None = None,
) -> dict:
    """One run of the control: ``model`` is a ``--model`` value, ``library`` the index of the
    lemmas its prompts show (None: no lemma)."""
    search, blind = MODES[mode]
    where = out / f"{problem}-{mode}-s{seed}"
    memory = where / "memory.json"
    memory.unlink(missing_ok=True)  # an empty memory of its own, even in a directory used before
    summary = loop.run(
        problem,
        loop.model_from(model),
        iterations,
        search,
        blind,
        workers,
        seed,
        out=where,
        memory=memory,
        library=library,
    )
    known = summary["known"]
    cells = [tuple(c["cell"]) for c in summary["cells"]]
    seed_cell = evaluate(PROBLEMS[problem].seed, PROBLEMS[problem], seed).cell()
    return {
        "problem": problem,
        "mode": mode,
        "seed": seed,
        "reached": {
            name: {
                "cell_at": k.get("cell_reached_at"),
                "same_at": k.get("same_at"),
                "lemma": k.get("lemma"),
            }
            for name, k in known.items()
        },
        "library": summary["library"],
        "lemmas": summary["lemmas"],
        "cells": [list(c) for c in cells],
        "new_cells": [list(c) for c in cells if c != seed_cell],
        "statuses": summary["statuses"],
        "no_program": summary["no_program"],
        "usage": summary["usage"],
        "wall_seconds": summary["wall_seconds"],
    }


def table(rows: list[dict]) -> str:
    out = [
        "| problem | mode | seed | known forms reached (iteration of the first same algorithm) "
        "| cells found | rejected | calls | tokens in/out | cost |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        reached = ", ".join(
            (
                f"{n} @{v['same_at']}"
                if v["same_at"] is not None
                else f"{n} cell @{v['cell_at']}"
                if v["cell_at"] is not None
                else f"{n} -"
            )
            + (" (in the prompt as a lemma)" if v.get("lemma") else "")
            for n, v in r["reached"].items()
        )
        cells = "; ".join(f"({c[0]},{c[1]},{c[2]})" for c in r["cells"])
        u = r["usage"]
        out.append(
            f"| {r['problem']} | {r['mode']} | {r['seed']} | {reached} | {cells} "
            f"| {r['statuses'].get('rejected', 0) + r['no_program']} | {u['calls']} "
            f"| {u['input_tokens']:,}/{u['output_tokens']:,} | ${u['cost_usd']:.2f} |"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--problems", nargs="+", default=sorted(PROBLEMS), choices=sorted(PROBLEMS))
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument(
        "--model",
        default="sonnet",
        help="a claude -p model (sonnet, opus, ...) or a spec: claude:MODEL, openai:MODEL@URL",
    )
    ap.add_argument(
        "--library",
        type=loop.library_option,
        default=None,
        metavar="PATH|none",
        help="index of the proved lemmas every prompt shows, copied once to OUT/library.json "
        "(default: none, so that no prompt holds a known form)",
    )
    ap.add_argument("--out", default="discovery/runs/rediscovery")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    library = None
    if args.library is not None:  # one copy for the whole experiment: the index may change
        library = out / "library.json"
        if args.library.resolve() != library.resolve():
            shutil.copyfile(args.library, library)
    jobs = [(p, m, s) for p in args.problems for m in args.modes for s in args.seeds]
    rows: list[dict] = []
    # one thread per problem: the runs of a problem go one after the other, the problems at once
    by_problem = {p: [j for j in jobs if j[0] == p] for p in args.problems}

    def series(problem: str) -> list[dict]:
        done = []
        for p, m, s in by_problem[problem]:
            r = one(p, m, s, args.iterations, args.workers, args.model, out, library)
            done.append(r)
            print(f"{p} {m} s{s}: {json.dumps(r['reached'])}", flush=True)
        return done

    with cf.ThreadPoolExecutor(max_workers=len(args.problems)) as pool:
        for part in pool.map(series, args.problems):
            rows.extend(part)
    rows.sort(key=lambda r: (r["problem"], list(MODES).index(r["mode"]), r["seed"]))
    shown = (
        f"the lemmas of {library} (a copy of {args.library})"
        if library is not None
        else "none (--library none)"
    )
    (out / "results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    (out / "results.md").write_text(
        f"Proved lemmas in the prompts: {shown}.\n\n{table(rows)}\n", encoding="utf-8"
    )
    print(table(rows))


if __name__ == "__main__":
    main()
