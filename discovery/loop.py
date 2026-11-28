"""The search loop: ask a model for programs, check each one, keep the best of every cell.

``python -m discovery.loop --problem softmax --iterations 12 --model sonnet --mode archive
--workers 3 --seed 0 --out discovery/runs/NAME``, with ``--blind`` for the blind mode,
``--library PATH|none`` for the proved lemmas the prompts show (default: the proof tier's index),
``--gpu`` to time every candidate and ``--prove`` to prove the elites after the search.

* ``archive`` mode: each iteration draws a parent and two inspirations from the MAP-Elites
  archive, shows them with the last failures and the proved lemmas, and inserts the reply's
  verdict;
* ``iid`` mode, the baseline: every proposal gets the prompt archive mode starts with (the seed as
  parent, nothing from the run itself), so the samples are independent.

Up to ``workers`` proposals are in flight at once, in threads, since the model is the slow part;
each reply is evaluated in the main thread as it arrives, and the next iteration is drawn from
the archive as it then is. Every candidate is one JSON line of ``candidates.jsonl`` under the
output directory, next to the saved archive and a summary. The summary says which of the
problem's known forms the run reached, by cell and by algorithm (the same states and merges over
the reals, tested on random rows), at the first iteration asked, whatever order the replies came
back in. The known forms are never put in a prompt as programs, but a proved lemma states the
merges of the program it proves: when the library holds a lemma about a known form, every prompt
shows that form, and the summary says which ones it showed (``library=None`` shows no lemma).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import itertools
import json
import threading
import time
import types
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from discovery import ir, semantics
from discovery import llm as llms
from discovery.archive import Archive, Record
from discovery.evaluate import EXACT_TOL, Verdict, evaluate
from discovery.knowledge import LIBRARY_INDEX, Knowledge
from discovery.problems import PROBLEMS, Problem
from discovery.propose import ProposalError, build_prompt, complete

RUNS = Path(__file__).resolve().parent / "runs"
PROOF_ROUNDS = 3  # model rounds the prover gets for each elite, with --prove


def same_algorithm(
    problem: Problem, a: ir.Program, b: ir.Program, rng: np.random.Generator, trials: int = 8
) -> bool:
    """Whether two programs are one algorithm over the reals: as many passes, and for each pass a
    one-to-one map between the state fields under which the block summaries (of a random block,
    its two halves and their merge) agree, and the same outputs, in float64 on random rows. A
    test on samples, not a proof; a program that fails to run is no algorithm."""
    if [len(p.fields) for p in a.passes] != [len(p.fields) for p in b.passes]:
        return False
    A = semantics.Arith("float64")
    values: dict[tuple[int, int, str], list[np.ndarray]] = {}
    for _ in range(trials):
        n = int(rng.integers(2, 13))
        cut = int(rng.integers(1, n))
        row = problem.generate(rng, n, "normal")
        outs = []
        with np.errstate(all="ignore"):
            for which, prog in enumerate((a, b)):
                finished: dict[str, np.ndarray] = {}
                try:
                    for k, p in enumerate(prog.passes):
                        L = semantics.block_summary(
                            p, {f: v[:cut] for f, v in row.items()}, finished, A
                        )
                        R = semantics.block_summary(
                            p, {f: v[cut:] for f, v in row.items()}, finished, A
                        )
                        T = semantics.block_summary(p, row, finished, A)
                        M = semantics.merge(p, L, R, finished, A)
                        for f in p.fields:
                            seen = [np.ravel(s[f]) for s in (L, R, T, M)]
                            values.setdefault((which, k, f), []).extend(seen)
                        finished.update({f"{p.name}.{f}": v for f, v in T.items()})
                    outs.append(semantics.run(prog, row, A, semantics.Schedule("one")))
                except (semantics.EvalError, ValueError):
                    return False
        if not problem.error(outs[0], outs[1], row) <= EXACT_TOL:
            return False

    def agree(x: list[np.ndarray], y: list[np.ndarray]) -> bool:
        u, w = np.concatenate(x), np.concatenate(y)
        return u.shape == w.shape and np.allclose(u, w, rtol=1e-9, atol=1e-12, equal_nan=True)

    for k, (pa, pb) in enumerate(zip(a.passes, b.passes, strict=True)):
        if not any(
            all(
                agree(values[0, k, f], values[1, k, g])
                for f, g in zip(pa.fields, perm, strict=True)
            )
            for perm in itertools.permutations(pb.fields)
        ):
            return False
    return True


def reached(problem: Problem, archive: Archive, seed: int, lemmas: Sequence[dict] = ()) -> dict:
    """For each known form: its cell, the first iteration whose correct candidate landed in that
    cell, the first whose candidate was the same algorithm (wrong forms included), and the name
    of a lemma of ``lemmas`` (those the prompts showed) about the same algorithm, if any. The
    first iteration is the earliest one asked, duplicates included: with several workers the
    replies come back in any order, and the earlier of two identical ones may be the duplicate."""
    programs: dict[str, ir.Program] = {}
    for r in archive.records:
        if r.verdict.stage != "parse" and r.verdict.digest not in programs:
            try:
                programs[r.verdict.digest] = problem.parse(r.verdict.text)
            except ir.IRError:
                continue  # an archive written under other parsing rules
    tried = sorted(
        (r for r in archive.records if r.verdict.digest in programs), key=lambda r: r.iteration
    )
    proved = []
    for e in lemmas:
        try:
            proved.append((str(e.get("name", "lemma")), problem.parse(e["program"])))
        except (KeyError, TypeError, ir.IRError):
            continue  # a lemma with no program of this problem's shape
    out = {}
    for name, text in problem.known.items():
        known = evaluate(text, problem, seed)
        form = problem.parse(text)
        rng = np.random.default_rng(seed)
        same: dict[str, bool] = {}  # by digest: each program is tested once, duplicates reuse it
        first_same = None
        for r in tried:
            d = r.verdict.digest
            if d not in same:
                same[d] = same_algorithm(problem, programs[d], form, rng)
            if same[d]:
                first_same = r.iteration
                break
        out[name] = {
            "correct": known.correct,
            "cell": list(known.cell()) if known.correct else None,
            "cell_reached_at": next(
                (
                    r.iteration
                    for r in tried
                    if known.correct and r.verdict.correct and r.verdict.cell() == known.cell()
                ),
                None,
            ),
            "same_at": first_same,
            "lemma": next((n for n, p in proved if same_algorithm(problem, p, form, rng)), None),
        }
    return out


def _ask(model, prompt: str, lock: threading.Lock) -> tuple[str | None, str, dict]:
    """One proposal, in a worker thread: (program text or None, error, the call's usage)."""
    # a copy with a call list of its own, so this call's tokens are known while others run
    mine = copy.copy(model)
    mine.calls = []
    try:
        return complete(mine, prompt), "", llms.usage(mine)
    except (llms.LLMError, ProposalError) as exc:
        return None, str(exc), llms.usage(mine)
    finally:
        with lock:
            model.calls.extend(mine.calls)


def prove_elites(problem: Problem, archive: Archive, llm, out: Path) -> dict[str, dict]:
    """Run the prover on every elite but the seed (the specification itself), with the round
    files under ``out/proofs``: by digest, whether it was proved, in how many rounds, and why
    not. An elite whose merge is not a homomorphism on the tests' random splits is not sent: its
    ``merge_block`` theorem would be false, so no proof exists."""
    from discovery import prove as provers

    results: dict[str, dict] = {}
    for _, rec in sorted(archive.elites.items(), key=lambda kv: kv[0]):
        if rec.iteration == 0:
            continue
        if not rec.verdict.homomorphic:
            results[rec.verdict.digest] = {
                "proved": False,
                "rounds": 0,
                "message": "not sent to the prover: merge(block A, block B) differs from "
                "block(A ++ B) on a random split, so its merge_block theorem is false",
            }
            continue
        try:
            r = provers.prove(
                problem.parse(rec.verdict.text),
                problem,
                llm,
                rounds=PROOF_ROUNDS,
                workdir=out / "proofs",
            )
        except Exception as exc:  # noqa: BLE001  Lean missing, the prover failing: the search stands
            results[rec.verdict.digest] = {
                "proved": False,
                "message": f"the prover failed: {type(exc).__name__}: {exc}",
            }
            continue
        results[rec.verdict.digest] = {
            "proved": bool(r.proved),
            "rounds": r.rounds,
            "message": r.message,
            "library": r.library,
        }
    return results


def run(
    problem: Problem | str,
    llm,
    iterations: int = 12,
    mode: str = "archive",
    blind: bool = False,
    workers: int = 3,
    seed: int = 0,
    out: str | Path | None = None,
    memory: str | Path | None = None,
    pass_bias: float = 1.0,
    gpu: bool = False,
    library: str | Path | None = LIBRARY_INDEX,
    prove: bool = False,
    certify=None,
) -> dict:
    """Run the search and return its summary (also saved as ``summary.json`` under ``out``).
    ``memory`` is the knowledge file (default: the problem's, under ``runs/knowledge``);
    ``library`` the index of the proved lemmas the prompts show, read once (None: no lemma).

    With ``gpu``, the seed and every new correct candidate are compiled, checked on lengths
    around the kernel's block and timed on the GPU, one at a time, before they enter the archive:
    the measured time is their fitness (two times whose repeated timings overlap are a tie), an
    elite is chosen among the measured programs only, and a program whose
    kernel fails the check is ``gpu-failed``, remembered as rejected with the check's message, and
    never an elite nor a parent (the seed is the parent only while no cell has an elite).

    With ``prove``, after the search the prover (:func:`discovery.prove.prove`, the same model,
    :data:`PROOF_ROUNDS` rounds) is run on every elite but the seed, and each such cell of the
    summary is labelled ``E`` (proved equal to the specification over the reals) or ``T`` (tested
    only); the prover's calls are counted apart from the search's."""
    if isinstance(problem, str):
        problem = PROBLEMS[problem]
    if mode not in ("archive", "iid"):
        raise ValueError(f"mode is archive or iid, not {mode!r}")
    out = Path(out) if out is not None else RUNS / _default_name(problem.name, mode, blind, seed)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # With ``certify`` (a function of a correct verdict returning {"value": a certified error
    # bound, ...} or None), only certified programs can be elites and the bound is the fitness,
    # lower better (discovery.certsearch).
    archive = Archive(pass_bias, measured_only=gpu or certify is not None)
    certificates: dict[str, dict] = {}
    knowledge = Knowledge(problem.name, blind, memory, library)
    timings: dict[str, dict] = {}

    def measure(verdict: Verdict, guarded: bool = True) -> dict:
        # the GPU is the clock: in the main thread, one kernel at a time, never two
        from discovery import gpu as gpus

        try:
            result = gpus.fitness(problem.parse(verdict.text), problem)
        except Exception as exc:  # noqa: BLE001  a program with no kernel; the seed's raises
            if not guarded:
                raise
            result = {"ok": False, "message": f"no kernel: {type(exc).__name__}: {exc}"}
        timings[verdict.digest] = result
        return result

    def insert(verdict: Verdict, it: int = 0, parent: str = "") -> tuple[Record, dict | None]:
        if certify is not None:
            cert = None
            if verdict.correct and verdict.digest not in archive.by_digest:
                cert = certify(verdict)
                certificates[verdict.digest] = cert or {}
            value = (cert or {}).get("value")
            return archive.insert(verdict, it, parent, measured_us=value), cert
        # a new correct program is measured first, so its status says what the measurement did
        timing = None
        if gpu and verdict.correct and verdict.digest not in archive.by_digest:
            timing = measure(verdict, guarded=it > 0)
        if timing is None:
            return archive.insert(verdict, it, parent), None
        if timing["ok"]:
            spread = None
            if timing.get("ms_min") is not None and timing.get("ms_max") is not None:
                spread = (timing["ms_min"] * 1000, timing["ms_max"] * 1000)
            rec = archive.insert(
                verdict, it, parent, measured_us=timing["ms"] * 1000, measured_range=spread
            )
            return rec, timing
        failure = timing.get("message") or "the kernel failed the GPU check"
        return archive.insert(verdict, it, parent, gpu_failure=failure), timing

    start, start_timing = insert(evaluate(problem.seed, problem, seed))
    first_prompt = build_prompt(problem, start, [], knowledge, blind)
    lock = threading.Lock()
    no_program = 0
    t0 = time.time()

    def draw() -> tuple[Record, list[Record], str]:
        if mode == "iid":
            return start, [], first_prompt
        parent = archive.select_parent(rng) or start
        inspirations = archive.inspirations(parent, rng)
        return parent, inspirations, build_prompt(problem, parent, inspirations, knowledge, blind)

    try:
        with (
            open(out / "candidates.jsonl", "w", encoding="utf-8") as log,
            cf.ThreadPoolExecutor(max_workers=workers) as pool,
        ):
            first: dict = {"iteration": 0, "source": "seed", **_line(start)}
            if start_timing is not None:
                first["gpu"] = start_timing
            log.write(json.dumps(first) + "\n")
            pending: dict[cf.Future, tuple[int, Record, list[Record]]] = {}
            submitted = 0
            while pending or submitted < iterations:
                while submitted < iterations and len(pending) < workers:
                    submitted += 1
                    parent, inspirations, prompt = draw()
                    future = pool.submit(_ask, llm, prompt, lock)
                    pending[future] = (submitted, parent, inspirations)
                done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                for future in done:
                    it, parent, inspirations = pending.pop(future)
                    text, error, usage = future.result()
                    line: dict = {
                        "iteration": it,
                        "parent": parent.verdict.digest,
                        "inspirations": [r.verdict.digest for r in inspirations],
                        "usage": usage,
                    }
                    if text is None:
                        no_program += 1
                        line["error"] = error
                    else:
                        verdict = evaluate(text, problem, seed)
                        rec, timing = insert(verdict, it, parent.verdict.digest)
                        knowledge.record(verdict, rec.gpu_failure)
                        line.update(text=text, **_line(rec))
                        if timing is not None:
                            line["gpu"] = timing
                    log.write(json.dumps(line) + "\n")
                    log.flush()
    finally:
        knowledge.save()
        archive.save(out / "archive.json")
    shown = knowledge.shown_lemmas()
    summary = {
        "problem": problem.name,
        "mode": mode,
        "blind": blind,
        "model": getattr(llm, "name", type(llm).__name__),
        "iterations": iterations,
        "workers": workers,
        "seed": seed,
        "wall_seconds": round(time.time() - t0, 1),
        "usage": llms.usage(llm),
        "no_program": no_program,
        "library": None if knowledge.library is None else str(knowledge.library),
        "lemmas": [str(e.get("name", "lemma")) for e in shown],
        "statuses": dict(Counter(r.status for r in archive.records[1:])),
        "cells": [
            {
                "cell": list(cell),
                "fitness_us": round(r.fitness, 1),
                "measured": r.measured_us is not None,
                "iteration": r.iteration,
                "name": r.verdict.name,
                "digest": r.verdict.digest,
                "text": r.verdict.text,
            }
            for cell, r in sorted(archive.elites.items(), key=lambda kv: (kv[0][0], kv[1].fitness))
        ],
        "known": reached(problem, archive, seed, shown),
        "gpu": timings,
        "certificates": certificates,
        "gpu_failed": [
            {
                "iteration": r.iteration,
                "name": r.verdict.name,
                "digest": r.verdict.digest,
                "message": r.gpu_failure,
            }
            for r in archive.records
            if r.status == "gpu-failed"
        ],
        "out": str(out),
    }
    _save(summary, out)
    if prove:
        calls = len(getattr(llm, "calls", []))
        proofs = prove_elites(problem, archive, llm, out)
        for c in summary["cells"]:
            if c["digest"] in proofs:
                c["label"] = "E" if proofs[c["digest"]]["proved"] else "T"
                c["proof"] = proofs[c["digest"]]
        spent = types.SimpleNamespace(calls=getattr(llm, "calls", [])[calls:])
        summary["proofs"] = {
            "tried": len(proofs),
            "proved": sum(p["proved"] for p in proofs.values()),
            "rounds": PROOF_ROUNDS,
            "usage": llms.usage(spent),
        }
        _save(summary, out)
    return summary


def _save(summary: dict, out: Path) -> None:
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


def _line(rec: Record) -> dict:
    return {"status": rec.status, "verdict": rec.verdict.to_json()}


def _default_name(problem: str, mode: str, blind: bool, seed: int) -> str:
    return f"{problem}-{mode}{'-blind' if blind else ''}-s{seed}"


def report(summary: dict) -> str:
    """The summary as text."""
    u = summary["usage"]
    lines = [
        f"{summary['problem']}, {summary['mode']} mode{', blind' if summary['blind'] else ''}, "
        f"{summary['iterations']} iterations, {summary['workers']} workers, {summary['model']}",
        f"model: {u['calls']} calls ({u['errors']} failed), {u['seconds']:.0f} s of calls, "
        f"{u['input_tokens']} tokens in, {u['output_tokens']} out, ${u['cost_usd']:.2f}; "
        f"wall {summary['wall_seconds']:.0f} s",
        "candidates: "
        + ", ".join(f"{n} {s}" for s, n in sorted(summary["statuses"].items()))
        + f", {summary['no_program']} replies without a program",
    ]
    lemmas, library = summary.get("lemmas", []), summary.get("library")
    if lemmas:
        lines.append(f"prompts: {len(lemmas)} proved lemma(s) of {library}: {', '.join(lemmas)}")
    elif library:
        lines.append(f"prompts: no proved lemma (none for this problem in {library})")
    else:
        lines.append("prompts: no proved lemma (no library)")
    lines.append("best per cell (passes, state fields, float32 class):")
    for c in summary["cells"]:
        cell = f"({c['cell'][0]}, {c['cell'][1]}, {c['cell'][2]})"
        how = "measured" if c.get("measured") else "estimated"
        found = "seed" if c["iteration"] == 0 else f"iteration {c['iteration']}"
        label = ""
        if c.get("label") == "E":
            label = "  E"
        elif c.get("label"):
            label = f"  {c['label']} ({c['proof']['message']})"
        lines.append(
            f"  {cell:<22} {c['fitness_us']:>10.0f} us {how:<9}  {found:<13} {c['name']}{label}"
        )
    for g in summary.get("gpu_failed", []):
        found = "the seed" if g["iteration"] == 0 else f"iteration {g['iteration']}"
        lines.append(f"  not an elite: {found} ({g['name']}), {g['message']}")
    if "proofs" in summary:
        p, pu = summary["proofs"], summary["proofs"]["usage"]
        lines.append(
            f"proofs: {p['proved']} of {p['tried']} elites but the seed proved (E), the others "
            f"tested only (T); {pu['calls']} prover calls, ${pu['cost_usd']:.2f}"
        )
    given = any(k.get("lemma") for k in summary["known"].values())
    lines.append(
        "known forms (given to the model only as the lemmas named):"
        if given
        else "known forms (not given to the model):"
    )
    for name, k in summary["known"].items():
        where = f"cell ({', '.join(map(str, k['cell']))})" if k["cell"] else "wrong over the reals"
        cell = "-" if not k["correct"] else _at(k["cell_reached_at"])
        shown = f"; in every prompt as lemma {k['lemma']}" if k.get("lemma") else ""
        lines.append(
            f"  {name:<15} {where}: cell {cell}; same algorithm {_at(k['same_at'])}{shown}"
        )
    lines.append(f"output: {summary['out']}")
    return "\n".join(lines)


def _at(it: int | None) -> str:
    return "not reached" if it is None else ("in the seed" if it == 0 else f"at iteration {it}")


def model_from(spec: str):
    """The model a ``--model`` value names: a ``claude -p`` model (sonnet, opus, ...), or a spec
    with a colon for :func:`discovery.llm.from_spec` (claude:MODEL, openai:MODEL@URL)."""
    return llms.from_spec(spec) if ":" in spec else llms.ClaudeCLI(spec)


def library_option(value: str) -> Path | None:
    """A ``--library`` value: ``none`` (no lemma in any prompt) or the path of an index."""
    if value.lower() == "none":
        return None
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"no library index at {value} (or say none)")
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--problem", choices=sorted(PROBLEMS), default="softmax")
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument(
        "--model",
        default="sonnet",
        help="a claude -p model (sonnet, opus, ...) or a spec: claude:MODEL, openai:MODEL@URL",
    )
    ap.add_argument("--mode", choices=("archive", "iid"), default="archive")
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="output directory (default: discovery/runs/PROBLEM-MODE-sSEED)")
    ap.add_argument("--memory", help="knowledge file (default: discovery/runs/knowledge/...)")
    ap.add_argument(
        "--library",
        type=library_option,
        default=LIBRARY_INDEX,
        metavar="PATH|none",
        help="index of the proved lemmas every prompt shows, or none for no lemma "
        "(default: discovery/lean/Discovery/Library/index.json)",
    )
    ap.add_argument("--pass-bias", type=float, default=1.0)
    ap.add_argument("--gpu", action="store_true", help="time candidates on the GPU: the fitness")
    ap.add_argument(
        "--prove",
        action="store_true",
        help=f"after the search, prove every elite but the seed in Lean ({PROOF_ROUNDS} rounds "
        "each, same model; label E when proved, T otherwise)",
    )
    args = ap.parse_args(argv)
    summary = run(
        args.problem,
        model_from(args.model),
        args.iterations,
        args.mode,
        args.blind,
        args.workers,
        args.seed,
        args.out,
        args.memory,
        args.pass_bias,
        args.gpu,
        library=args.library,
        prove=args.prove,
    )
    print(report(summary))


if __name__ == "__main__":
    main()
