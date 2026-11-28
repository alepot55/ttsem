"""The search tier, with no network: the archive, the proposer's prompt and its reply parser, the
loop on scripted answers (with stand-ins for the GPU and the prover), the control experiment's
wiring, and the enumerator on the known merges. One test times real kernels; it skips without
CUDA."""

from __future__ import annotations

import json
import math
import sys
import types
from collections import Counter

import numpy as np
import pytest

import discovery
from discovery import experiments, ir, llm, loop
from discovery import prove as provers
from discovery.archive import Archive
from discovery.enumerate import synthesise
from discovery.evaluate import Verdict, evaluate
from discovery.knowledge import LIBRARY_INDEX, Knowledge
from discovery.problems import ATTENTION, PROBLEMS, SOFTMAX, VARIANCE
from discovery.propose import ProposalError, build_prompt, parse_reply

ONLINE_RENAMED = """(program mine
  (input x)
  (pass p
    (state s top)
    (block (top (rmax x)) (s (sum (exp (- x top)))))
    (merge (top (max R.top L.top))
           (s (+ (* R.s (exp (- R.top top))) (* L.s (exp (- L.top top)))))))
  (output elementwise (/ (exp (- x p.top)) p.s)))"""

# the online merge with the output written exp(x - m - log d): the same cell, one node more
LOGFOLD = """(program softmax-logfold
  (input x)
  (pass a
    (state m d)
    (block (m (rmax x)) (d (sum (exp (- x m)))))
    (merge (m (max L.m R.m))
           (d (+ (* L.d (exp (- L.m m))) (* R.d (exp (- R.m m)))))))
  (output elementwise (exp (- x (+ a.m (log a.d))))))"""

# right over the reals on every blocking, but its state keeps the first block's maximum, not the
# maximum: merge(block A, block B) is not block(A ++ B), so no merge_block theorem holds for it
FIRSTMAX = """(program softmax-firstmax
  (input x)
  (pass a
    (state m d)
    (block (m (rmax x)) (d (sum (exp (- x m)))))
    (merge (m L.m)
           (d (+ L.d (* R.d (exp (- R.m L.m)))))))
  (output elementwise (/ (exp (- x a.m)) a.d)))"""

TELLING = ("online", "welford", "flash", "rescal", "unshifted")


def fake(digest: str, cell=(2, 2, "stable"), time_us=100.0, stage="passed", nodes=10) -> Verdict:
    return Verdict(
        stage,
        "a message",
        passes=cell[0],
        state_fields=cell[1],
        fp_class=cell[2] if stage == "passed" else "",
        cost={"time_us": time_us} if stage == "passed" else {},
        digest=digest,
        nodes=nodes,
        text=f"(program {digest})",
    )


def fenced(text: str) -> str:
    return f"Here is the program.\n\n```lisp\n{text}\n```\n"


def known_forms_in(problem, prompt: str) -> list[str]:
    """What a prompt gives away of the problem's known forms: a whole program, a program's name, a
    merge of more than three nodes (a sum of two fields is also the seed's), a telling word."""
    found = []
    for form, text in problem.known.items():
        known = problem.parse(text)
        if known.text() in prompt:
            found.append(f"{form}: program")
        if known.name in prompt:
            found.append(f"{form}: name")
        for q in known.passes:
            for f, e in q.merge:
                if sum(1 for _ in ir.walk(e)) > 3 and ir.unparse_expr(e) in prompt:
                    found.append(f"{form}: merge of {q.name}.{f}")
    return found + [word for word in TELLING if word in prompt.lower()]


def lemma(problem, form: str) -> dict:
    """The library entry of a proved known form, as the prover writes it: the statement spells
    out the program's merges."""
    prog = problem.parse(problem.known[form])
    merges = "; ".join(
        f"in pass {p.name}, merging the summaries "
        + " ".join(f"({f} {ir.unparse_expr(e)})" for f, e in p.block)
        + " of two nonempty blocks with "
        + " ".join(f"({f} {ir.unparse_expr(e)})" for f, e in p.merge)
        + " gives the summary of the two blocks together"
        for p in prog.passes
    )
    return {
        "name": prog.name,
        "problem": problem.name,
        "statement": f"over the reals, {prog.name} computes what the seed computes: {merges}",
        "program": prog.text(),
        "digest": prog.digest(),
    }


def fake_gpu(monkeypatch, times: dict, ratios: dict | None = None) -> list[str]:
    """Stand in for :mod:`discovery.gpu`: by program name, ``fitness`` gives a time in ms (a
    number, or a tuple of the median and the range of the repeated timings), fails with a message
    (a string) or raises (an exception). With ``ratios`` ((new, elite) names -> a / b), ``paired``
    times two kernels side by side and records ``("paired", new, elite)``. Returns the names
    measured, in order."""
    measured: list = []

    def fitness(prog, problem, shape=None):
        measured.append(prog.name)
        t = times[prog.name]
        if isinstance(t, Exception):
            raise t
        if isinstance(t, str):
            return {"ok": False, "message": t}
        if isinstance(t, tuple):
            ms, lo, hi = t
            return {"ok": True, "ms": ms, "ms_min": lo, "ms_max": hi, "shape": [1024, 32768]}
        return {"ok": True, "ms": t, "shape": [1024, 32768], "worst_error": 1e-7}

    module = types.ModuleType("discovery.gpu")
    module.fitness = fitness
    if ratios is not None:

        def paired(a, b, problem, shape=None, rounds=5):
            measured.append(("paired", a.name, b.name))
            r = ratios[(a.name, b.name)]
            return {"ratio": r, "ratios": [r] * rounds, "a_ms": [], "b_ms": [], "shape": []}

        module.paired = paired
    monkeypatch.setitem(sys.modules, "discovery.gpu", module)
    monkeypatch.setattr(discovery, "gpu", module, raising=False)
    return measured


def fake_prover(monkeypatch, proved: set[str]) -> list[tuple]:
    """Stand in for :func:`discovery.prove.prove`: a program is proved when its name is in
    ``proved``. Returns the calls: (program name, problem name, model, keyword arguments)."""
    calls: list[tuple] = []

    def prove(prog, problem, model, **kwargs):
        calls.append((prog.name, problem.name, model, kwargs))
        ok = prog.name in proved
        rounds = 1 if ok else kwargs["rounds"]
        return types.SimpleNamespace(
            proved=ok,
            rounds=rounds,
            message=f"{'proved' if ok else 'not proved'} in {rounds} round(s)",
            library="Library/X.lean" if ok else "",
        )

    monkeypatch.setattr(provers, "prove", prove)
    return calls


def _cuda() -> bool:
    try:
        import torch
        import triton  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def lines_of(out) -> list[dict]:
    return [json.loads(s) for s in (out / "candidates.jsonl").read_text().splitlines()]


def search(model, iterations: int, where, **kwargs) -> dict:
    """:func:`loop.run` on softmax in archive mode with one worker and no library (unless given),
    its output in ``where/run`` and its memory in ``where/k.json``."""
    kwargs.setdefault("library", None)
    return loop.run(
        SOFTMAX, model, iterations, workers=1, out=where / "run", memory=where / "k.json", **kwargs
    )


# --------------------------------------------------------------------------- the archive


def test_the_elite_is_the_fastest_correct_program_of_its_cell():
    a = Archive()
    assert a.insert(fake("a", time_us=100)).status == "new-cell"
    assert a.insert(fake("b", time_us=50)).status == "improved"
    assert a.insert(fake("c", time_us=70)).status == "kept"
    assert a.insert(fake("b", time_us=1)).status == "duplicate"
    assert a.insert(fake("d", stage="exact")).status == "rejected"
    assert a.insert(fake("e", cell=(1, 3, "stable"), time_us=500)).status == "new-cell"
    assert a.insert(fake("f", time_us=50, nodes=8)).status == "improved"  # a tie: smaller wins
    assert len(a.records) == 7
    assert {c: r.verdict.digest for c, r in a.elites.items()} == {
        (2, 2, "stable"): "f",
        (1, 3, "stable"): "e",
    }
    a.attach_time("c", 20.0)  # a measured time replaces the estimate
    assert a.elites[(2, 2, "stable")].verdict.digest == "c"
    assert a.by_digest["c"].status == "improved"  # what the measurement did
    a.attach_time("c", 90.0)
    assert a.elites[(2, 2, "stable")].verdict.digest == "f"
    assert a.by_digest["c"].status == "kept"


def test_times_within_the_tolerance_tie_and_the_smaller_program_wins():
    a = Archive()  # 2%
    assert a.insert(fake("a", time_us=100, nodes=10)).status == "new-cell"
    assert a.insert(fake("b", time_us=98.5, nodes=12)).status == "kept"  # faster by noise only
    assert a.insert(fake("c", time_us=99.0, nodes=9)).status == "improved"  # slower, but smaller
    assert a.insert(fake("d", time_us=96.0, nodes=30)).status == "improved"  # 3% faster
    assert a.elites[(2, 2, "stable")].verdict.digest == "d"
    exact = Archive(tolerance=0.0)  # only exact ties go to the smaller program
    exact.insert(fake("a", time_us=100, nodes=10))
    assert exact.insert(fake("b", time_us=98.5, nodes=12)).status == "improved"
    assert exact.insert(fake("c", time_us=98.5, nodes=11)).status == "improved"
    # two medians of one kernel family on the GPU, 0.2% apart: the earlier program stays
    timed = Archive(measured_only=True)
    timed.insert(fake("a", time_us=3247.2), measured_us=3241.0)
    assert timed.insert(fake("b", time_us=3247.2), measured_us=3235.8).status == "kept"


def test_measured_times_whose_ranges_overlap_tie(tmp_path):
    a = Archive(measured_only=True)  # 2%
    a.insert(fake("a", nodes=10), measured_us=1500.0, measured_range=(1470.0, 1530.0))
    # 3% faster, but within the spread of the elite's timings: a tie, lost by the larger program
    slow = (1450.0, 1475.0)
    assert a.insert(fake("b", nodes=12), measured_us=1455.0, measured_range=slow).status == "kept"
    assert (
        a.insert(fake("c", nodes=9), measured_us=1455.0, measured_range=slow).status == "improved"
    )
    # 4% faster and apart: the faster program wins, whatever its size
    apart = (1395.0, 1405.0)
    assert (
        a.insert(fake("d", nodes=30), measured_us=1400.0, measured_range=apart).status == "improved"
    )
    # a time with no range is compared on the tolerance alone
    assert a.insert(fake("e", nodes=30), measured_us=1350.0).status == "improved"
    a.save(tmp_path / "archive.json")
    b = Archive.load(tmp_path / "archive.json")
    assert b.by_digest["d"].measured_range == apart and b.by_digest["e"].measured_range is None
    assert b.elites[(2, 2, "stable")].verdict.digest == "e"


def test_a_measured_archive_ranks_measured_programs_only(tmp_path):
    a = Archive(measured_only=True)
    seed = a.insert(fake("seed", cell=(3, 2, "stable"), time_us=4000), measured_us=4100.0)
    assert seed.status == "new-cell"
    # an estimate never competes with measured times, and a kernel that failed never competes
    assert a.insert(fake("guess", time_us=1.0)).status == "unmeasured"
    failed = a.insert(fake("broken", time_us=1.0), gpu_failure="the kernel is off")
    assert failed.status == "gpu-failed" and failed.fitness == 1.0
    assert (2, 2, "stable") not in a.elites
    assert a.insert(fake("timed", time_us=5000), measured_us=3300.0).status == "new-cell"
    assert a.insert(fake("broken", time_us=1.0)).status == "duplicate"
    a.attach_time("guess", 3000.0)  # with a time of its own, it competes
    assert a.elites[(2, 2, "stable")].verdict.digest == "guess"
    assert a.by_digest["guess"].status == "improved"
    rng = np.random.default_rng(0)
    for _ in range(50):
        parent = a.select_parent(rng)
        assert parent.verdict.digest != "broken"
        assert "broken" not in {r.verdict.digest for r in a.inspirations(parent, rng)}
    a.save(tmp_path / "archive.json")
    b = Archive.load(tmp_path / "archive.json")
    assert b.measured_only and b.tolerance == a.tolerance
    assert [r.status for r in b.records] == [r.status for r in a.records]
    assert {c: r.verdict.digest for c, r in b.elites.items()} == {
        (3, 2, "stable"): "seed",
        (2, 2, "stable"): "guess",  # not the failed kernel, whose estimate is lower
    }
    assert b.by_digest["broken"].gpu_failure == "the kernel is off"


def test_parents_lean_to_fewer_passes_and_inspirations_come_from_other_cells():
    a = Archive(pass_bias=1.0)
    for d, cell in [("one", (1, 3, "stable")), ("three", (3, 2, "stable"))]:
        a.insert(fake(d, cell=cell))
    rng = np.random.default_rng(0)
    picks = Counter(a.select_parent(rng).verdict.passes for _ in range(4000))
    assert 0.84 < picks[1] / 4000 < 0.92  # weights 1 and e^-2
    a.pass_bias = 0.0
    picks = Counter(a.select_parent(rng).verdict.passes for _ in range(4000))
    assert 0.45 < picks[1] / 4000 < 0.55
    a.insert(fake("two", cell=(2, 2, "stable")))
    a.insert(fake("unstable", cell=(2, 1, "nonfinite")))
    parent = a.elites[(2, 2, "stable")]
    for _ in range(20):
        insp = a.inspirations(parent, rng)
        assert len(insp) == 2 and len({r.verdict.digest for r in insp}) == 2
        assert parent not in insp
    assert Archive().select_parent(rng) is None


def test_archive_saves_and_loads(tmp_path):
    a = Archive(pass_bias=0.5)
    a.insert(evaluate(SOFTMAX.seed, SOFTMAX))
    a.insert(evaluate(SOFTMAX.known["online"], SOFTMAX), 1, "x")
    a.insert(evaluate(SOFTMAX.known["no-rescale"], SOFTMAX), 2, "y")
    a.insert(evaluate("(program broken", SOFTMAX), 3, "y")
    a.insert(evaluate(SOFTMAX.known["online"], SOFTMAX), 4, "y")
    a.attach_time(a.records[0].verdict.digest, 123.0)
    a.save(tmp_path / "archive.json")
    b = Archive.load(tmp_path / "archive.json")
    assert b.pass_bias == 0.5 and b.tolerance == a.tolerance and not b.measured_only
    assert [r.status for r in b.records] == [r.status for r in a.records]
    assert [r.status for r in b.records] == [
        "new-cell",
        "new-cell",
        "rejected",
        "rejected",
        "duplicate",
    ]
    assert {c: r.verdict.digest for c, r in b.elites.items()} == {
        c: r.verdict.digest for c, r in a.elites.items()
    }
    assert b.records[0].fitness == 123.0
    assert math.isinf(b.records[3].verdict.exact_err)
    assert b.records[1].verdict.fp == a.records[1].verdict.fp
    assert b.records[1].verdict.cell() == (2, 2, "stable")
    # a file written before the tolerance: only exact ties went to the smaller program
    old = json.loads((tmp_path / "archive.json").read_text())
    del old["tolerance"], old["measured_only"]
    for r in old["records"]:
        del r["gpu_failure"]
    (tmp_path / "old.json").write_text(json.dumps(old))
    c = Archive.load(tmp_path / "old.json")
    assert c.tolerance == 0.0 and not c.measured_only and c.records[1].gpu_failure == ""
    assert {k: r.verdict.digest for k, r in c.elites.items()} == {
        k: r.verdict.digest for k, r in a.elites.items()
    }


# --------------------------------------------------------------------------- the prompt


def _seed(problem):
    return Archive().insert(evaluate(problem.seed, problem))


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_prompt_holds_the_parts_and_no_known_form(name, tmp_path):
    p = PROBLEMS[name]
    parent = _seed(p)
    # built as the control experiment builds it: a memory of its own, and no library
    memory = Knowledge(p.name, path=tmp_path / "k.json", library=None)
    prompt = build_prompt(p, parent, [], memory, blind=False)
    assert p.description in prompt
    assert parent.verdict.text in prompt and parent.verdict.message in prompt
    assert "(merge (FIELD EXPR) ...)" in prompt and "rmax" in prompt  # grammar, operators
    # what the exact stage checks, what a proof needs, what the estimate depends on
    assert "Only the outputs decide this stage" in prompt and "is not required here" in prompt
    assert "merge(block(A), block(B)) = block(A ++ B)" in prompt and "cannot be proved" in prompt
    assert "set by the bytes moved per element" in prompt
    assert "the exps computed do not change it" in prompt and "within 2%" in prompt
    assert "single fenced block" in prompt
    assert known_forms_in(p, prompt) == []


def test_a_lemma_about_a_known_form_reaches_every_prompt(tmp_path):
    index = tmp_path / "index.json"
    index.write_text(json.dumps([lemma(SOFTMAX, "online")]))
    shown = llm.Scripted([fenced(SOFTMAX.known["online"])] * 2)
    s = search(shown, 2, tmp_path / "a", library=index)
    for prompt in shown.prompts:
        assert "online: merge of a.d" in known_forms_in(SOFTMAX, prompt)
    assert s["library"] == str(index) and s["lemmas"] == ["softmax-online"]
    assert s["known"]["online"]["lemma"] == "softmax-online"
    assert s["known"]["log-sum-exp"]["lemma"] is None
    text = loop.report(s)
    assert (
        "prompts: 1 proved lemma(s)" in text and "in every prompt as lemma softmax-online" in text
    )
    hidden = llm.Scripted([fenced(SOFTMAX.known["online"])])
    s = search(hidden, 1, tmp_path / "b", library=None)
    assert known_forms_in(SOFTMAX, hidden.prompts[0]) == []
    assert s["library"] is None and s["lemmas"] == [] and s["known"]["online"]["lemma"] is None
    assert "prompts: no proved lemma (no library)" in loop.report(s)


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_blind_prompt_hides_the_operator(name, tmp_path):
    p = PROBLEMS[name]
    (tmp_path / "i.json").write_text(
        json.dumps([{"name": f"{p.name}_merge_block", "statement": f"{p.name}.merge = block"}])
    )
    memory = Knowledge(p.name, blind=True, path=tmp_path / "k.json", library=tmp_path / "i.json")
    wrong = Verdict(
        "exact", f"wrong result for the {p.name} candidate", digest="d1", text="(program x)"
    )
    memory.record(wrong)
    memory.record(Verdict("parse", "does not parse", text=f"(program {p.name}-fast (input x))"))
    prompt = build_prompt(p, _seed(p), [], memory, blind=True)
    assert p.name not in prompt.lower()
    assert p.blind_description in prompt and p.description not in prompt
    assert "(program candidate" in prompt
    assert "op_merge_block: op.merge = block" in prompt  # the lemma is shown, renamed


def test_prompt_shows_inspirations_failures_and_lemmas(tmp_path):
    (tmp_path / "index.json").write_text(
        json.dumps(
            {
                "lemmas": [
                    {
                        "name": "merge_block_a",
                        "statement": "∀ A B, m A B = b (A ++ B)",
                        "problem": "softmax",
                    },
                    {"name": "elsewhere", "statement": "about variance", "problem": "variance"},
                    {"name": "general", "statement": "a fold of a lawful merge is the block"},
                ]
            }
        )
    )
    memory = Knowledge("softmax", path=tmp_path / "k.json", library=tmp_path / "index.json")
    wrong = evaluate(SOFTMAX.known["no-rescale"], SOFTMAX)
    memory.record(wrong)
    archive = Archive()
    parent = archive.insert(evaluate(SOFTMAX.seed, SOFTMAX))
    other = archive.insert(evaluate(SOFTMAX.known["unshifted"], SOFTMAX))
    prompt = build_prompt(SOFTMAX, parent, [other], memory, blind=False)
    assert other.verdict.text in prompt and "float32 nonfinite, estimated 3247 us" in prompt
    assert wrong.text in prompt and wrong.message in prompt
    assert "merge_block_a: ∀ A B" in prompt and "general:" in prompt
    assert "about variance" not in prompt


def test_knowledge_is_kept_once_and_saved(tmp_path):
    memory = Knowledge("softmax", path=tmp_path / "k.json", library=tmp_path / "missing.json")
    wrong = evaluate(SOFTMAX.known["no-rescale"], SOFTMAX)
    good = evaluate(SOFTMAX.known["online"], SOFTMAX)
    for v in (wrong, wrong, good, good):
        memory.record(v)
    memory.save()
    again = Knowledge("softmax", path=tmp_path / "k.json", library=tmp_path / "missing.json")
    assert [e["digest"] for e in again.rejected] == [wrong.digest]
    assert again.accepted[0]["cell"] == [2, 2, "stable"]
    assert again.lemmas() == []
    (tmp_path / "bad.json").write_text("{not json")
    assert (
        Knowledge("softmax", path=tmp_path / "k.json", library=tmp_path / "bad.json").lemmas() == []
    )


def test_knowledge_reads_its_library_once_and_none_means_no_lemma(tmp_path):
    index = tmp_path / "index.json"
    index.write_text(json.dumps([lemma(SOFTMAX, "online")]))
    memory = Knowledge("softmax", path=tmp_path / "k.json", library=index)
    # the prover adds a lemma while the run goes on: the run keeps the lemmas it started with
    index.write_text(json.dumps([lemma(SOFTMAX, "online"), lemma(SOFTMAX, "log-sum-exp")]))
    assert [e["name"] for e in memory.lemmas()] == ["softmax-online"]
    assert (
        "softmax-online" in memory.prompt_section() and "softmax-lse" not in memory.prompt_section()
    )
    assert memory.prompt_section(lemmas=0) == ""
    none = Knowledge("softmax", path=tmp_path / "k.json", library=None)
    assert none.library is None and none.lemmas() == [] and none.prompt_section() == ""


def test_a_kernel_that_fails_on_the_gpu_is_remembered_as_rejected(tmp_path):
    memory = Knowledge("softmax", path=tmp_path / "k.json", library=None)
    good = evaluate(SOFTMAX.known["online"], SOFTMAX)
    memory.record(good)  # accepted by a run without --gpu
    memory.record(good, gpu_failure="the Triton kernel is off on shape rows=3, N=7")
    memory.record(good, gpu_failure="once is enough")
    memory.record(good)
    assert memory.accepted == []
    assert [(e["stage"], e["reason"]) for e in memory.rejected] == [
        ("gpu", "the Triton kernel is off on shape rows=3, N=7")
    ]
    assert "Rejected: the Triton kernel is off" in memory.prompt_section()


# --------------------------------------------------------------------------- the reply


@pytest.mark.parametrize(
    ("reply", "program"),
    [
        (fenced("(program p (input x))"), "(program p (input x))"),
        ("```lisp\r\n(program p (input x))\r\n```", "(program p (input x))"),
        ("```\n(program q)\n```", "(program q)"),
        ("Was:\n```lisp\n(program old)\n```\nNow:\n```lisp\n(program new)\n```", "(program new)"),
        ("```python\nprint(1)\n```\n```lisp\n(program p)\n```", "(program p)"),
        ("I propose (program r (input x) (pass a)) as it is.", "(program r (input x) (pass a))"),
        ("Unfinished: (program r (input x)", "(program r (input x)"),
    ],
)
def test_reply_parsing(reply, program):
    assert parse_reply(reply) == program


@pytest.mark.parametrize("reply", ["I cannot improve on this.", "```python\nprint(1)\n```", ""])
def test_a_reply_without_program_is_an_error(reply):
    with pytest.raises(ProposalError):
        parse_reply(reply)


# --------------------------------------------------------------------------- the loop


def test_same_algorithm_is_up_to_names_and_order():
    rng = np.random.default_rng(0)
    online = SOFTMAX.parse(SOFTMAX.known["online"])
    assert loop.same_algorithm(SOFTMAX, SOFTMAX.parse(ONLINE_RENAMED), online, rng)
    assert not loop.same_algorithm(SOFTMAX, SOFTMAX.parse(SOFTMAX.known["no-rescale"]), online, rng)
    assert not loop.same_algorithm(SOFTMAX, SOFTMAX.parse(SOFTMAX.seed), online, rng)
    welford = VARIANCE.parse(VARIANCE.known["welford"])
    sumsq = VARIANCE.parse(VARIANCE.known["sum-of-squares"])
    assert not loop.same_algorithm(VARIANCE, sumsq, welford, rng)


def test_reached_counts_the_first_iteration_asked_not_the_first_reply():
    a = Archive()
    a.insert(evaluate(SOFTMAX.seed, SOFTMAX))
    online, wrong, lse = (
        evaluate(SOFTMAX.known[k], SOFTMAX) for k in ("online", "no-rescale", "log-sum-exp")
    )
    # with two workers the reply to iteration 2 can come back before the identical one to 1
    for verdict, it in [(online, 2), (online, 1), (wrong, 4), (wrong, 3), (lse, 6), (lse, 5)]:
        a.insert(verdict, it)
    a.insert(Verdict("exact", "wrong", digest="old", text="(program from-an-older-parser"), 7)
    assert [r.status for r in a.records[1:]] == [
        "new-cell",
        "duplicate",
        "rejected",
        "duplicate",
        "new-cell",
        "duplicate",
        "rejected",  # its text no longer parses: reached() passes over it
    ]
    known = loop.reached(SOFTMAX, a, 0)
    assert known["online"]["same_at"] == 1 and known["online"]["cell_reached_at"] == 1
    assert known["no-rescale"]["same_at"] == 3 and known["no-rescale"]["cell_reached_at"] is None
    assert known["log-sum-exp"]["same_at"] == 5 and known["log-sum-exp"]["cell_reached_at"] == 5
    assert known["unshifted"]["same_at"] is None and known["online"]["lemma"] is None


def test_archive_loop_on_scripted_answers(tmp_path):
    answers = [
        fenced(ONLINE_RENAMED),
        "Nothing better comes to mind.",
        fenced(SOFTMAX.known["no-rescale"]),
        fenced(SOFTMAX.known["online"]),
        fenced(SOFTMAX.known["unshifted"]),
        fenced(SOFTMAX.known["online"]),
    ]
    model = llm.Scripted(answers)
    out = tmp_path / "run"
    summary = loop.run(
        SOFTMAX, model, 6, "archive", False, 1, 0, out, tmp_path / "k.json", library=None
    )
    assert summary["usage"]["calls"] == 6 and summary["no_program"] == 1
    lines = lines_of(out)
    assert [x["iteration"] for x in lines] == [0, 1, 2, 3, 4, 5, 6]
    assert [x.get("status", "error") for x in lines] == [
        "new-cell",
        "new-cell",
        "error",
        "rejected",
        "kept",
        "new-cell",
        "duplicate",
    ]
    assert lines[1]["parent"] == lines[0]["verdict"]["digest"]
    assert "usage" in lines[1] and lines[2]["error"].startswith("the reply holds no program")
    known = summary["known"]
    assert known["online"]["cell_reached_at"] == 1 and known["online"]["same_at"] == 1
    assert known["unshifted"]["same_at"] == 5 and known["no-rescale"]["same_at"] == 3
    assert {tuple(c["cell"]) for c in summary["cells"]} == {
        (3, 2, "stable"),
        (2, 2, "stable"),
        (2, 1, "nonfinite"),
    }
    assert not any(c["measured"] for c in summary["cells"])
    assert len(Archive.load(out / "archive.json").records) == 6
    assert json.loads((out / "summary.json").read_text())["known"] == known
    memory = json.loads((tmp_path / "k.json").read_text())
    assert [e["stage"] for e in memory["rejected"]] == ["exact"]
    for text in SOFTMAX.known.values():  # nothing known before the model wrote it
        assert SOFTMAX.parse(text).text() not in model.prompts[0]
    assert "wrong result for N=" in model.prompts[3]  # the failure fed back
    text = loop.report(summary)
    assert "known forms" in text and " estimated " in text and "never shown" not in text


def test_iid_loop_asks_the_same_prompt_in_parallel(tmp_path):
    model = llm.Scripted([fenced(SOFTMAX.known["online"])] * 3 + ["no program"] * 2)
    summary = loop.run(
        SOFTMAX, model, 5, "iid", True, 3, 0, tmp_path / "run", tmp_path / "k.json", library=None
    )
    assert len(model.prompts) == 5 and len(set(model.prompts)) == 1
    assert "softmax" not in model.prompts[0].lower()
    assert summary["usage"]["calls"] == 5 and summary["no_program"] == 2
    assert summary["statuses"] == {"new-cell": 1, "duplicate": 2}


# --------------------------------------------------------------------------- with --gpu


def test_with_gpu_a_failed_kernel_is_no_elite_and_no_parent(tmp_path, monkeypatch):
    off = "the Triton kernel is off on shape rows=3, N=40977 (normal inputs): error 0.9"
    measured = fake_gpu(
        monkeypatch,
        {
            "softmax-3pass": 4.17,
            # online's kernel runs slower than its estimate; the log-fold variant's estimate,
            # 3247 us, is 4.5% below it, but its kernel fails the check
            "softmax-online": 3.40,
            "softmax-logfold": off,
            "mine": ValueError("no layout for this output"),  # no kernel at all
        },
    )
    answers = [SOFTMAX.known["online"], LOGFOLD, LOGFOLD, ONLINE_RENAMED, SOFTMAX.known["online"]]
    model = llm.Scripted([fenced(a) for a in answers])
    s = search(model, 5, tmp_path, gpu=True)
    assert measured == ["softmax-3pass", "softmax-online", "softmax-logfold", "mine"]
    lines = lines_of(tmp_path / "run")
    assert [x["status"] for x in lines] == [
        "new-cell",
        "new-cell",
        "gpu-failed",
        "duplicate",  # not measured again
        "gpu-failed",
        "duplicate",
    ]
    assert lines[2]["gpu"] == {"ok": False, "message": off}
    assert lines[4]["gpu"]["message"] == "no kernel: ValueError: no layout for this output"
    failed = {lines[2]["verdict"]["digest"], lines[4]["verdict"]["digest"]}
    assert not failed & {x.get("parent") for x in lines}
    assert {(c["name"], c["measured"]) for c in s["cells"]} == {
        ("softmax-3pass", True),
        ("softmax-online", True),
    }
    assert [(g["iteration"], g["name"]) for g in s["gpu_failed"]] == [
        (2, "softmax-logfold"),
        (4, "mine"),
    ]
    assert s["gpu"][lines[2]["verdict"]["digest"]]["message"] == off
    memory = json.loads((tmp_path / "k.json").read_text())
    assert {e["digest"] for e in memory["accepted"]} == {lines[1]["verdict"]["digest"]}
    assert [(e["stage"], e["reason"]) for e in memory["rejected"]][0] == ("gpu", off)
    assert off in model.prompts[2]  # the proposer reads why
    text = loop.report(s)
    assert " measured " in text and "not an elite: iteration 2 (softmax-logfold)" in text
    archive = Archive.load(tmp_path / "run" / "archive.json")
    assert archive.measured_only and {r.verdict.name for r in archive.elites.values()} == {
        "softmax-3pass",
        "softmax-online",
    }


def test_with_gpu_the_status_says_what_the_measurement_did(tmp_path, monkeypatch):
    # online runs faster than its estimate; the renamed copy's estimate (3247 us) is above
    # online's measured 3100 us, but its own measurement is 6% faster; the log-fold variant is
    # 0.3% faster still, within the tolerance, and one node larger
    fake_gpu(
        monkeypatch,
        {"softmax-3pass": 4.2, "softmax-online": 3.10, "mine": 2.90, "softmax-logfold": 2.89},
    )
    answers = [SOFTMAX.known["online"], ONLINE_RENAMED, LOGFOLD]
    s = search(llm.Scripted([fenced(a) for a in answers]), 3, tmp_path, gpu=True)
    statuses = [x["status"] for x in lines_of(tmp_path / "run")]
    assert statuses == ["new-cell", "new-cell", "improved", "kept"]
    (elite,) = [c for c in s["cells"] if c["cell"] == [2, 2, "stable"]]
    assert elite["name"] == "mine" and elite["measured"]
    assert elite["fitness_us"] == pytest.approx(2900.0)
    assert s["statuses"] == {"new-cell": 1, "improved": 1, "kept": 1}


def test_with_gpu_a_time_within_the_elite_s_spread_is_a_tie(tmp_path, monkeypatch):
    fake_gpu(
        monkeypatch,
        {
            "softmax-3pass": (4.20, 4.19, 4.21),
            "softmax-online": (3.30, 3.25, 3.40),
            # 3% faster than online, one node larger, and its timings overlap online's
            "softmax-logfold": (3.20, 3.19, 3.26),
        },
    )
    s = search(
        llm.Scripted([fenced(SOFTMAX.known["online"]), fenced(LOGFOLD)]), 2, tmp_path, gpu=True
    )
    assert [x["status"] for x in lines_of(tmp_path / "run")] == ["new-cell", "new-cell", "kept"]
    (elite,) = [c for c in s["cells"] if c["cell"] == [2, 2, "stable"]]
    assert elite["name"] == "softmax-online"
    archive = Archive.load(tmp_path / "run" / "archive.json")
    assert archive.by_digest[elite["digest"]].measured_range == (3250.0, 3400.0)


def test_with_gpu_a_seed_with_no_kernel_stops_the_run_before_any_call(tmp_path, monkeypatch):
    fake_gpu(monkeypatch, {"softmax-3pass": RuntimeError("no CUDA device")})
    model = llm.Scripted([fenced(SOFTMAX.known["online"])])
    with pytest.raises(RuntimeError, match="no CUDA device"):
        loop.run(SOFTMAX, model, 1, out=tmp_path / "run", memory=tmp_path / "k.json", gpu=True)
    assert model.prompts == []


# a string condition is evaluated when the test is set up, so torch is imported only for it
@pytest.mark.skipif("not _cuda()", reason="needs a CUDA GPU, torch and triton")
def test_on_the_gpu_two_copies_of_a_kernel_tie(tmp_path):
    model = llm.Scripted([fenced(SOFTMAX.known["online"]), fenced(ONLINE_RENAMED)])
    s = search(model, 2, tmp_path, gpu=True)
    lines = lines_of(tmp_path / "run")
    assert [x["gpu"]["ok"] for x in lines] == [True, True, True], lines
    us = [x["gpu"]["ms"] * 1000 for x in lines]
    lo = [x["gpu"]["ms_min"] for x in lines]
    hi = [x["gpu"]["ms_max"] for x in lines]
    # the renamed copy compiles to the same kernel: within 2%, or within the spread of the
    # timings, the earlier program stays
    overlap = lo[2] <= hi[1] and lo[1] <= hi[2]
    tie = overlap or abs(us[2] - us[1]) <= 0.02 * min(us[1], us[2])
    expected = "kept" if tie or us[2] > us[1] else "improved"
    assert [x["status"] for x in lines] == ["new-cell", "new-cell", expected]
    assert all(c["measured"] for c in s["cells"])
    (elite,) = [c for c in s["cells"] if c["cell"] == [2, 2, "stable"]]
    assert elite["name"] == ("mine" if expected == "improved" else "softmax-online")


# --------------------------------------------------------------------------- with --prove


def test_prove_labels_every_elite_but_the_seed(tmp_path, monkeypatch):
    calls = fake_prover(monkeypatch, {"softmax-online"})
    model = llm.Scripted([fenced(SOFTMAX.known["online"]), fenced(SOFTMAX.known["unshifted"])])
    s = search(model, 2, tmp_path, prove=True)
    out = tmp_path / "run"
    asked = {"rounds": 3, "workdir": out / "proofs"}
    assert [(name, kw) for name, _, _, kw in calls] == [
        ("softmax-unshifted", asked),
        ("softmax-online", asked),
    ]
    assert all(problem == "softmax" and m is model for _, problem, m, _ in calls)
    assert {c["name"]: c.get("label") for c in s["cells"]} == {
        "softmax-3pass": None,  # the seed is the specification
        "softmax-online": "E",
        "softmax-unshifted": "T",
    }
    assert s["proofs"]["tried"] == 2 and s["proofs"]["proved"] == 1
    assert json.loads((out / "summary.json").read_text())["cells"] == s["cells"]
    text = loop.report(s)
    assert "softmax-online  E" in text
    assert "softmax-unshifted  T (not proved in 3 round(s))" in text
    assert "proofs: 1 of 2 elites but the seed proved (E)" in text


def test_prove_does_not_send_an_elite_whose_merge_is_no_homomorphism(tmp_path, monkeypatch):
    calls = fake_prover(monkeypatch, {"softmax-online", "softmax-firstmax"})
    model = llm.Scripted([fenced(SOFTMAX.known["online"]), fenced(FIRSTMAX)])
    s = search(model, 2, tmp_path, prove=True)
    assert [name for name, *_ in calls] == ["softmax-online"]
    cells = {c["name"]: c for c in s["cells"]}
    first = cells["softmax-firstmax"]  # right on every test, float32 nonfinite: a cell of its own
    assert first["cell"] == [2, 2, "nonfinite"] and first["label"] == "T"
    assert first["proof"]["rounds"] == 0 and "theorem is false" in first["proof"]["message"]
    assert cells["softmax-online"]["label"] == "E"
    assert s["proofs"]["tried"] == 2 and s["proofs"]["proved"] == 1


def test_prove_runs_the_prover_and_a_proof_it_refuses_leaves_t(tmp_path):
    # a program the library does not hold (the online form is in it, and would not be asked)
    held = [SOFTMAX.parse(t) for t in (SOFTMAX.known["online"], LOGFOLD)]
    lib = provers.LIBRARY
    assert [provers.in_library(p, "softmax-3pass", lib) is not None for p in held] == [True, False]
    cheat = "```lean merge_block_a\nsorry\n```\n\n```lean spec_eq\nsorry\n```\n"
    model = llm.Scripted([fenced(LOGFOLD)] + [cheat] * 3)
    s = search(model, 1, tmp_path, prove=True)
    (logfold,) = [c for c in s["cells"] if c["iteration"] == 1]
    assert logfold["label"] == "T" and not logfold["proof"]["proved"]
    assert s["usage"]["calls"] == 1  # the search's; the prover's calls are counted apart
    assert s["proofs"]["usage"]["calls"] == len(model.prompts) - 1 == 3


# --------------------------------------------------------------------------- the command lines


def test_the_loop_command_line_takes_a_library_and_prove(tmp_path, monkeypatch, capsys):
    runs: list[dict] = []
    real = loop.run

    def spy(*args, **kwargs):
        runs.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(loop, "run", spy)
    monkeypatch.setattr(loop, "model_from", lambda spec: llm.Scripted([fenced(LOGFOLD)]))
    fake_prover(monkeypatch, {"softmax-logfold"})
    where = ["--out", str(tmp_path / "run"), "--memory", str(tmp_path / "k.json")]
    loop.main(["--iterations", "1", "--workers", "1", *where, "--library", "none", "--prove"])
    assert runs[-1]["library"] is None and runs[-1]["prove"] is True
    printed = capsys.readouterr().out
    assert "prompts: no proved lemma (no library)" in printed and "softmax-logfold  E" in printed
    loop.main(["--iterations", "1", "--workers", "1", *where])
    assert runs[-1]["library"] == LIBRARY_INDEX and runs[-1]["prove"] is False
    with pytest.raises(SystemExit):
        loop.main(["--library", str(tmp_path / "missing.json")])
    assert loop.library_option("NONE") is None


def test_model_specs():
    claude = loop.model_from("sonnet")
    assert isinstance(claude, llm.ClaudeCLI) and claude.model == "sonnet"
    assert loop.model_from("claude:opus").model == "opus"
    endpoint = loop.model_from("openai:qwen@http://localhost:9/v1")
    assert isinstance(endpoint, llm.OpenAICompatible)
    assert endpoint.model == "qwen" and endpoint.base_url == "http://localhost:9/v1"


def _control(tmp_path, monkeypatch, *extra: str, between=None) -> tuple[list, list[dict]]:
    """Run the control on softmax in archive and IID mode, one iteration each, with a scripted
    model behind a spec with a colon; ``between`` runs before every run but the first. Returns
    the models made, with their spec, and the result rows."""
    made: list[tuple[str, llm.Scripted]] = []

    def from_spec(spec):
        if made and between is not None:
            between()
        model = llm.Scripted([fenced(SOFTMAX.known["online"])])
        made.append((spec, model))
        return model

    monkeypatch.setattr(llm, "from_spec", from_spec)
    out = tmp_path / "exp"
    args = ["--problems", "softmax", "--modes", "archive", "iid", "--seeds", "0"]
    args += ["--iterations", "1", "--workers", "1", "--model", "test:model", "--out", str(out)]
    experiments.main([*args, *extra])
    return made, json.loads((out / "results.json").read_text())


def test_the_control_shows_no_lemma_by_default(tmp_path, monkeypatch):
    runs: list[dict] = []
    real = loop.run

    def spy(*args, **kwargs):
        runs.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(loop, "run", spy)
    made, rows = _control(tmp_path, monkeypatch)
    assert [spec for spec, _ in made] == ["test:model", "test:model"]  # through llm.from_spec
    assert [r["library"] for r in runs] == [None, None]
    assert [(r["mode"], r["library"], r["lemmas"]) for r in rows] == [
        ("archive", None, []),
        ("iid", None, []),
    ]
    for _, model in made:
        assert model.prompts and all(known_forms_in(SOFTMAX, q) == [] for q in model.prompts)
    results = (tmp_path / "exp" / "results.md").read_text()
    assert results.startswith("Proved lemmas in the prompts: none")


def test_the_control_copies_the_library_once(tmp_path, monkeypatch):
    index = tmp_path / "index.json"
    first, later = lemma(SOFTMAX, "online"), lemma(SOFTMAX, "log-sum-exp")
    index.write_text(json.dumps([first]))

    def prover_adds_a_lemma():
        index.write_text(json.dumps([first, later]))

    made, rows = _control(
        tmp_path, monkeypatch, "--library", str(index), between=prover_adds_a_lemma
    )
    copy = tmp_path / "exp" / "library.json"
    assert json.loads(copy.read_text()) == [first]
    assert [(r["library"], r["lemmas"]) for r in rows] == [(str(copy), ["softmax-online"])] * 2
    assert all(r["reached"]["online"]["lemma"] == "softmax-online" for r in rows)
    assert len(made) == 2
    for _, model in made:
        for prompt in model.prompts:
            assert first["statement"] in prompt and later["statement"] not in prompt
    assert "(in the prompt as a lemma)" in (tmp_path / "exp" / "results.md").read_text()


# --------------------------------------------------------------------------- the enumerator


@pytest.mark.parametrize(
    ("problem", "start"),
    [(SOFTMAX, SOFTMAX.known["no-rescale"]), (ATTENTION, ATTENTION.known["online"])],
    ids=["softmax", "attention"],
)
def test_enumerator_finds_the_rescaled_merge(problem, start):
    res = synthesise(problem, start, size_cap=13, time_cap=60)
    assert res.verdict is not None, res.gave_up
    assert res.verdict.correct, res.verdict.message
    assert res.verdict.cell() == evaluate(problem.known["online"], problem).cell()
    assert [f.field for f in res.fields] == [f for f, _ in problem.parse(start).passes[0].block]
    assert max(f.size for f in res.fields) == 13 and all(f.enumerated > 0 for f in res.fields)
    assert loop.same_algorithm(
        problem,
        problem.parse(res.text),
        problem.parse(problem.known["online"]),
        np.random.default_rng(1),
    )


def test_enumerator_gives_up_and_says_why():
    res = synthesise(VARIANCE, VARIANCE.known["welford"], size_cap=13, time_cap=60)
    assert res.verdict is None and res.text is None
    assert "field q" in res.gave_up and "size 13" in res.gave_up
    assert [f.field for f in res.fields] == ["n", "mu", "q"] and res.fields[-1].expr == ""
    late = synthesise(SOFTMAX, SOFTMAX.known["online"], time_cap=0.0)
    assert "time cap" in late.gave_up


def test_with_gpu_close_times_are_timed_side_by_side_and_the_ratio_decides(tmp_path, monkeypatch):
    """Overlapping timings: the two kernels are timed again in alternation, and their ratio
    (5% faster, beyond the 2% tolerance) makes the newcomer the elite."""
    times = {
        "softmax-3pass": (4.20, 4.19, 4.21),
        "softmax-online": (3.30, 3.25, 3.40),
        "softmax-logfold": (3.20, 3.19, 3.26),
    }
    measured = fake_gpu(monkeypatch, times, {("softmax-logfold", "softmax-online"): 0.95})
    s = search(
        llm.Scripted([fenced(SOFTMAX.known["online"]), fenced(LOGFOLD)]), 2, tmp_path, gpu=True
    )
    assert ("paired", "softmax-logfold", "softmax-online") in measured
    assert [x["status"] for x in lines_of(tmp_path / "run")] == ["new-cell", "new-cell", "improved"]
    (elite,) = [c for c in s["cells"] if c["cell"] == [2, 2, "stable"]]
    assert elite["name"] == "softmax-logfold"


def test_with_gpu_a_paired_tie_goes_to_the_smaller_program(tmp_path, monkeypatch):
    times = {
        "softmax-3pass": (4.20, 4.19, 4.21),
        "softmax-online": (3.30, 3.25, 3.40),
        "softmax-logfold": (3.20, 3.19, 3.26),
    }
    fake_gpu(monkeypatch, times, {("softmax-logfold", "softmax-online"): 0.99})
    s = search(
        llm.Scripted([fenced(SOFTMAX.known["online"]), fenced(LOGFOLD)]), 2, tmp_path, gpu=True
    )
    (elite,) = [c for c in s["cells"] if c["cell"] == [2, 2, "stable"]]
    assert elite["name"] == "softmax-online"  # within 2%: the smaller program


def test_with_gpu_times_far_apart_are_not_timed_again(tmp_path, monkeypatch):
    times = {"softmax-3pass": 4.2, "softmax-online": 3.3, "softmax-logfold": 2.0}
    measured = fake_gpu(monkeypatch, times, {})
    search(llm.Scripted([fenced(SOFTMAX.known["online"]), fenced(LOGFOLD)]), 2, tmp_path, gpu=True)
    assert not [m for m in measured if isinstance(m, tuple)]


@pytest.mark.skipif("not _cuda()", reason="needs a CUDA GPU, torch and triton")
def test_on_the_gpu_a_kernel_timed_side_by_side_with_itself_has_ratio_one():
    from discovery import gpu

    prog = SOFTMAX.parse(SOFTMAX.known["online"])
    pair = gpu.paired(prog, prog, SOFTMAX, shape=(256, 4096), rounds=3)
    assert 0.9 < pair["ratio"] < 1.1 and len(pair["ratios"]) == 3
