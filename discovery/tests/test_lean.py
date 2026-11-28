"""The proof tier: the Lean export of every seed and known form, splicing, the judge (the
whitelist of the model's text, the kernel replay and the checker that reads the compiled
modules), and the acceptance of a correct proof.

The tests that run Lean (10 to 30 s each once the framework is built: a check compiles two modules
and runs the checker) are skipped when lake cannot be found.
"""

from __future__ import annotations

import json
import math
import types
from pathlib import Path

import pytest

from discovery import ir, lean_export, llm, prove
from discovery.problems import PROBLEMS, SOFTMAX

FORMS = [
    (p, name, text) for p in PROBLEMS.values() for name, text in {"seed": p.seed, **p.known}.items()
]


def _has_lake() -> bool:
    try:
        prove.lake()
    except prove.LeanError:
        return False
    return True


needs_lean = pytest.mark.skipif(not _has_lake(), reason="lake not found")


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """One work directory for the Lean tests: a candidate's statements compile once."""
    return tmp_path_factory.mktemp("lean")


# The attacks of the review, on softmax-no-rescale, whose merge is false. The whitelist rejects
# each before Lean runs. Two of them, the faked report and the kernel's check turned off, also go
# through `check` directly, as if they had slipped past the whitelist: the checker and the kernel
# replay must still reject them.
NO_RESCALE = SOFTMAX.known["no-rescale"]
# an elaborator for `#print axioms` that prints a clean report, with a `sorryAx` proof
FAKE_REPORT = """open Lean Elab Command in
@[command_elab Lean.Parser.Command.printAxioms]
def fakeReport : CommandElab := fun stx => do
  logInfo m!"'{(← getCurrNamespace) ++ stx[2].getId}' depends on axioms: [propext]"
"""
# the kernel's check off for one theorem, through a comment between `set_option` and the option
KERNEL_OFF = """set_option/- -/debug.skipKernelTC true in
theorem two_eq_three : (2 : Nat) = 3 := by decide +kernel
"""
ATTACKS = {
    "print axioms faked": (FAKE_REPORT, {"merge_block_a": "exact sorryAx _ false"}),
    "print axioms faked in a proof block": (
        None,
        {"spec_eq": "rfl\n\n@[command_elab Lean.Parser.Command.printAxioms]\ndef f := 0"},
    ),
    "set_option behind a comment": (
        KERNEL_OFF,
        {"merge_block_a": "exact absurd two_eq_three (by decide)"},
    ),
    "attribute behind a comment": (
        "attribute/- -/[command_elab Lean.Parser.Command.printAxioms] fakeReport",
        {"merge_block_a": "exact sorryAx _ false"},
    ),
    "namespace and end": (
        "end Discovery.Library.SoftmaxNoRescale\n"
        "namespace Discovery.Library.SoftmaxNoRescale.Fake\n"
        "theorem merge_block_a_statement : True := trivial",
        {},
    ),
    "run_tac": (
        None,
        {"merge_block_a": "run_tac Lean.Elab.Tactic.evalTactic (← `(tactic| sorry))"},
    ),
    "native evaluation": (None, {"merge_block_a": "decide +native"}),
    "eval% at elaboration": (None, {"spec_eq": "exact eval% (rfl : 1 = 1)"}),
    "a theorem in a proof block": (None, {"spec_eq": "rfl\ntheorem t : True := trivial"}),
}

# the softmax seed's obligations, proved by hand: the max merge, the sum merge, and the seed
# against itself
SEED_PROOFS = {
    "merge_block_a": "simp only [merge_a, block_a, lmax_append hA hB]",
    "merge_block_b": "simp only [merge_b, block_b, List.map_append, List.sum_append]",
    "spec_eq": "rfl",
}


def _reply(proofs: dict[str, str], helpers: str | None = None) -> str:
    blocks = [f"```lean {name}\n{body}\n```" for name, body in proofs.items()]
    if helpers is not None:
        blocks.insert(0, f"```lean helpers\n{helpers}\n```")
    return "Here are the proofs.\n\n" + "\n\n".join(blocks) + "\n"


def _seed():
    return SOFTMAX.parse(SOFTMAX.seed)


def _attack_reply(name: str) -> tuple[str | None, dict[str, str]]:
    """The helpers and the proofs of an attack on softmax-no-rescale, harmless proofs elsewhere."""
    helpers, proofs = ATTACKS[name]
    return helpers, {"merge_block_a": "simp", "spec_eq": "rfl", **proofs}


@pytest.mark.parametrize(
    ("problem", "name", "text"), FORMS, ids=[f"{p.name}-{n}" for p, n, _ in FORMS]
)
def test_every_form_states_its_obligations(problem, name, text):
    prog = problem.parse(text)
    lf = lean_export.export(prog, problem)
    assert [o.name for o in lf.obligations] == [
        *(f"merge_block_{p.name}" for p in prog.passes),
        "spec_eq",
    ]
    assert lf.text.count("\n  sorry\n") == len(lf.obligations)
    assert list(lf.checks.values()) == [o.name for o in lf.obligations] + ["blocked_eq_spec"]
    for o in lf.obligations:  # each statement is fixed before the slots and checked after them
        assert f"def {o.name}_statement : Prop :=" in lf.text
        assert f"theorem {o.name}_holds : {o.name}_statement" in lf.text
    assert lf.text.startswith("import Discovery.Blocked\n")
    # the judge's two modules: the statements alone, then the rest, at the same lines
    assert "sorry" not in lf.statements and "theorem" not in lf.statements
    assert lf.statements.endswith(f"\nend {lf.namespace}\n")
    for o in lf.obligations:
        assert f"def {o.name}_statement : Prop :=" in lf.statements
    module, offset = lf.proofs(lf.text, "S")
    assert module.startswith(f"import S\nnamespace {lf.namespace}\n")
    lines, own = lf.text.splitlines(), module.splitlines()
    start = own.index("/-! ## The proofs -/")
    assert own[start:] == lines[start + offset :]
    slots = {lean_export.HELPERS: "", **{o.name: "  sorry" for o in lf.obligations}}
    assert lf.slots(lf.text) == slots
    assert lf.slots(lf.text.replace("-- helper lemmas", "-- helpers")) is None


def test_splice_puts_each_body_in_its_slot():
    lf = lean_export.export(_seed(), SOFTMAX)
    proofs = {
        "merge_block_a": "by\n    simp only [merge_a, block_a]\n    rw [lmax_append hA hB]",
        "merge_block_b": "  simp only [merge_b]\n  rfl",
        "spec_eq": "rfl",
    }
    helpers = "theorem two : (2 : ℝ) = 1 + 1 := by\n  norm_num"
    text, spans = lf.layout(proofs, helpers)
    lines = text.splitlines()

    def slot(name):
        first, last = spans[name]
        return lines[first - 1 : last]

    assert slot("merge_block_a") == ["  simp only [merge_a, block_a]", "  rw [lmax_append hA hB]"]
    assert slot("merge_block_b") == ["  simp only [merge_b]", "  rfl"]
    assert slot("spec_eq") == ["  rfl"]
    assert slot("helpers") == helpers.splitlines()
    assert lines[spans["spec_eq"][0] - 2].endswith(":= by")
    assert text == lf.splice(proofs, helpers)
    with pytest.raises(ValueError, match="missing"):
        lf.splice({"merge_block_a": "rfl"})
    with pytest.raises(ValueError, match="empty"):
        lf.splice({**proofs, "spec_eq": "by"})


OPTION = "set_option (other than `set_option maxHeartbeats N in`)"


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("exact sorry", ["sorry"]),
        ("exact sorryAx _ false", ["sorryAx"]),
        ("axiom cheat : False", ["axiom"]),
        ("admit", ["admit"]),
        ("native_decide", ["native_decide"]),
        ("decide +native", ["native"]),
        ("@[implemented_by f] def g := 1", ["@[", "def", "implemented_by"]),
        ("macro_rules | `(tactic| done) => `(tactic| rfl)", ["macro_rules"]),
        ("set_option debug.skipKernelTC true in", [OPTION]),
        ("set_option/- -/debug.skipKernelTC true in", [OPTION]),
        ("set_option/--/debug.skipKernelTC true in", [OPTION]),
        ("set_option maxHeartbeats/- -/400000 in ring", [OPTION]),
        ("set_option maxHeartbeats 400000 in\nring", []),
        ("#eval IO.println 1", ["#eval", "IO", "println"]),
        ("exact eval% (2 + 2)", ["eval%"]),
        ("exact «Lean».Elab.Term.elabTerm", ["Lean."]),
        ("open Real in exact exp_pos 1", ["open"]),
        ("unsafeIO (pure ())", ["unsafeIO"]),
        ("simp only [List.sum_append, Real.exp_add, max_def]; ring", []),
        ("induction A with\n| nil => simp\n| cons y ys ih => rw [ih]; ring", []),
        ("rfl\ntheorem t : True := trivial", ["theorem"]),  # a declaration in a proof block
    ],
)
def test_banned_constructs_are_found(text, found):
    assert prove.banned(text) == found


def test_the_helper_block_holds_theorems_only():
    assert prove.banned("private theorem t : True := trivial\nlemma u : True := t", True) == []
    assert prove.banned("instance : Inhabited ℕ := ⟨0⟩", True) == ["instance"]
    assert prove.banned("example : True := trivial", True) == ["example"]
    assert prove.banned("attribute [simp] t", True) == ["attribute"]
    assert prove.banned("@[simp] theorem t : True := trivial", True) == ["@["]


@pytest.mark.parametrize("attack", list(ATTACKS))
def test_every_attack_is_rejected_before_lean(attack, work):
    helpers, proofs = _attack_reply(attack)
    found = {n: prove.banned(t) for n, t in proofs.items()}
    found[lean_export.HELPERS] = prove.banned(helpers or "", helpers=True)
    assert any(found.values()), found
    model = llm.Scripted([_reply(proofs, helpers)])
    prog = SOFTMAX.parse(NO_RESCALE)
    r = prove.prove(prog, SOFTMAX, model, rounds=1, workdir=work, library=None)
    assert not r.proved and r.lean_seconds == 0, r.errors
    assert r.attempts[0]["status"] == "banned" and "rejected before Lean" in r.errors[0]


def test_reply_format():
    names = {"merge_block_a", "spec_eq"}
    proofs, helpers = prove.parse_reply(
        _reply({"merge_block_a": "simp", "spec_eq": "rfl"}, "theorem t : True := trivial"), names
    )
    assert proofs == {"merge_block_a": "simp\n", "spec_eq": "rfl\n"}
    assert helpers == "theorem t : True := trivial\n"
    assert prove.parse_reply(_reply({"spec_eq": "rfl"}), names)[1] is None
    with pytest.raises(prove.ReplyError, match="unknown"):
        prove.parse_reply(_reply({"merge_block_z": "rfl"}), names)
    with pytest.raises(prove.ReplyError, match="no block"):
        prove.parse_reply("```lean\nrfl\n```", names)
    framework = prove.framework_statements()
    assert "theorem eval_eq_block" in framework and ":= by" not in framework


def test_lean_output_is_read():
    # Lean 4.36 prints a `#print axioms` report with no position, and a long one on several lines
    output = (
        "F.lean:12:4: error(lean.unknownIdentifier): Unknown identifier `foo`\n"
        "F.lean:20:0: warning: declaration uses `sorry`\n"
        "'N.a_holds' depends on axioms: [propext, Classical.choice, Quot.sound]\n"
        "'N.b_holds' depends on axioms: [propext,\n sorryAx,\n Classical.choice]\n"
        "'N.c_holds' does not depend on any axioms\n"
        "'N.merge_block_a'_holds' depends on axioms: [propext]\n"  # a pass named a'
        "'N.d''_holds' does not depend on any axioms\n"
    )
    messages = prove.parse_messages(output)
    assert [(m.line, m.severity, m.text) for m in messages] == [
        (12, "error", "Unknown identifier `foo`"),
        (20, "warning", "declaration uses `sorry`"),
    ]
    assert prove.axiom_reports(output, "N") == [
        ("a_holds", ["propext", "Classical.choice", "Quot.sound"]),
        ("b_holds", ["propext", "sorryAx", "Classical.choice"]),
        ("c_holds", []),
        ("merge_block_a'_holds", ["propext"]),
        ("d''_holds", []),
    ]


def test_library_names_do_not_collide(tmp_path):
    prog = SOFTMAX.parse(SOFTMAX.known["online"])
    assert prove.library_module(SOFTMAX, prog, tmp_path) == "SoftmaxOnline"
    entry = {"module": "Discovery.Library.SoftmaxOnline", "digest": "another program"}
    (tmp_path / "index.json").write_text(json.dumps([entry]), encoding="utf-8")
    assert prove.library_module(SOFTMAX, prog, tmp_path) == "SoftmaxOnline" + prog.digest()[:6]


def test_banned_text_is_rejected_before_lean(work):
    model = llm.Scripted([_reply({**SEED_PROOFS, "merge_block_b": "sorry"})])
    r = prove.prove(_seed(), SOFTMAX, model, rounds=1, workdir=work, library=None)
    assert not r.proved and r.lean_seconds == 0
    assert "merge_block_b` contains sorry" in r.errors[0]
    assert "`merge_block_b` contains sorry" in prove.text_feedback(1, r.errors[0])


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_a_program_lean_cannot_state_is_a_clean_result(value, tmp_path):
    # the IR's reader refuses these literals; a program built by hand still reaches the prover
    prog = _seed()
    ((name, expr),) = prog.output.exprs
    prog.output = ir.Output("elementwise", [(name, ir.Call("*", (ir.Num(value), expr)))])
    model = llm.Scripted([])
    r = prove.prove(prog, SOFTMAX, model, rounds=1, workdir=tmp_path, library=None)
    assert not r.proved and r.message.startswith("cannot be stated in Lean"), r.message
    assert model.prompts == [] and r.rounds == 0
    with pytest.raises(lean_export.ExportError, match="not a real number"):
        lean_export.export(prog, SOFTMAX)


@needs_lean
def test_every_export_type_checks_with_sorry(tmp_path):
    files = [lean_export.export(problem.parse(text), problem) for problem, _, text in FORMS]
    path = tmp_path / "All.lean"
    path.write_text(files[0].header + "".join(f.body for f in files), encoding="utf-8")
    output, _ = prove.run_lean(path)
    assert [m for m in prove.parse_messages(output) if m.severity == "error"] == []
    for f in files:
        reports = dict(prove.axiom_reports(output, f.namespace))
        for c, theorem in f.checks.items():  # every proof is `sorry`, and the checks see it
            assert "sorryAx" in reports[c], (f.module, theorem)


@needs_lean
def test_each_obligation_is_judged_on_its_own(work):
    lf = lean_export.export(_seed(), SOFTMAX)
    proofs = {**SEED_PROOFS, "merge_block_b": "exact sorryAx _ false", "spec_eq": "exact le_rfl"}
    v = prove.check(lf, proofs, "", work / "partial.lean")
    assert v.accepted == {"merge_block_a": True, "merge_block_b": False, "spec_eq": False}
    assert "sorryAx" in v.axioms["merge_block_b_holds"]  # no error, but an axiom
    # the error is in the proofs' module, reported at its line of the spliced file
    assert [v.region(m.line) for m in v.errors] == ["spec_eq"]
    assert not v.ok and not v.checked  # an error: no module for the checker to read


@needs_lean
def test_the_checker_reads_the_axioms_of_the_compiled_module(work):
    lf = lean_export.export(_seed(), SOFTMAX)
    proofs = {**SEED_PROOFS, "merge_block_b": "exact sorryAx _ false"}
    v = prove.check(lf, proofs, "", work / "sorry_ax.lean")
    assert v.checked and not v.errors and not v.problems, v.problems
    assert v.accepted == {"merge_block_a": True, "merge_block_b": False, "spec_eq": True}
    assert "sorryAx" in v.axioms["merge_block_b_holds"]
    assert "sorryAx" in v.axioms["blocked_eq_spec_holds"]  # it uses merge_block_b
    assert not v.composed and not v.ok and v.checker_seconds > 0


def _modules(root, statements: str, proofs: str) -> list:
    """Compile two hand-written modules as the judge's pair: the search path of the checker."""
    stmts, pfs = root / "statements", root / "proofs"
    for where, name, text in ((stmts, prove.STATEMENTS, statements), (pfs, prove.PROOFS, proofs)):
        where.mkdir(parents=True, exist_ok=True)
        (where / f"{name}.lean").write_text(text, encoding="utf-8")
        output, _ = prove.run_lean(
            where / f"{name}.lean", olean=where / f"{name}.olean", roots=[pfs, stmts]
        )
        assert not [m for m in prove.parse_messages(output) if m.severity == "error"], output
    return [pfs, stmts]


@needs_lean
def test_the_checker_wants_each_statement_of_the_first_module_as_the_type(tmp_path):
    statements = """namespace T
universe u
def good_statement : Prop := ∀ n : Nat, n + 0 = n
def bad_statement : Prop := ∀ n : Nat, 0 + n = n
def poly_statement : Prop := ∀ (α : Sort u) (a : α), a = a
def inst_statement {α : Type u} [Inhabited α] : Prop := (default : α) = default
def early_statement : Prop := True
theorem early_holds : early_statement := trivial
end T
"""
    proofs = f"""import {prove.STATEMENTS}
namespace T
universe u
theorem good_holds : good_statement := fun _ => rfl
theorem bad_holds : ∀ n : Nat, n + 0 = n := fun _ => rfl
theorem poly_holds : poly_statement.{{u}} := fun _ _ => rfl
theorem inst_holds {{α : Type u}} [Inhabited α] : inst_statement (α := α) := rfl
def late_statement : Prop := True
theorem late_holds : late_statement := trivial
end T
"""
    names = ["good", "bad", "poly", "inst", "missing", "early", "late"]
    lf = types.SimpleNamespace(namespace="T", checks={f"{x}_holds": x for x in names})
    verdict, _ = prove.run_checker(lf, _modules(tmp_path, statements, proofs))
    assert verdict["replay"] is None and not prove._whole(verdict), verdict
    seen = {x: prove._typed(verdict["checks"][x]) for x in names}
    assert seen == {x: x in ("good", "poly", "inst") for x in names}
    assert verdict["checks"]["bad"]["found"] and not verdict["checks"]["bad"]["type"]
    assert not verdict["checks"]["early"]["found"]  # its own theorem, in the statements' module
    assert not verdict["checks"]["late"]["statement"]  # a statement the proofs' module made
    assert verdict["checks"]["good"]["axioms"] == []


@needs_lean
def test_native_decide_shows_in_the_axioms(tmp_path):
    # Lean 4.36's native_decide runs compiled code while elaborating and adds an axiom of its own
    statements = "def T.n_statement : Prop := 2 + 2 = 4\n"
    proofs = (
        f"import {prove.STATEMENTS}\n"
        "theorem T.n_holds : T.n_statement := show 2 + 2 = 4 by native_decide\n"
    )
    lf = types.SimpleNamespace(namespace="T", checks={"n_holds": "n"})
    verdict, _ = prove.run_checker(lf, _modules(tmp_path, statements, proofs))
    info = verdict["checks"]["n"]
    assert prove._typed(info) and not prove._whole(verdict), verdict
    assert any("_native.native_decide" in a for a in info["axioms"])
    assert not prove._clean(info["axioms"])


@needs_lean
def test_a_forged_axiom_report_does_not_fool_the_checker(work):
    # the attack of the review, past the whitelist: Lean's own report is clean, the checker's not
    helpers, proofs = _attack_reply("print axioms faked")
    lf = lean_export.export(SOFTMAX.parse(NO_RESCALE), SOFTMAX)
    v = prove.check(lf, proofs, helpers, work / "forged.lean")
    output, _ = prove.run_lean(work / "forged.lean")  # the file as one, as the old judge read it
    assert dict(prove.axiom_reports(output, lf.namespace))["merge_block_a_holds"] == ["propext"]
    assert v.checked and not v.errors, v.errors
    assert "sorryAx" in v.axioms["merge_block_a_holds"]
    assert not v.accepted["merge_block_a"] and not v.ok


@needs_lean
def test_the_kernel_replay_finds_a_declaration_it_did_not_check(work):
    # the comment trick past the whitelist: Lean accepts 2 = 3 with the kernel's check off, and
    # with it the false merge; replaying the module in the kernel rejects it
    helpers, proofs = _attack_reply("set_option behind a comment")
    lf = lean_export.export(SOFTMAX.parse(NO_RESCALE), SOFTMAX)
    v = prove.check(lf, proofs, helpers, work / "kernel_off.lean")
    assert v.checked and not v.errors, v.errors
    assert any("kernel rejects" in p and "two_eq_three" in p for p in v.problems), v.problems
    assert not any(v.accepted.values()) and not v.ok


@needs_lean
def test_the_checker_rejects_what_the_text_rules_let_through(work):
    # `stop` (core Lean: `repeat sorry`) holds no forbidden word; the axioms give it away
    model = llm.Scripted([_reply({**SEED_PROOFS, "merge_block_b": "stop\n  simp"})])
    r = prove.prove(_seed(), SOFTMAX, model, rounds=1, workdir=work, library=None)
    assert not r.proved and r.attempts[0]["status"] == "rejected by the checker", r.errors
    assert "sorryAx" in r.axioms["merge_block_b"] and "axioms not allowed" in r.errors[-1]


@needs_lean
def test_a_correct_proof_is_accepted_and_kept(tmp_path, work):
    bad = _reply({**SEED_PROOFS, "merge_block_b": "sorry"})
    good = _reply({"merge_block_b": SEED_PROOFS["merge_block_b"]})  # the others are kept
    model = llm.Scripted([bad, good])
    library = tmp_path / "Library"
    r = prove.prove(_seed(), SOFTMAX, model, rounds=2, workdir=work, library=library)
    assert r.proved and r.rounds == 2, r.errors
    assert "`merge_block_b` contains sorry" in model.prompts[1]
    assert set(r.axioms) == {"merge_block_a", "merge_block_b", "spec_eq", "blocked_eq_spec"}
    assert all(set(a) <= prove.ALLOWED_AXIOMS for a in r.axioms.values())
    assert r.attempts[1]["checker_seconds"] > 0
    (entry,) = json.loads((library / "index.json").read_text(encoding="utf-8"))
    assert entry["module"] == "Discovery.Library.Softmax3pass"
    assert entry["name"] == "softmax-3pass" and "(m (max L.m R.m))" in entry["statement"]
    assert set(entry["statements"]) == set(r.axioms)
    assert (library / "Softmax3pass.lean").read_text(encoding="utf-8") == r.text


@needs_lean
def test_a_stored_proof_of_the_library_passes_the_judge(work):
    (entry,) = [e for e in prove._index(prove.LIBRARY) if e["file"] == "SoftmaxOnline.lean"]
    r = prove.recheck(entry, prove.LIBRARY, work)
    assert r["ok"], r
    assert set(r["axioms"]) == {"merge_block_a_holds", "spec_eq_holds", "blocked_eq_spec_holds"}


@needs_lean
def test_a_relative_work_directory_is_judged_like_an_absolute_one(work, monkeypatch):
    # Lean runs in discovery/lean: a work directory relative to the caller's directory (a run's
    # `--out discovery/runs/...`) once made every check fail with "Lean exited with 1"
    (entry,) = [e for e in prove._index(prove.LIBRARY) if e["file"] == "SoftmaxOnline.lean"]
    monkeypatch.chdir(work)
    r = prove.recheck(entry, prove.LIBRARY, Path("relative") / "work")
    assert r["ok"], r
    assert (work / "relative" / "work" / "statements").is_dir()


@needs_lean
def test_a_program_the_library_holds_is_judged_again_not_asked(work, tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    (entry,) = [e for e in prove._index(prove.LIBRARY) if e["file"] == "SoftmaxOnline.lean"]
    stored = (prove.LIBRARY / entry["file"]).read_text(encoding="utf-8")
    (library / entry["file"]).write_text(stored, encoding="utf-8")
    (library / "index.json").write_text(json.dumps([entry]), encoding="utf-8")
    # the same program under another name, as a search proposes it
    renamed = SOFTMAX.parse(SOFTMAX.known["online"].replace("softmax-online", "softmax-2pass"))
    model = llm.Scripted([])
    r = prove.prove(renamed, SOFTMAX, model, rounds=3, workdir=work, library=library)
    assert r.proved and r.rounds == 0 and model.prompts == [], r.message
    assert r.message.startswith("already in the library as softmax-online (SoftmaxOnline.lean)")
    assert r.library == str(library / "SoftmaxOnline.lean")
    assert prove._index(library) == [entry]


def test_a_new_proof_of_a_program_the_library_holds_replaces_it(tmp_path):
    prog = SOFTMAX.parse(SOFTMAX.known["online"])
    for module in ("First", "Second"):
        lf = lean_export.export(prog, SOFTMAX, module=module)
        result = prove.ProofResult("softmax", prog.name, prog.digest())
        prove.add_to_library(lf, prog, lf.text, "", result, tmp_path, "scripted")
    assert [e["module"] for e in prove._index(tmp_path)] == ["Discovery.Library.Second"]
    assert sorted(p.name for p in tmp_path.glob("*.lean")) == ["Second.lean"]


def test_a_lean_failure_with_no_message_says_what_lean_printed(tmp_path, monkeypatch):
    monkeypatch.setattr(prove, "_lean", lambda args, roots, timeout: (1, "file not found: X", 0.5))
    output, _ = prove.run_lean(tmp_path / "A.lean")
    assert "Lean exited with 1, printing: file not found: X" in output
