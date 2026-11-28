"""The proof tier: a language model writes the proofs of a candidate's Lean obligations, Lean
checks them.

``prove`` exports the candidate (``lean_export``), asks the model for the body of every proof slot,
splices the bodies into the generated file and checks it with Lean; Lean's errors go back to the
model for the next round. The judge does not trust anything the model's text could print or
change while Lean reads it:

1. ``banned``: the model's text is read as raw text first. The helper block may hold only
   ``theorem`` and ``lemma`` declarations, a proof block only a proof; no word that starts another
   command, declares syntax, runs code, changes an option, a scope or an attribute, or proves
   nothing (``sorry``, ``native_decide``, ...) may appear anywhere, comments included, and
   ``set_option`` only as ``set_option maxHeartbeats N in``.
2. ``check`` compiles the file as two modules: the generated definitions and statements alone
   (``X_statement``, a ``Prop`` per obligation), then the model's text with the generated checks
   ``theorem X_holds : X_statement := X``, importing the first.
3. A fixed Lean program, ``lean/Discovery/Checker.lean``, run apart, reads the compiled modules
   without elaborating anything: it replays every declaration of the second module in the kernel
   (what ``leanchecker`` does, so ``debug.skipKernelTC`` or an environment hack is found), checks
   that each ``X_holds`` has exactly the type ``X_statement`` of the first module, and collects
   its axioms from the declarations' bodies.

A proof is accepted when Lean reports no error and the checker finds every ``X_holds`` with
nothing beyond ``propext``, ``Classical.choice`` and ``Quot.sound``. An accepted file goes into
the lemma library, ``lean/Discovery/Library/`` with ``index.json``, which the third tier feeds
back to later runs (``--recheck-library`` judges the stored proofs again, with no model).

Lean runs in the environment ``lake env`` sets up, read once per process; each check then starts
``lean`` directly, which saves the six or seven seconds Lake takes to load the workspace.

    python -m discovery.prove --problem softmax --program known:online --model sonnet
    python -m discovery.prove --recheck-library
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

from discovery import ir, lean_export, llm
from discovery.lean_export import FRAMEWORK, HELPERS, LEAN_DIR, LeanFile, ident
from discovery.problems import PROBLEMS, Problem

LIBRARY = LEAN_DIR / "Discovery" / "Library"
WORKDIR = Path(tempfile.gettempdir()) / "discovery-prove"
CHECKER = LEAN_DIR / "Discovery" / "Checker.lean"
# the modules a spliced file is compiled as: the generated statements, then the proofs
STATEMENTS, PROOFS = "JudgeStatements", "JudgeProofs"
ELAN_DEFAULT = Path(r"C:\dev\tools\elan\bin\lake.exe")
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
LEAN_TIMEOUT = 900.0

# The model's text is read as raw text: a word below rejects it wherever it stands, in a comment
# or a string too (comments are not stripped: Lean's block comments nest). The words come from
# the command, term and tactic syntax of Lean 4.36 and of the packages Mathlib brings.
# Every command but a theorem, and the modifiers of other declarations.
_COMMANDS = frozenset({
    "def", "abbrev", "instance", "example", "axiom", "opaque", "structure", "class", "inductive",
    "coinductive", "mutual", "noncomputable", "partial", "unsafe", "nonrec", "protected", "public",
    "meta", "deriving", "irreducible_def", "alias", "aux_def", "proof_wanted", "theorem_wanted",
    "def_wanted", "instance_wanted", "initialize", "init_quot", "import", "prelude", "lrat_proof",
    "mk_iff_of_inductive_prop", "add_decl_doc", "library_note", "extend_docs",
    "recommended_spelling", "deprecate", "deprecated_module", "deprecated_syntax", "docs_to_verso",
    "reprove", "recall", "test_extern", "unif_hint", "norm_cast_add_elim",
    "insert_to_additive_translation", "insert_to_dual_translation", "to_additive_name_hint",
    "to_dual_name_hint", "to_dual_for", "to_dual_insert_cast", "to_dual_insert_cast_fun",
    "initialize_simps_projections", "name_poly_vars", "name_power_vars", "suppress_compilation",
    "unsuppress_compilation", "set_library_suggestions", "postprocess_traces", "store_traces_as",
    "show_panel_widgets", "whatsnew", "guard_min_heartbeats", "assert_no_sorry",
    "assert_not_exists", "assert_not_imported", "allow_unused_tactic", "stop_at_first_error",
    "add_aesop_rules", "erase_aesop_rules", "grind_pattern", "grind_propagator", "grind_annotated",
    "make_elab_grind_config", "make_elab_simp_config", "def_eval_config_item",
    "derive_eval_expr_instance_using_meta_eval", "ensure_eval_expr_instance",
    "ensure_eval_term_expr_instances", "ensure_eval_term_instance", "gen_cnstr_fns",
    "command_code_action", "tactic_extension", "elab_stx_quot", "binder_predicate",
    "leading_parser", "trailing_parser",
    # scopes, attributes, options
    "namespace", "section", "end", "open", "export", "variable", "variables", "universe",
    "include", "omit", "local", "scoped", "with_weak_namespace", "end_local_scope", "attribute",
    "unset_option", "sudo", "unlock_limits", "seal", "unseal",
    # syntax, and what runs code while the file is checked
    "syntax", "macro", "macro_rules", "elab", "elab_rules", "notation", "notation3", "infix",
    "infixl", "infixr", "prefix", "postfix", "by_elab", "by_elabq", "include_str", "idbg",
})  # fmt: skip
# what proves nothing, has compiled code or an outside program decide a goal, or does input and
# output (`ofReduceBool` and `reduceBool` are gone from Lean 4.36, refused all the same)
_UNPROVED = frozenset({
    "sorry", "sorryAx", "admit", "native_decide", "native", "ofReduceBool", "ofReduceNat",
    "reduceBool", "reduceNat", "implemented_by", "extern", "trustCompiler", "bv_decide",
    "bv_check", "polyrith", "IO", "EIO", "BaseIO", "println",
})  # fmt: skip
_PREFIXES = ("register_", "declare_", "builtin_", "run_", "unsafe", "simproc", "dsimproc", "cbv_")
_DECLARATIONS = frozenset({"theorem", "lemma", "private"})  # the helper block's own words
# A word exactly as Lean 4 reads an identifier (`Lean.isIdFirst`, `isIdRest`, `isLetterLike`,
# `isSubScriptAlnum`): ASCII letters and `_`, the letter-like ranges (Greek but λ, Π and Σ,
# Coptic, polytonic Greek, the letterlike block, mathematical alphanumerics), then also ASCII
# digits, `'`, `!`, `?` and subscripts. Every other character separates words, as it does for
# Lean. An ASCII-only pattern split `hφdef` into `h` and `def` and refused a correct proof
# (diary, 10:35 UTC); its first fix, Python's `\w`, was too wide: `λdef` read as one word hid a
# `def` that Lean reads after the token `λ` (the review, diary, 12:35 UTC).
_LETTERLIKE = (
    "α-κμ-ω"  # lower Greek but λ
    "Α-ΟΡ΢Τ-Ω"  # upper Greek but Π and Σ
    "ϊ-ϻ"  # Coptic
    "ἀ-῾"  # polytonic Greek
    "℀-⅏"  # letterlike symbols
    "\U0001d49c-\U0001d59f"  # script, double-struck, fraktur
)
_SUBSCRIPTS = "₀-₉ₐ-ₜᵢ-ᵪ"
_WORD = re.compile(rf"[A-Za-z_{_LETTERLIKE}][A-Za-z0-9_'!?{_LETTERLIKE}{_SUBSCRIPTS}]*")
_OPTION = re.compile(r"set_option\s+maxHeartbeats\s+\d+\s+in\b")
# the words the prompt names, among those a proof might plausibly write
_PROMPT_WORDS = (
    "`def`", "`abbrev`", "`instance`", "`example`", "`structure`", "`open`", "`namespace`",
    "`section`", "`end`", "`variable`", "`universe`", "`local`", "`notation`", "`macro`",
    "`syntax`", "`elab`", "`run_tac`", "`by_elab`", "`unsafe`", "`partial`", "`noncomputable`",
)  # fmt: skip
_PROMPT_UNPROVED = ("sorry", "sorryAx", "admit", "native_decide", "decide +native", "extern")

SYSTEM = (
    "You are an expert in Lean 4 and Mathlib. You write complete proofs that compile, and you "
    "answer exactly in the format asked."
)


class LeanError(Exception):
    """Lean or Lake could not be run at all (not a proof that fails to check)."""


class ReplyError(Exception):
    """A reply that does not follow the answer format; the message is for the model."""


# --------------------------------------------------------------------------- running Lean


def lake() -> str:
    found = shutil.which("lake")
    if found:
        return found
    if ELAN_DEFAULT.exists():
        return str(ELAN_DEFAULT)
    raise LeanError("lake not found: put elan's bin directory on PATH")


def _base_env() -> dict[str, str]:
    exe = Path(lake())
    env = dict(os.environ)
    env.setdefault("ELAN_HOME", str(exe.parent.parent))
    env["PATH"] = str(exe.parent) + os.pathsep + env.get("PATH", "")
    return env


@functools.cache
def lean_env() -> dict[str, str]:
    """The environment `lake env` runs commands in, from discovery/lean."""
    proc = subprocess.run(
        [lake(), "env"],
        cwd=LEAN_DIR,
        env=_base_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    if proc.returncode != 0:
        raise LeanError(f"lake env failed: {(proc.stdout + proc.stderr)[-1000:]}")
    env = _base_env()
    for line in proc.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep and key and " " not in key:
            env[key] = value
    if "LEAN" not in env or "LEAN_PATH" not in env:
        raise LeanError("lake env did not report LEAN and LEAN_PATH")
    return env


@functools.cache
def build() -> None:
    """Build the framework (lake build Discovery), once per process."""
    proc = subprocess.run(
        [lake(), "build", "Discovery"],
        cwd=LEAN_DIR,
        env=_base_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=3600,
    )
    if proc.returncode != 0:
        raise LeanError(f"lake build Discovery failed: {(proc.stdout + proc.stderr)[-2000:]}")


@dataclasses.dataclass
class Message:
    line: int
    col: int
    severity: str  # error | warning | info
    text: str


_MESSAGE = re.compile(
    r"^(?P<file>[^\n]*?):(?P<line>\d+):(?P<col>\d+): "
    r"(?P<sev>error|warning|info)(?:\([^)\n]*\))?: ?",
    re.M,
)


# `#print axioms` prints its report with no position, like any message of severity info; a name
# may hold a quote (`merge_block_a'_holds`), so it runs to the first quote the report goes on from
_REPORT = re.compile(
    r"^'(?P<name>[^\n]+?)' (?:depends on axioms: \[(?P<axioms>[^\]]*)\]"
    r"|(?P<none>does not depend on any axioms))",
    re.M,
)


def parse_messages(output: str) -> list[Message]:
    """The positioned messages of Lean's output (the axiom reports taken out)."""
    output = _REPORT.sub("", output)
    starts = list(_MESSAGE.finditer(output))
    out = []
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(output)
        out.append(Message(int(m["line"]), int(m["col"]), m["sev"], output[m.end() : end].strip()))
    return out


def axiom_reports(output: str, namespace: str) -> list[tuple[str, list[str]]]:
    """Every `#print axioms` report, in order: (theorem relative to ``namespace``, axioms)."""
    out = []
    for m in _REPORT.finditer(output):
        name = m["name"].removeprefix(namespace + ".").replace("«", "").replace("»", "")
        axioms = [] if m["none"] else [a.strip() for a in m["axioms"].split(",") if a.strip()]
        out.append((name, axioms))
    return out


def _lean(args: list[str], roots: list[Path], timeout: float) -> tuple[int | None, str, float]:
    """Run ``lean`` with ``roots`` first in its search path: exit code (None on a timeout), output
    and seconds. Lean runs in :data:`LEAN_DIR`, so every path in ``args`` must be absolute."""
    env = dict(lean_env())
    env["LEAN_PATH"] = os.pathsep.join([*(str(Path(r).resolve()) for r in roots), env["LEAN_PATH"]])
    build()
    t0 = time.time()
    try:
        proc = subprocess.run(
            [env["LEAN"], *args],
            cwd=LEAN_DIR,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, "", timeout
    return proc.returncode, proc.stdout + proc.stderr, time.time() - t0


def run_lean(
    path: Path,
    timeout: float = LEAN_TIMEOUT,
    olean: Path | None = None,
    roots: list[Path] | None = None,
) -> tuple[str, float]:
    """Check one file: Lean's output and the seconds it took. With ``olean``, the file is a module
    whose name is its file name, and Lean writes it there when it has no error; ``roots`` go
    first in the search path (where the modules it imports are). Relative paths are taken from
    the current directory, not from the directory Lean runs in."""
    path = Path(path).resolve()
    if olean is None:
        args = [str(path)]
    else:
        args = ["-R", str(path.parent), "-o", str(Path(olean).resolve()), str(path)]
    code, output, seconds = _lean(args, list(roots or []), timeout)
    if code is None:
        return f"{path}:1:0: error: Lean did not finish in {timeout:.0f} s", seconds
    if code != 0 and not any(m.severity == "error" for m in parse_messages(output)):
        # no message Lean placed in the file: say what it printed, for the model and the log
        said = " ".join(output.split())[-800:] or "nothing"
        output += f"\n{path}:1:0: error: Lean exited with {code}, printing: {said}"
    return output, seconds


# --------------------------------------------------------------------------- the model's text


def banned(text: str, helpers: bool = False) -> list[str]:
    """What ``text``, the model's text for a proof block (or for the helper block, with
    ``helpers``), may not contain: anything that is not a proof (or a ``theorem``/``lemma``
    declaration), that proves nothing, or that changes how Lean reads or checks the file. The text
    is read raw, comments and strings included, so a comment between two tokens hides nothing."""
    found = {m.group() for m in re.finditer(r"#[A-Za-z_][\w!?]*", text)}  # #eval, #print, ...
    if "@[" in text:
        found.add("@[")  # an attribute
    allowed = {m.start() for m in _OPTION.finditer(text)}
    for m in re.finditer("set_option", text):
        if m.start() not in allowed:
            found.add("set_option (other than `set_option maxHeartbeats N in`)")
    found.update(m.group() for m in re.finditer(r"[\w'!?]+%", text))  # eval%, ... elaborators
    if re.search(r"\bLean\s*\.", text.replace("«", "").replace("»", "")):
        found.add("Lean.")  # Lean.Elab, Lean.Meta, ...: the elaborator's own code
    for word in _WORD.findall(text):
        bare = word.rstrip("!?'")
        for w in {word, bare}:
            if w in _COMMANDS or w in _UNPROVED or w.startswith(_PREFIXES):
                found.add(w)
            elif w in _DECLARATIONS and not helpers:
                found.add(w)
    return sorted(found)


_FENCE = re.compile(r"^```lean[ \t]+(?P<name>\S+)[ \t]*\r?\n(?P<body>.*?)^```[ \t]*$", re.M | re.S)


def parse_reply(reply: str, names: set[str]) -> tuple[dict[str, str], str | None]:
    """The proof bodies of a reply, by theorem, and its helper lemmas (None when it has none)."""
    blocks: dict[str, str] = {}
    for m in _FENCE.finditer(reply):
        if m["name"] in blocks:
            raise ReplyError(f"two blocks are named `{m['name']}`")
        blocks[m["name"]] = m["body"]
    if not blocks:
        raise ReplyError(
            "no block of the form ```lean NAME ... ``` was found: open each block with "
            "```lean followed by the theorem's name on the same line"
        )
    helpers = blocks.pop(HELPERS, None)
    unknown = sorted(set(blocks) - names)
    if unknown:
        raise ReplyError(
            f"unknown block name(s) {', '.join(unknown)}: the blocks are named after the "
            f"theorems ({', '.join(sorted(names))}) or `{HELPERS}`"
        )
    return blocks, helpers


def framework_statements() -> str:
    """The definitions and theorem statements of the framework, without proofs."""
    out: list[str] = []
    namespaces: list[str] = []
    shown = ""
    decl = re.compile(r"^(@\[[^\]]*\] )?(theorem|lemma|noncomputable def|def|inductive) ")
    for block in re.split(r"\n\s*\n", FRAMEWORK.read_text(encoding="utf-8")):
        lines = block.strip("\n").splitlines()
        for line in lines:
            if line.startswith("namespace "):
                namespaces.append(line.split()[1])
            elif line.startswith("end ") and namespaces and line.split()[1] == namespaces[-1]:
                namespaces.pop()
        start = next((i for i, line in enumerate(lines) if decl.match(line)), None)
        keep = [line for line in lines if line.startswith("variable ")]
        if start is not None:
            keep = lines[:start]
            theorem = re.search(r"\b(theorem|lemma) ", lines[start]) is not None
            for line in lines[start:]:
                if theorem and line.startswith("  |"):
                    break
                if theorem and " :=" in line:
                    keep.append(line[: line.index(" :=")])
                    break
                keep.append(line)
        if not keep:
            continue
        where = ".".join(namespaces)
        if where != shown:
            out += ["", f"-- namespace {where}"]
            shown = where
        out += keep
    return "\n".join(out).strip()


# --------------------------------------------------------------------------- checking


@dataclasses.dataclass
class Check:
    """The judge's verdict on one spliced file."""

    errors: list[Message]  # Lean's errors, at their lines in the spliced file
    axioms: dict[str, list[str]]  # check theorem -> its axioms
    spans: dict[str, tuple[int, int]]  # slot -> its first and last line
    accepted: dict[str, bool]  # obligation -> proved: no error in its slot, allowed axioms only
    composed: bool  # blocked_eq_spec holds with the allowed axioms only
    seconds: float  # Lean in all: the statements' module (when not cached), the proofs, the checker
    checked: bool = False  # the checker read the compiled proofs (they compiled with no error)
    problems: list[str] = dataclasses.field(default_factory=list)  # what the checker rejected
    checker_seconds: float = 0.0
    statements_seconds: float = 0.0  # compiling the statements' module, 0 when it was cached

    @property
    def ok(self) -> bool:
        return (
            self.checked
            and not self.errors
            and not self.problems
            and all(self.accepted.values())
            and self.composed
        )

    def region(self, line: int) -> str:
        """The slot a line is in, or ''."""
        for name, (first, last) in self.spans.items():
            if first <= line <= last:
                return name
        return ""


def _clean(axioms: list[str] | None) -> bool:
    return axioms is not None and set(axioms) <= ALLOWED_AXIOMS


def _framework_key() -> str:
    """What a compiled statements module depends on besides its text: the toolchain, the
    packages, the framework."""
    parts = [lean_env()["LEAN"], FRAMEWORK.read_text(encoding="utf-8")]
    parts += [(LEAN_DIR / f).read_text(encoding="utf-8") for f in ("lake-manifest.json",)]
    return "\n".join(parts)


def statements_module(lf: LeanFile, cache: Path) -> tuple[Path, str, float]:
    """Compile ``lf.statements`` as the module ``STATEMENTS``, once: the directory that holds it
    (under ``cache``, named after its text), Lean's output and the seconds it took (0 when it was
    already there)."""
    key = hashlib.sha256((lf.statements + _framework_key()).encode()).hexdigest()[:16]
    root = Path(cache).resolve() / f"{lf.module}-{key}"
    olean = root / f"{STATEMENTS}.olean"
    if olean.exists():
        return root, "", 0.0
    root.mkdir(parents=True, exist_ok=True)
    source = root / f"{STATEMENTS}.lean"
    source.write_text(lf.statements, encoding="utf-8")
    part = root / f"{STATEMENTS}.{os.getpid()}-{threading.get_ident()}.part"
    output, seconds = run_lean(source, olean=part)
    if part.exists():
        os.replace(part, olean)  # whole, or not at all: other runs may read the cache
    return root, output, seconds


def run_checker(
    lf: LeanFile, roots: list[Path], timeout: float = LEAN_TIMEOUT
) -> tuple[dict, float]:
    """Run ``CHECKER`` on the compiled ``PROOFS`` module of ``roots[0]``: its JSON verdict (with
    ``"error"`` when it could not give one) and the seconds it took."""
    roots = [Path(r).resolve() for r in roots]
    request = roots[0] / "check.json"
    request.write_text(
        json.dumps(
            {
                "statements": STATEMENTS,
                "proofs": PROOFS,
                "namespace": lf.namespace.split("."),
                "obligations": list(lf.checks.values()),
            }
        ),
        encoding="utf-8",
    )
    code, output, seconds = _lean(["--run", str(CHECKER), str(request)], roots, timeout)
    if code is None:
        return {"error": f"the checker did not finish in {timeout:.0f} s"}, seconds
    lines = output.strip().splitlines()
    try:
        verdict = json.loads(lines[-1]) if code == 0 and lines else None
    except json.JSONDecodeError:
        verdict = None
    if not isinstance(verdict, dict) or not isinstance(verdict.get("checks"), dict):
        return {"error": f"the checker failed (exit {code}): {output[-1500:]}"}, seconds
    return verdict, seconds


def _whole(verdict: dict) -> list[str]:
    """What the checker's verdict rejects in the whole module, for the model."""
    if "error" in verdict:
        return [str(verdict["error"])]
    if verdict.get("replay"):
        replay = _clip(str(verdict["replay"]), 1500)
        return [f"the kernel rejects a declaration when the module is checked again: {replay}"]
    return []


def _typed(info: dict) -> bool:
    """The checker found the check theorem, with exactly the type of its generated statement."""
    return all(info.get(k) for k in ("found", "statement", "type"))


def check(lf: LeanFile, proofs: dict[str, str], helpers: str, path: Path) -> Check:
    """Splice ``proofs`` and ``helpers`` into ``lf`` and judge every obligation: no error in its
    slot nor in its check, and, from the checker, its check theorem of the right type with only
    the allowed axioms. ``path`` gets the spliced file; the modules are compiled in the directory
    of the same name, the statements' once, in ``path.parent / "statements"``."""
    path = Path(path).resolve()  # Lean runs in another directory
    text, spans = lf.layout(proofs, helpers)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    statements, stmt_output, stmt_seconds = statements_module(lf, path.parent / "statements")
    root = path.with_suffix("")
    root.mkdir(parents=True, exist_ok=True)
    source, olean = root / f"{PROOFS}.lean", root / f"{PROOFS}.olean"
    module, offset = lf.proofs(text, STATEMENTS)
    source.write_text(module, encoding="utf-8")
    olean.unlink(missing_ok=True)
    output, seconds = run_lean(source, olean=olean, roots=[root, statements])
    errors = [m for m in parse_messages(stmt_output) if m.severity == "error"]
    for m in parse_messages(output):
        if m.severity == "error":
            errors.append(dataclasses.replace(m, line=m.line + offset))
    seconds += stmt_seconds
    # the lines of each check theorem and its #print, after the last slot
    lines = text.splitlines()
    after = max(last for _, last in spans.values())
    check_lines = {
        c: {
            i
            for i in range(after + 1, len(lines) + 1)
            if lines[i - 1].startswith((f"theorem {ident(c)} :", f"#print axioms {ident(c)}"))
        }
        for c in lf.checks
    }
    own = {
        o.name: [
            m
            for m in errors
            if spans[o.name][0] <= m.line <= spans[o.name][1]
            or m.line in check_lines[f"{o.name}_holds"]
        ]
        for o in lf.obligations
    }
    if errors or not olean.exists():
        # No module to read: the `#print axioms` reports of Lean's output (which the model's text
        # could have written) only tell the model which obligations look done.
        reports = dict(axiom_reports(output, lf.namespace))
        axioms = {c: reports[c] for c in lf.checks if c in reports}
        accepted = {
            o.name: not own[o.name] and _clean(axioms.get(f"{o.name}_holds"))
            for o in lf.obligations
        }
        composed = _clean(axioms.get("blocked_eq_spec_holds"))
        return Check(
            errors, axioms, spans, accepted, composed, seconds, statements_seconds=stmt_seconds
        )
    verdict, checker_seconds = run_checker(lf, [root, statements])
    problems = _whole(verdict)
    info = {c: (verdict.get("checks") or {}).get(o) or {} for c, o in lf.checks.items()}
    axioms = {c: [str(a) for a in i.get("axioms", [])] for c, i in info.items() if i}
    good = {c: not problems and _typed(i) and _clean(axioms.get(c)) for c, i in info.items()}
    for c, o in lf.checks.items():
        if not problems and not _typed(info[c]):
            problems.append(f"`{c}` is not a theorem of the generated statement `{o}_statement`")
    return Check(
        errors,
        axioms,
        spans,
        {o.name: good[f"{o.name}_holds"] for o in lf.obligations},
        good["blocked_eq_spec_holds"],
        seconds + checker_seconds,
        checked=True,
        problems=problems,
        checker_seconds=checker_seconds,
        statements_seconds=stmt_seconds,
    )


# --------------------------------------------------------------------------- the prompt

_FACTS = """\
- Unfold a definition with `simp only [merge_a, block_a]` (or `rw [block_a]`). The state \
structures have `@[ext]`: `ext` splits an equation between states into one goal per field, \
such as `(merge_a (block_a A) (block_a B)).m = (block_a (A ++ B)).m`, which \
`simp only [merge_a, block_a]` then reduces.
- `output` and `Spec.output` bind every pass with `let`; `simp only [output, Spec.output, \
block_a, Spec.block_a]` (with every block function involved) unfolds and substitutes them. The \
named outputs are a structure with `@[ext]` too.
- `lmax` and `lmin` are the largest and smallest element of a nonempty list: use the \
framework's lemmas (`lmax_append`, `lmax_map_append`, `le_lmax`, ...), not their definition.
- Sums over `A ++ B`: `List.map_append`, `List.sum_append`.
- Constants out of a sum: `List.sum_map_mul_left : (l.map fun b => r * f b).sum = \
r * (l.map f).sum`, `List.sum_map_mul_right : (l.map fun b => f b * r).sum = \
(l.map f).sum * r`, `List.smul_sum : r • l.sum = (l.map fun x => r • x).sum`, \
`List.sum_map_add : (l.map fun i => f i + g i).sum = (l.map f).sum + (l.map g).sum`.
- An identity between sums whose terms are polynomials in the element: prove a helper lemma by \
induction on the list, `induction A with | nil => simp | cons y ys ih => simp only \
[List.map_cons, List.sum_cons, ih]; ring`, then use it with the summands' constants.
- `(A.map fun _ => (1 : ℝ)).sum = A.length` by `simp`. A nonempty list has a positive length: \
`List.length_pos_iff.mpr hA : 0 < A.length`, and `(0 : ℝ) < A.length` from it by \
`exact_mod_cast`.
- `Real.exp_add : Real.exp (x + y) = Real.exp x * Real.exp y`, `Real.exp_sub`, \
`Real.exp_pos`, `Real.exp_ne_zero`, `Real.exp_zero`; `max_comm`, `le_max_left`, \
`le_max_right`, `max_def`.
- Vectors (`V` is any real vector space): `smul_smul : a • b • v = (a * b) • v`, `add_smul`, \
`smul_add`, `mul_smul`, and the tactic `module` for goals that are linear in vectors.
- Divisions: `field_simp` with the nonzero denominators as hypotheses, then `ring`; \
`linear_combination` for an equation that follows from others."""


def base_prompt(lf: LeanFile, problem: Problem) -> str:
    duties = "\n".join(f"- `{o.name}`: {o.meaning}." for o in lf.obligations)
    blocks = "\n".join(f"```lean {o.name}\n...\n```" for o in lf.obligations)
    return f"""Prove the theorems of the Lean 4 file below (Lean v4.36.0, Mathlib v4.36.0).

The file was generated from a program that computes, block by block as a GPU kernel does, this \
operator: {problem.description}

The program is correct when these theorems hold; their proofs are `sorry` and are yours to write:
{duties}

`blocked_eq_spec` (the program computes the operator over any blocking and any merge tree) is \
already proved from them, and the `_holds` theorems at the end check every theorem against its \
statement, fixed before your text. Once the file compiles, a checker reads it apart from your \
text: the kernel checks every declaration again, and each `_holds` theorem must have exactly its \
generated statement as type and use no axiom but `propext`, `Classical.choice` and `Quot.sound`.

You write proof bodies only: you cannot change a definition, a statement or an import. You may \
add helper lemmas; they are placed at the line `-- helper lemmas`, after every definition and \
before the first theorem, so they can use the definitions.

## The file

```lean
{lf.text}```

## The framework, imported (`Discovery/Blocked.lean`)

```lean
{framework_statements()}
```

## Facts that help here

{_FACTS}

## Answer

One fenced block per theorem, opened with ```lean and the theorem's name on the same line, \
holding the tactics that replace `sorry` (what follows `:= by`, without `by`):

{blocks}

and, if you need helper lemmas, one more block named `{HELPERS}` with complete declarations \
(`theorem my_lemma ... := by ...`). Nothing else is read.

Rules, checked on the raw text before Lean runs (an answer that breaks one is sent back):
- the `{HELPERS}` block holds complete `theorem` or `lemma` declarations (`private` allowed) and \
nothing else; a theorem's block holds the tactics of its proof and nothing else;
- no attribute (`@[...]`, `attribute`), no `#` command, no other command or declaration \
({", ".join(_PROMPT_WORDS)}, ...), no `name%` elaborator, no name under `Lean.` and no `IO`;
- no {", ".join(f"`{w}`" for w in _PROMPT_UNPROVED)};
- `set_option` only as `set_option maxHeartbeats 400000 in` (any number) before a tactic or a \
declaration;
- write no comments: these words are refused in comments too.
"""


def _numbered(text: str) -> str:
    return "\n".join(f"{i:4d} | {line}" for i, line in enumerate(text.splitlines(), 1))


def _clip(text: str, n: int = 1500) -> str:
    return text if len(text) <= n else text[:n] + f"\n... ({len(text) - n} more characters)"


def error_feedback(round_: int, lf: LeanFile, text: str, result: Check) -> str:
    done = [o.name for o in lf.obligations if result.accepted.get(o.name)]
    todo = [o.name for o in lf.obligations if not result.accepted.get(o.name)]
    lines = [
        f"## Your previous answer (round {round_})",
        "",
        "Lean checked the file with your text spliced in."
        + (f" Proved: {', '.join(done)}." if done else "")
        + (f" Not proved: {', '.join(todo)}." if todo else ""),
        "",
        "The file as checked, with its line numbers:",
        "",
        "```",
        _numbered(text),
        "```",
        "",
    ]
    if result.errors:
        lines += ["Lean's errors:", ""]
    for m in result.errors[:12]:
        where = result.region(m.line)
        place = (
            "in your helper lemmas"
            if where == HELPERS
            else f"in your proof of `{where}`"
            if where
            else "outside your text"
        )
        lines += [f"line {m.line}, column {m.col}, {place}:", _clip(m.text), ""]
    if len(result.errors) > 12:
        lines += [f"... and {len(result.errors) - 12} more errors", ""]
    if result.problems:
        lines += [
            "The file compiles, but the checker, which reads it apart from your text, says:",
            "",
        ]
        lines += [f"- {_clip(p)}" for p in result.problems] + [""]
    for name, axioms in result.axioms.items():
        extra = sorted(set(axioms) - ALLOWED_AXIOMS)
        if extra:
            lines += [f"`{name}` depends on {', '.join(extra)}: prove it without them.", ""]
    lines += [
        "Answer again in the same format. A block you leave out keeps its previous text; send "
        f"every block you change, and `{HELPERS}` in full if you change it.",
    ]
    return "\n".join(lines)


def text_feedback(round_: int, problem_text: str) -> str:
    return (
        f"## Your previous answer (round {round_})\n\nIt was not checked: {problem_text}\n\n"
        "Answer again in the same format."
    )


# --------------------------------------------------------------------------- the prover


@dataclasses.dataclass
class ProofResult:
    problem: str
    program: str
    digest: str
    module: str = ""
    proved: bool = False
    rounds: int = 0
    seconds: float = 0.0
    lean_seconds: float = 0.0
    usage: dict = dataclasses.field(default_factory=dict)
    statements: dict[str, str] = dataclasses.field(default_factory=dict)  # theorem -> statement
    axioms: dict[str, list[str]] = dataclasses.field(default_factory=dict)  # from the last check
    text: str = ""  # the last file checked: the proved one, on success
    errors: list[str] = dataclasses.field(default_factory=list)  # every error of every round
    attempts: list[dict] = dataclasses.field(default_factory=list)  # per round: what, how long
    library: str = ""  # the library file written
    message: str = ""

    def summary(self) -> dict:
        return {
            "problem": self.problem,
            "program": self.program,
            "proved": self.proved,
            "rounds": self.rounds,
            "seconds": round(self.seconds, 1),
            "lean_seconds": round(self.lean_seconds, 1),
            "usage": self.usage,
            "library": self.library,
            "message": self.message,
            "last_error": self.errors[-1][:2000] if self.errors and not self.proved else "",
        }


def library_module(problem: Problem, prog: ir.Program, library: Path) -> str:
    """The program's module name in the library: its own, or with its digest when another
    program already holds that name."""
    name = lean_export.module_name(problem, prog)
    for entry in _index(library):
        if entry.get("module") == f"Discovery.Library.{name}" and entry.get("digest") != (
            prog.digest()
        ):
            return name + prog.digest()[:6]
    return name


def _index(library: Path) -> list[dict]:
    path = library / "index.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def in_library(prog: ir.Program, spec: str, library: Path) -> dict | None:
    """The library's entry for ``prog`` against the specification named ``spec``: the same
    program up to its name (the same digest), proved once already."""
    digest = prog.digest()
    for entry in _index(library):
        if entry.get("digest") == digest and entry.get("spec") == spec:
            return entry
    return None


def _helper_statements(helpers: str) -> list[str]:
    """The statements of the helper lemmas, for the library's index."""
    out = []
    for m in re.finditer(r"^(?:theorem|lemma) .*?(?=:=)", helpers, re.M | re.S):
        out.append(re.sub(r"\s+", " ", m.group()).strip())
    return out


def claim(prog: ir.Program, spec: str) -> str:
    """What a library entry proves, in the IR's terms, for the proposer of a later run."""
    merges = [
        f"in pass {p.name}, merging the summaries "
        + " ".join(f"({f} {ir.unparse_expr(e)})" for f, e in p.block)
        + " of two nonempty blocks with "
        + " ".join(f"({f} {ir.unparse_expr(e)})" for f, e in p.merge)
        + " gives the summary of the two blocks together"
        for p in prog.passes
    ]
    return (
        f"over the reals, for any blocking and any merge tree, {prog.name} computes what {spec} "
        f"computes: {'; '.join(merges)}"
    )


def add_to_library(
    lf: LeanFile,
    prog: ir.Program,
    text: str,
    helpers: str,
    result: ProofResult,
    library: Path,
    model: str,
) -> Path:
    library.mkdir(parents=True, exist_ok=True)
    path = library / f"{lf.module}.lean"
    path.write_text(text, encoding="utf-8")
    # one entry per module and per program: a proof of a program the library holds under another
    # name replaces it (prove() asks the model only when the stored proof no longer passes)
    entries = []
    for e in _index(library):
        if e.get("module") == lf.namespace:
            continue
        if e.get("digest") == lf.digest and e.get("spec") == lf.spec:
            if e.get("file") and e["file"] != path.name:
                (library / str(e["file"])).unlink(missing_ok=True)
            continue
        entries.append(e)
    entries.append(
        {
            "name": prog.name,
            "problem": result.problem,
            "statement": claim(prog, lf.spec),
            "module": lf.namespace,
            "file": path.name,
            "program": prog.text(),
            "digest": lf.digest,
            "spec": lf.spec,
            "statements": result.statements,
            "helpers": _helper_statements(helpers),
            "axioms": result.axioms,
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "model": model,
            "rounds": result.rounds,
        }
    )
    (library / "index.json").write_text(
        json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def prove(
    prog: ir.Program,
    problem: Problem,
    model,
    rounds: int = 4,
    workdir: Path | str | None = None,
    spec: ir.Program | None = None,
    library: Path | None = LIBRARY,
) -> ProofResult:
    """Ask ``model`` for the proofs of ``prog``'s obligations, ``rounds`` times at most. A program
    the library already holds (the same digest, against the same specification) is not sent to
    the model: its stored proof is judged again, and only if it fails is the model asked."""
    t0 = time.time()
    first_call = len(getattr(model, "calls", []))
    work = Path(workdir) if workdir is not None else WORKDIR
    result = ProofResult(problem.name, prog.name, "")
    try:
        result.digest = prog.digest()
        module = library_module(problem, prog, library) if library is not None else None
        lf = lean_export.export(prog, problem, spec, module=module)
    except (lean_export.ExportError, ir.IRError, ArithmeticError, ValueError) as exc:
        # a program the IR or the Lean reading cannot write (a non-finite number, ...)
        result.message = f"cannot be stated in Lean: {type(exc).__name__}: {exc}"
        return _finish(result, model, first_call, t0)
    result.module = lf.module
    result.statements = {o.name: o.statement for o in lf.obligations}
    result.statements["blocked_eq_spec"] = lf.composed
    result.text = lf.text
    stored = in_library(prog, lf.spec, library) if library is not None else None
    if stored is not None and library is not None:
        again = recheck(stored, library, work)
        if again["ok"]:
            result.proved = True
            result.module = str(stored["module"]).removeprefix("Discovery.Library.")
            result.library = str(library / stored["file"])
            result.lean_seconds = float(again.get("seconds", 0.0))
            result.axioms = dict(again.get("axioms", {}))
            result.message = (
                f"already in the library as {stored['name']} ({stored['file']}); its stored proof "
                "was judged again, with no model call"
            )
            return _finish(result, model, first_call, t0)
        result.errors.append(f"the library's proof of this program fails now: {again['message']}")
    # the statements' module compiles while the model writes its first answer
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    early = pool.submit(statements_module, lf, work / "statements")
    try:
        return _rounds(
            prog, problem, model, rounds, work, library, lf, result, early, first_call, t0
        )
    finally:
        pool.shutdown(wait=True)


def _rounds(
    prog: ir.Program,
    problem: Problem,
    model,
    rounds: int,
    work: Path,
    library: Path | None,
    lf: LeanFile,
    result: ProofResult,
    early: concurrent.futures.Future | None,
    first_call: int,
    t0: float,
) -> ProofResult:
    """The rounds of ``prove``: ask, read, judge, and tell the model what failed."""
    names = {o.name for o in lf.obligations}
    base = base_prompt(lf, problem)
    proofs: dict[str, str] = {}
    helpers = ""
    feedback = ""
    for k in range(1, rounds + 1):
        result.rounds = k
        attempt: dict = {"round": k}
        result.attempts.append(attempt)
        prompt = base + ("\n" + feedback if feedback else "")
        calls = len(getattr(model, "calls", []))
        try:
            reply = model(prompt, system=SYSTEM)
        except llm.LLMError as exc:
            result.errors.append(f"round {k}: the model failed: {exc}")
            attempt["status"] = "model failed"
            continue
        finally:
            attempt["llm_seconds"] = round(
                sum(c.seconds for c in getattr(model, "calls", [])[calls:]), 1
            )
        try:
            new, new_helpers = parse_reply(reply, names)
            missing = sorted(names - set(proofs) - set(new))
            if missing:
                raise ReplyError(f"no block for {', '.join(missing)}")
        except ReplyError as exc:
            result.errors.append(f"round {k}: unreadable answer: {exc}")
            attempt["status"] = "unreadable"
            feedback = text_feedback(k, f"the answer could not be read: {exc}.")
            continue
        proofs = {**proofs, **new}
        helpers = helpers if new_helpers is None else new_helpers
        found = {name: banned(body) for name, body in proofs.items()}
        found[HELPERS] = banned(helpers, helpers=True)
        if any(found.values()):
            what = "; ".join(f"`{n}` contains {', '.join(w)}" for n, w in found.items() if w)
            result.errors.append(f"round {k}: rejected before Lean: {what}")
            attempt["status"] = "banned"
            feedback = text_feedback(
                k,
                f"{what}, which the rules do not allow. Write only proofs (and helper theorems), "
                "with no comments, and prove every statement in full.",
            )
            continue
        try:
            text = lf.splice(proofs, helpers)
        except ValueError as exc:
            result.errors.append(f"round {k}: {exc}")
            attempt["status"] = "unreadable"
            feedback = text_feedback(k, f"{exc}.")
            continue
        # the statements' module, compiled meanwhile (check finds it done), counted once
        spent = early.result()[2] if early is not None else 0.0
        early = None
        verdict = check(lf, proofs, helpers, work / f"{lf.module}_round{k}.lean")
        verdict.seconds += spent
        verdict.statements_seconds += spent
        result.lean_seconds += verdict.seconds
        result.text = text
        result.axioms = {lf.checks[c]: a for c, a in verdict.axioms.items() if c in lf.checks}
        attempt.update(
            lean_seconds=round(verdict.seconds, 1),
            checker_seconds=round(verdict.checker_seconds, 1),
            errors=len(verdict.errors),
            proved=sorted(n for n, ok in verdict.accepted.items() if ok),
        )
        if verdict.ok:
            attempt["status"] = "proved"
            result.proved = True
            result.message = f"proved in {k} round(s)"
            _finish(result, model, first_call, t0)
            if library is not None:
                path = add_to_library(lf, prog, text, helpers, result, library, _model_name(model))
                result.library = str(path)
            return result
        attempt["status"] = "lean errors" if verdict.errors else "rejected by the checker"
        for m in verdict.errors:
            where = verdict.region(m.line) or "generated text"
            result.errors.append(f"round {k}, {where}, line {m.line}: {_clip(m.text, 3000)}")
        for p in verdict.problems:
            result.errors.append(f"round {k}, the checker: {_clip(p, 3000)}")
        if not verdict.errors and not verdict.problems:
            result.errors.append(f"round {k}: axioms not allowed: {result.axioms}")
        feedback = error_feedback(k, lf, text, verdict)
    result.message = f"not proved in {rounds} round(s)"
    return _finish(result, model, first_call, t0)


def _model_name(model) -> str:
    return str(getattr(model, "name", type(model).__name__))


def _finish(result: ProofResult, model, first_call: int, t0: float) -> ProofResult:
    calls = getattr(model, "calls", [])[first_call:]
    result.usage = llm.usage(types.SimpleNamespace(calls=calls))
    result.seconds = time.time() - t0
    return result


def recheck(entry: dict, library: Path = LIBRARY, workdir: Path | str | None = None) -> dict:
    """Judge the stored proofs of a library entry again, with no model: the program is exported
    again, its stored file must be that export with the slots filled, and the filled slots go
    through ``banned`` and ``check`` as a model's answer would."""
    work = Path(workdir) if workdir is not None else WORKDIR
    out: dict = {"module": entry.get("module", ""), "ok": False}
    try:
        problem = PROBLEMS[entry["problem"]]
        prog = problem.parse(entry["program"])
        named = {p.name: p for p in map(problem.parse, [problem.seed, *problem.known.values()])}
        if entry["spec"] not in named:
            raise ValueError(f"no program of {problem.name} is named {entry['spec']}")
        spec = named[entry["spec"]]
        module = entry["module"].removeprefix("Discovery.Library.")
        lf = lean_export.export(prog, problem, spec, module=module)
        slots = lf.slots((library / entry["file"]).read_text(encoding="utf-8"))
    except (KeyError, OSError, ValueError, ir.IRError, lean_export.ExportError) as exc:
        return {**out, "message": f"cannot be read: {type(exc).__name__}: {exc}"}
    if lf.digest != entry.get("digest") or slots is None:
        return {**out, "message": "the stored file is not the export of its program"}
    helpers = slots.pop(HELPERS)
    found = {name: banned(body) for name, body in slots.items()}
    found[HELPERS] = banned(helpers, helpers=True)
    if any(found.values()):
        return {**out, "message": f"rejected before Lean: {found}"}
    verdict = check(lf, slots, helpers, work / f"{lf.module}_recheck.lean")
    return {
        **out,
        "ok": verdict.ok,
        "seconds": round(verdict.seconds, 1),
        "statements_seconds": round(verdict.statements_seconds, 1),
        "checker_seconds": round(verdict.checker_seconds, 1),
        "axioms": verdict.axioms,
        "message": "; ".join(
            [*(f"line {m.line}: {m.text}" for m in verdict.errors), *verdict.problems]
        ),
    }


# --------------------------------------------------------------------------- command line


def _program(problem: Problem, which: str) -> ir.Program:
    if which == "seed":
        return problem.parse(problem.seed)
    if which.startswith("known:"):
        return problem.parse(problem.known[which.removeprefix("known:")])
    return problem.parse(Path(which).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prove a candidate correct in Lean with an LLM.")
    ap.add_argument("--problem", choices=sorted(PROBLEMS), help="required unless --recheck-library")
    ap.add_argument("--program", default="seed", help="seed, known:NAME, or a file")
    ap.add_argument("--spec", default="seed", help="the specification: seed, known:NAME or a file")
    ap.add_argument("--model", default="sonnet", help="sonnet, opus, or a spec like openai:M@URL")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--workdir", default=str(WORKDIR))
    ap.add_argument("--no-library", action="store_true", help="do not add a proof to the library")
    ap.add_argument(
        "--recheck-library",
        action="store_true",
        help="judge every stored proof of the library again, with no model",
    )
    args = ap.parse_args(argv)
    if args.recheck_library:
        results = [recheck(e, LIBRARY, args.workdir) for e in _index(LIBRARY)]
        print(json.dumps(results, indent=2))
        return 0 if all(r["ok"] for r in results) else 1
    if args.problem is None:
        ap.error("--problem is required")
    problem = PROBLEMS[args.problem]
    try:
        prog = _program(problem, args.program)
        spec = _program(problem, args.spec)
    except (OSError, KeyError, ir.IRError) as exc:
        print(json.dumps({"proved": False, "message": f"cannot read the program: {exc}"}))
        return 2
    model = llm.from_spec(args.model) if ":" in args.model else llm.ClaudeCLI(args.model)
    result = prove(
        prog,
        problem,
        model,
        rounds=args.rounds,
        workdir=args.workdir,
        spec=spec,
        library=None if args.no_library else LIBRARY,
    )
    record = Path(args.workdir) / f"{result.module or prog.name}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(dataclasses.asdict(result), indent=2), encoding="utf-8")
    print(json.dumps({**result.summary(), "record": str(record)}, indent=2))
    return 0 if result.proved else 1


if __name__ == "__main__":
    sys.exit(main())
