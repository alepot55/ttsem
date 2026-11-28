"""Certified floating-point bounds: the model proposes ``B``, the adversary attacks, Lean proves.

For one program of the proof library:

1. the model reads the program, the rounding model and the statement's form, and proposes ``B``
   (:mod:`discovery.bound`); nothing in its prompt states a bound, an invariant or a metric
   other than the statement itself;
2. the adversary looks for a run that breaks ``B``: floating-point runs of the program
   (:mod:`discovery.fsem`, numpy float32) on families of rows, climbing on ``error / B`` over
   blockings of 64 and 256 and left, balanced and random merge trees. A ``B`` it breaks goes back
   to the model with the run, and Lean is not called;
3. a ``B`` that survives goes to the prover: the model writes the Lean proof of the generated
   statement, judged as the exact proofs are (deny list on the raw text, statements compiled on
   their own, the checker's kernel replay, type and axioms), three rounds with ``sonnet`` and then
   three with a stronger model.

The tightness of a proved ``B`` is ``B / error`` on the run where ``error / B`` is largest.

    python -m discovery.fpbound --program SoftmaxOnline --out discovery/results/m3
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from discovery import adversary, bound, fp_export, fptable, fsem, ir, llm, problems, prove

U32 = 2.0**-24
# the rounding of numpy's float32 on this machine (discovery/results/m2/libm.json), rounded up
ENV32 = {"u": U32, "ue": 4 * U32, "ul": 5 * U32, "eta": 2.0**-150, "etae": 2.0**-148}
BLOCKS = (64, 256)
ORDERS = ("left", "balanced", "random")
STATEMENTS = "FPStatements"
PROOFS = "FPProofs"

# --------------------------------------------------------------------------- measuring


def measure(problem: str, prog: ir.Program, el64: dict, block: int, order: str) -> tuple:
    """The certified quantity's error on one float32 run, and the atoms of the run.

    softmax: |log Ẑ − log Σ exp x|; attention: max over components of |ô − o| / Σ p|v|;
    variance: |v̂ − var|."""
    F = fsem.FORMATS["float32"]
    el = {k: np.asarray(F.value(v), dtype=np.float64) for k, v in el64.items()}
    r = fsem.run(prog, el, "float32", block=block, order=order)
    xs = el["s" if problem == "attention" else "x"][:, 0]
    lo, hi = float(xs.min()), float(xs.max())
    env = dict(ENV32)
    env.update(
        {
            "D": float(math.ceil(math.log2(min(block, xs.size)))) if xs.size > 1 else 0.0,
            "d": float(r.depth),
            "n": float(xs.size),
            "R": hi - lo,
            "M": max(abs(lo), abs(hi)),
        }
    )
    if problem == "softmax":
        m = float(xs.max())
        zhat = fsem._normalizer(prog, r, m)
        D_ref = math.fsum(np.exp(xs - m).tolist())
        ok = zhat > 0 and math.isfinite(zhat)
        err = abs(math.log(zhat) - math.log(D_ref)) if ok else math.inf
    elif problem == "attention":
        ref = fsem.reference("attention", el, exact=False)
        o = np.asarray(r.outputs["o"], dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(o)):
            return math.inf, env
        env["V"] = float(np.max(np.abs(el["v"])))
        # the statement is about one component: one run per component, with its own scale
        return [
            (abs(float(o[k]) - float(ref["o"][k])), {**env, "S": float(ref["scale"][k])})
            for k in range(o.size)
        ]
    else:
        ref = fsem.reference("variance", el, exact=False)
        v = float(r.outputs["var"])
        err = abs(v - ref["var"]) if math.isfinite(v) else math.inf
        env.update({"S2": ref["var"], "Q": math.fsum((xs * xs).tolist()) / xs.size})
    return err, env


def log_normalizer_error(prog: ir.Program, x: np.ndarray, block: int, order: str) -> tuple:
    """|log Ẑ - log Z| of one float32 run of a softmax program, and the atoms of the run."""
    return measure("softmax", prog, {"x": np.asarray(x)[:, None]}, block, order)


def ratio(problem: str, prog: ir.Program, b: bound.Bound, el: dict) -> tuple[float, dict]:
    """The largest error / B over the blockings and merge orders, and that run."""
    worst, info = -1.0, {}
    for block in BLOCKS:
        for order in ORDERS:
            got = measure(problem, prog, el, block, order)
            for err, env in got if isinstance(got, list) else [got]:
                B = b.value(env)
                q = err / B if B > 0 else math.inf
                if q > worst:
                    worst, info = q, {"block": block, "order": order, "error": err, "B": B, **env}
    return worst, info


def attack(
    prog: ir.Program,
    b: bound.Bound,
    steps: int = 150,
    seeds: tuple[int, ...] = (0, 1),
    problem: str = "softmax",
) -> dict:
    """Climb on error / B from every family of the problem; the worst run found."""
    best: dict = {"ratio": -1.0}
    F = fsem.FORMATS["float32"]
    for fam in adversary.families(problem):
        for seed in seeds:
            rng = np.random.default_rng(7000 + seed)
            el = adversary.generate(problem, fam, rng, 4096, 256)
            el = {k: np.asarray(F.value(v), dtype=np.float64) for k, v in el.items()}
            cur, info = ratio(problem, prog, b, el)
            for _ in range(steps):
                cand = adversary.mutate(problem, el, F, rng, 256)
                q, inf2 = ratio(problem, prog, b, cand)
                if math.isfinite(q) and q > cur:
                    el, cur, info = cand, q, inf2
            if cur > best["ratio"]:
                best = {"ratio": cur, "family": fam, "seed": seed, **info, "x": el}
    return best


# --------------------------------------------------------------------------- prompts

SYSTEM = (
    "You are a careful numerical analyst and an expert in Lean 4 and Mathlib. You answer in the "
    "exact format asked for."
)

ROUNDING = """\
The rounding model (Lean, `Discovery.Rounding`, every field a parameter):

  u  : relative error of + - * / sqrt          η  : absolute error of * / with a subnormal result
  ue : relative error of exp                   ηe : absolute error of exp with a subnormal result
  ul : relative error of log

  F.add x v : ∃ δ, |δ| ≤ F.u ∧ v = x * (1 + δ)                       -- + and -, x the exact result
  F.mul x v : ∃ δ e, |δ| ≤ F.u ∧ |e| ≤ F.η ∧ v = x * (1 + δ) + e ∧ 0 ≤ x * v   -- * and /
  F.exp x v : ∃ δ e, |δ| ≤ F.ue ∧ |e| ≤ F.ηe ∧ v = Real.exp x * (1 + δ) + e ∧ 0 ≤ v
  F.log x v : ∃ δ, |δ| ≤ F.ul ∧ v = Real.log x * (1 + δ)
  F.sqrt x v : ∃ δ, |δ| ≤ F.u ∧ v = Real.sqrt x * (1 + δ)
  max, min, neg, abs: exact.

Every operation of the program is one such rounding with its own error variables; a block's sum
adds its terms along any binary tree of depth at most D (`Discovery.fsum`); the block summaries
merge along any merge tree `t` (`Discovery.Tree.FEval`). Overflow is outside the model."""


def bound_prompt(fp: fp_export.FPFile, problem: problems.Problem, history: list[str]) -> str:
    docs = {**_ATOM_DOC, **_EXTRA_DOC.get(problem.name, {})}
    grammar = "\n".join(f"  {a:5s} {doc}" for a, doc in docs.items())
    template = fp.statement_template.replace("{B}", "B")
    past = ""
    if history:
        past = "\n\nYour earlier proposals and what happened to them:\n" + "\n".join(history)
    defs = fp.definitions
    defs = defs[defs.index("/-- Pass") : defs.index("/-- The rounding is small")].strip()
    return f"""{problem.description}

The program below computes it by blocked folds: each block of a row is summarised (block), and
adjacent summaries are combined (merge) along a merge tree; the output reads the final summary.

```
{fp.program}
```

{ROUNDING}

The Lean definitions generated from the program:

```lean
{defs}
```

The statement to be proved, with B to be chosen:

```lean
{template}
```

where `small F` is: {fp_export.SMALL}.

Give B, an upper bound that makes the statement true for every such X, lo, hi, D, merge tree t,
and every admissible choice of the rounding errors. B is a sum of products of nonnegative numbers
and these atoms, with parentheses and integer powers 1 to 4 (no subtraction; divide only by a
number):

{grammar}

The bound will be tested against floating-point runs and then proved in Lean. A false bound is
rejected; among true bounds a smaller one is better.{past}

Reply with the bound alone on the last line, in the form
B = <expression>"""


_ATOM_DOC = {
    "u": "relative error of + - * / sqrt (F.u)",
    "ue": "relative error of exp (F.ue)",
    "ul": "relative error of log (F.ul)",
    "eta": "absolute error of * / below the normal range (F.η)",
    "etae": "absolute error of exp below the normal range (F.ηe)",
    "D": "the bound on the depth of the in-block summation trees",
    "d": "the depth of the merge tree t (t.depth)",
    "n": "the number of elements of the row (X.length)",
    "R": "hi - lo",
    "M": "max |lo| |hi|",
}

_EXTRA_DOC = {
    "variance": {
        "S2": "the exact variance of the row (vr X)",
        "Q": "the exact mean of the squares of the row (msq X)",
    },
    "attention": {
        "S": "the scale Σ exp(s)·|v| / Σ exp(s) of the row (attnScale X)",
        "V": "the bound Vmax on |v|",
    },
    "policy": {
        "tau": "the program's parameter tau",
        "rd": "the right depth of the merge tree t (rdepth t)",
    },
}

# what the statement bounds, as the counterexamples name it
QUANTITY = {
    "softmax": "|log(znorm st) - log Σ exp x|",
    "attention": "|w - attn X|",
    "variance": "|w - vr X|",
    "policy": "|log(znorm st) - log Σ exp x|",
}


def parse_bound_reply(reply: str, problem: str = "") -> bound.Bound:
    lines = re.findall(r"^\s*B\s*=\s*(.+?)\s*$", reply, re.M)
    if not lines:
        raise bound.BoundError("no line `B = <expression>` in the reply")
    return bound.parse(lines[-1], problem)


def framework_text() -> str:
    """The statements of the framework lemmas the proof may use (no proofs).

    Left out: the modules' prose, and the framework's demonstration, the classical bound for a
    sum of positive numbers (`sum_tree_bound`, `sum_tree_error`): a textbook bound is not a tool,
    and no bound written by us or taken from a paper reaches the model."""
    out = []
    for path in (prove.FRAMEWORK, prove.LEAN_DIR / "Discovery" / "Approx.lean"):
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"/-!\n.*?-/\n", "", text, flags=re.S)  # the module's prose
        text = re.sub(
            r"/-! ## The sum of positive numbers.*?(?=/-! ## A block's sum)", "", text, flags=re.S
        )
        out.append(_signatures(text))
    return "\n\n".join(out)


def _signatures(text: str) -> str:
    """Definitions in full, theorems up to `:=`: what a proof can use."""
    keep, out, skip = [], [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(("theorem ", "lemma ")):
            skip = False
            keep = [line]
            if " := " in line or line.rstrip().endswith(":="):
                out.append(line.split(" :=")[0])
                keep = []
                skip = True
            continue
        if keep:
            keep.append(line)
            if ":=" in line:
                out.append("\n".join(keep).split(" :=")[0])
                keep = []
                skip = True
            continue
        if skip:
            if s and not line.startswith((" ", "\t")):
                skip = False
            else:
                continue
        out.append(line)
    return "\n".join(out)


def library_text(library: str) -> str:
    """The exact semantics of the program, as the library module states it (proofs elided)."""
    path = prove.LIBRARY / f"{library}.lean"
    text = path.read_text(encoding="utf-8")
    head = text.split("/-! ## The proofs -/")[0]
    merge = re.search(r"theorem merge_block_\w+ .*?:=", text, re.S)
    return head + ("\n" + merge.group(0)[:-3] + " := ...\n" if merge else "")


def proof_prompt(fp: fp_export.FPFile, b: bound.Bound, with_library: bool = True) -> str:
    library = ""
    if fp.library and with_library:
        library = f"""
The module `Discovery.Library.{fp.library}` holds the program's exact semantics (proved earlier):

```lean
{library_text(fp.library)}
```
"""
    return f"""Prove a theorem about a program's floating-point semantics in Lean 4 (Mathlib, Lean
v4.36.0).
{library}
The framework (`Discovery.Blocked`, `Discovery.Approx`), statements only:

```lean
{framework_text()}
```

The generated statements module (fixed; your text goes in a module that imports it):

```lean
{fp.statements(b)}
```

Write the proof of

```lean
theorem fp_bound : {fp.statement_template.replace("{B}", b.lean()).strip()} := by
  ...
```

Rules: your text is read raw before Lean sees it. Helper lemmas are allowed only as `theorem` or
`lemma` declarations (no `def`, `abbrev`, `instance`, attributes, `#` commands, `set_option`
other than `set_option maxHeartbeats N in` before a declaration); no `sorry`, `admit`,
`native_decide`. Only the axioms propext, Classical.choice and Quot.sound may be used.

Reply with exactly two fenced blocks:

```lean helpers
-- helper lemmas, or a comment if none
```

```lean fp_bound
-- the tactic proof of fp_bound (the text after `:= by`)
```"""


# --------------------------------------------------------------------------- the judge

# A proof's tactics can search without bound: one model proof took 15 GB of memory on this
# 32 GB machine (diary, 10:15 UTC). Lean's own limit stops it with an error instead.
LEAN_MEMORY_MB = 6000


def capped_lean(src: Path, olean: Path, roots: list[Path], timeout: float = 900.0) -> str:
    """`prove.run_lean` with Lean's memory limit."""
    src, olean = Path(src).resolve(), Path(olean).resolve()
    args = [f"--memory={LEAN_MEMORY_MB}", "-R", str(src.parent), "-o", str(olean), str(src)]
    code, output, _ = prove._lean(args, roots, timeout)
    if code is None:
        return f"{src}:1:0: error: Lean did not finish in {timeout:.0f} s"
    if code != 0 and not any(m.severity == "error" for m in prove.parse_messages(output)):
        said = " ".join(output.split())[-800:] or "nothing"
        output += f"\n{src}:1:0: error: Lean exited with {code}, printing: {said}"
    return output


@dataclasses.dataclass
class Verdict:
    ok: bool
    stage: str  # deny | statements | compile | checker | ok
    errors: list[str]
    seconds: float
    axioms: list[str] = dataclasses.field(default_factory=list)


def judge(fp: fp_export.FPFile, b: bound.Bound, helpers: str, proof: str, work: Path) -> Verdict:
    """The judge of `prove.py`, on the floating-point statement."""
    t0 = time.time()
    bad = prove.banned(helpers, helpers=True) + prove.banned(proof)
    if bad:
        return Verdict(False, "deny", [f"not allowed in your text: {', '.join(bad)}"], 0.0)
    work = Path(work).resolve()
    stmts = work / "statements" / f"{fp.library}-{fp.key(b)}"
    olean = stmts / f"{STATEMENTS}.olean"
    if not olean.exists():
        stmts.mkdir(parents=True, exist_ok=True)
        src = stmts / f"{STATEMENTS}.lean"
        src.write_text(fp.statements(b), encoding="utf-8")
        out, _ = prove.run_lean(src, olean=olean)
        errs = [m for m in prove.parse_messages(out) if m.severity == "error"]
        if errs or not olean.exists():
            msgs = [f"{m.line}: {m.text}" for m in errs] or [out[-1500:]]
            return Verdict(False, "statements", msgs, time.time() - t0)
    root = work / f"proof-{int(time.time() * 1000)}"
    root.mkdir(parents=True, exist_ok=True)
    src = root / f"{PROOFS}.lean"
    text = fp.proofs(b, helpers, proof, STATEMENTS)
    src.write_text(text, encoding="utf-8")
    out = capped_lean(src, root / f"{PROOFS}.olean", [root, stmts])
    errs = [m for m in prove.parse_messages(out) if m.severity == "error"]
    if errs or not (root / f"{PROOFS}.olean").exists():
        lines = text.splitlines()
        msgs = []
        for m in errs[:8]:
            where = lines[m.line - 1].strip() if 0 < m.line <= len(lines) else ""
            msgs.append(f"line {m.line} `{where[:100]}`: {prove._clip(m.text, 900)}")
        return Verdict(False, "compile", msgs or [out[-1500:]], time.time() - t0)
    request = root / "check.json"
    request.write_text(
        json.dumps(
            {
                "statements": STATEMENTS,
                "proofs": PROOFS,
                "namespace": fp.namespace.split("."),
                "obligations": ["fp_bound"],
            }
        ),
        encoding="utf-8",
    )
    code, output, _ = prove._lean(["--run", str(prove.CHECKER), str(request)], [root, stmts], 900)
    try:
        verdict = json.loads(output.strip().splitlines()[-1]) if code == 0 else None
    except (json.JSONDecodeError, IndexError):
        verdict = None
    if not isinstance(verdict, dict):
        return Verdict(False, "checker", [f"checker failed: {output[-800:]}"], time.time() - t0)
    if verdict.get("replay"):
        return Verdict(False, "checker", [f"kernel replay: {verdict['replay']}"], time.time() - t0)
    info = (verdict.get("checks") or {}).get("fp_bound") or {}
    axioms = [str(a) for a in info.get("axioms", [])]
    typed = all(info.get(k) for k in ("found", "statement", "type"))
    if not typed or not set(axioms) <= prove.ALLOWED_AXIOMS:
        why = "not a theorem of the generated statement" if not typed else f"axioms {axioms}"
        return Verdict(False, "checker", [why], time.time() - t0, axioms)
    return Verdict(True, "ok", [], time.time() - t0, axioms)


# --------------------------------------------------------------------------- the loop


def _cost(model) -> float:
    return float(llm.usage(model).get("cost_usd", 0.0))


def certify(
    module: str,
    out: Path,
    proposer: str = "claude:sonnet",
    strong: str = "claude:opus",
    proposals: int = 3,
    rounds: int = 3,
    strong_rounds: int = 3,
    steps: int = 150,
    work: Path | None = None,
) -> dict:
    """Propose, attack and prove a bound for the library program ``module``."""
    entry = {m: (p, prog) for m, p, prog in fptable.library()}[module]
    problem = problems.PROBLEMS[entry[0]]
    prog = entry[1]
    fp = fp_export.export(prog, problem, module)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompts").mkdir(exist_ok=True)
    work = Path(work or prove.WORKDIR / "fp") / module
    model = llm.from_spec(proposer)
    stronger = llm.from_spec(strong)
    record: dict = {"program": module, "proposals": [], "proved": None, "started": time.time()}
    history: list[str] = []
    for k in range(proposals):
        prompt = bound_prompt(fp, problem, history)
        (out / "prompts" / f"{module}-bound-{k + 1}.txt").write_text(prompt, encoding="utf-8")
        reply = model(prompt, SYSTEM)
        item: dict = {"k": k + 1, "reply_tail": reply[-400:]}
        record["proposals"].append(item)
        try:
            b = parse_bound_reply(reply, problem.name)
        except bound.BoundError as e:
            item["rejected"] = f"grammar: {e}"
            history.append(f"- proposal {k + 1}: not in the grammar ({e})")
            continue
        item["B"] = b.text
        found = attack(prog, b, steps, problem=problem.name)
        item["attack"] = {k2: v for k2, v in found.items() if k2 != "x"}
        if found["ratio"] > 1.0:
            arrays = {k2: np.asarray(v, dtype=np.float32) for k2, v in found["x"].items()}
            np.savez_compressed(out / f"{module}-counterexample-{k + 1}.npz", **arrays)
            item["broken"] = True
            history.append(
                f"- proposal {k + 1}: B = {b.text}. A float32 run breaks it: n = {found['n']:.0f}, "
                f"R = {found['R']:.6g}, M = {found['M']:.6g}, blocks of {found['block']} "
                f"(D = {found['D']:.0f}), {found['order']} merge tree of depth "
                f"d = {found['d']:.0f}; measured {QUANTITY[problem.name]} = "
                f"{found['error']:.4g}, where B = {found['B']:.4g}."
            )
            continue
        item["broken"] = False
        item["tightness"] = 1.0 / found["ratio"] if found["ratio"] > 0 else math.inf
        # the prover
        pprompt = proof_prompt(fp, b)
        (out / "prompts" / f"{module}-proof.txt").write_text(pprompt, encoding="utf-8")
        attempts = []
        item["proof"] = attempts
        conversation = pprompt
        for r in range(rounds + strong_rounds):
            who = model if r < rounds else stronger
            reply = who(conversation, SYSTEM)
            try:
                blocks, helpers = prove.parse_reply(reply, {"fp_bound"})
                helpers, proof = helpers or "", blocks["fp_bound"]
            except (prove.ReplyError, KeyError) as e:
                attempts.append(
                    {"round": r + 1, "model": _name(who), "stage": "reply", "error": str(e)}
                )
                conversation = (
                    pprompt + f"\n\nYour last reply could not be read: {e}. Use the format."
                )
                continue
            v = judge(fp, b, helpers, proof, work)
            attempts.append(
                {
                    "round": r + 1,
                    "model": _name(who),
                    "stage": v.stage,
                    "ok": v.ok,
                    "seconds": round(v.seconds, 1),
                    "errors": v.errors[:3],
                }
            )
            if v.ok:
                record["proved"] = {
                    "B": b.text,
                    "lean": b.lean(),
                    "round": r + 1,
                    "model": _name(who),
                    "tightness": item["tightness"],
                    "axioms": v.axioms,
                }
                (out / f"{module}-proof.lean").write_text(
                    fp.proofs(b, helpers, proof, STATEMENTS), encoding="utf-8"
                )
                (out / f"{module}-statements.lean").write_text(fp.statements(b), encoding="utf-8")
                break
            conversation = (
                pprompt
                + "\n\nYour previous attempt:\n\n```lean helpers\n"
                + helpers
                + "\n```\n\n```lean fp_bound\n"
                + proof
                + "\n```\n\nThe judge rejected it ("
                + v.stage
                + "):\n"
                + "\n".join(v.errors)
                + "\n\nFix the proof. Same reply format."
            )
            if r == 0:
                (out / "prompts" / f"{module}-proof-feedback.txt").write_text(
                    conversation, encoding="utf-8"
                )
        if record["proved"]:
            break
        history.append(f"- proposal {k + 1}: B = {b.text} survived the runs but was not proved.")
    record["cost_usd"] = round(_cost(model) + _cost(stronger), 4)
    record["seconds"] = round(time.time() - record.pop("started"), 1)
    (out / f"{module}.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    return record


def _name(model) -> str:
    return getattr(model, "model", None) or type(model).__name__


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--program", required=True, help="a module of the proof library")
    ap.add_argument("--out", default="discovery/results/m3")
    ap.add_argument("--proposer", default="claude:sonnet")
    ap.add_argument("--strong", default="claude:opus")
    ap.add_argument("--proposals", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--strong-rounds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=150)
    args = ap.parse_args(argv)
    rec = certify(
        args.program,
        Path(args.out),
        args.proposer,
        args.strong,
        args.proposals,
        args.rounds,
        args.strong_rounds,
        args.steps,
    )
    print(json.dumps(rec, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
