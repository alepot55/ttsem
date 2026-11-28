"""An agentic prover: the model with Lean in the loop, in a sandbox outside the repository.

The prover of :mod:`discovery.fpbound` asks for a whole proof in one reply and gives Lean's errors
back for the next round. A long proof never gets written that way: in the first runs the models
replied with an outline marked INCOMPLETE, and the deny list refuses ``sorry``, so a partial proof
never reaches Lean. Here ``claude -p`` runs as an agent:

* four tools: ``Read``, ``Write`` and ``Edit``, and ``Bash`` allowed only for commands that
  start with ``lean``; no web, no MCP, no user settings. Until the review (diary, 12:35 UTC) the
  first three reached the whole disk and Lean's ``#eval`` could read any file: the prompts held
  no bound, but nothing stopped an agent from reading one (the sandboxes show no trace of it,
  and the transcripts were not kept). Since then :data:`DENIED` keeps the tools out of the
  repository, the home directory and the earlier sandboxes, and the ``lean`` of the sandbox
  refuses a file that names file or process IO (:data:`GUARD`);
* the sandbox holds the compiled statements module (first on ``LEAN_PATH``), its source for
  reading, and one file to edit, ``FPProofs.lean``, with the statement fixed and ``sorry`` in
  place of the proof. ``sorry`` is allowed while the agent works;
* the repository is not in the sandbox: no diary, no research notes, no prior bounds. The prompt
  is the prover's prompt of ``fpbound`` (statement, framework signatures, exact semantics) and the
  rules of the sandbox.

When the agent stops, the helper lemmas and the proof are cut out of its file and judged by
:func:`discovery.fpbound.judge`, which rebuilds the module around the generated statement, so
what the agent did to the statement's text does not count.

    python -m discovery.agentprove --program SoftmaxOnline --bound "2*u*(d+1)*R + ..."
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from discovery import bound, fp_export, fpbound, fptable, llm, problems, prove

SANDBOXES = Path(r"C:\dev\fpbox2")  # outside the repository; a path without spaces
# The sandboxes of the runs before the review: their files hold the earlier proofs.
EARLIER = Path(r"C:\dev\fpbox")

# What the agent's Read, Write and Edit may not touch, as Claude Code's permission rules write
# Windows paths (a deny beats an allow): the projects (the diary, the research notes, the
# results), the home directory (the session's transcripts and notes, the judge's temporary
# files), the earlier sandboxes and the personal folders.
DENIED = (
    "//c/dev/progetti/**",
    "//c/Users/**",
    "//c/dev/fpbox/**",
    "//c/dev/assistente/**",
    "//c/dev/.claude/**",
    "//c/dev/CLAUDE.md",
)

# The sandbox's `lean`: Lean's IO reaches the whole disk, so a file that names file or process IO
# is refused before Lean runs. Printing numbers with `#eval` stays allowed.
GUARD = r"""import re, subprocess, sys
BAN = re.compile(r"IO\.FS|readFile|readBinFile|IO\.Process|FilePath|getEnv"
                 r"|appPath|currentDir|Handle|System\.")
real, mem, args = sys.argv[1], sys.argv[2], sys.argv[3:]
for a in args:
    if a.endswith(".lean"):
        try:
            text = open(a, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        if BAN.search(text):
            print(f"{a}:1:0: error: file and process IO are not allowed in the sandbox")
            sys.exit(1)
sys.exit(subprocess.call([real, "--memory=" + mem, *args]))
"""


def denied_tools() -> str:
    return ",".join(f"{tool}({path})" for tool in ("Read", "Write", "Edit") for path in DENIED)


SYSTEM = (
    "You are an expert in Lean 4 and Mathlib and a careful numerical analyst. You work alone in "
    "a sandbox directory and check your work with Lean."
)

RULES = """\
You work in the current directory, a sandbox. It holds:

- `FPStatements.lean`: the generated statements module, already compiled (read it; do not edit
  it; your file imports it);
- `FPProofs.lean`: your file. It contains the theorem `fp_bound` with its statement fixed and
  `sorry` in place of the proof. Write helper lemmas where the file says so, and the proof of
  `fp_bound`.

Check your file with the command `lean FPProofs.lean` (the only shell command you may run; it may
take a minute). You may also create other files here, such as `Scratch.lean` with `#check` or
`example` lines, and run `lean Scratch.lean`; `exact?` and `apply?` work there.

You may use `sorry` while you work, but the final file must have none, and Lean must report no
error. Helper lemmas are `theorem` or `lemma` declarations only (no `def`, `abbrev`, `instance`,
attribute, `set_option` other than `set_option maxHeartbeats N in` before a declaration, no `#`
command in FPProofs.lean); no `native_decide`, `admit` or `axiom`. Do not change the statement of
`fp_bound`, the `fp_bound_holds` line or the file's first lines: they are restored before judging.
If the statement is false, say why in your final message instead of proving it.

Work until `lean FPProofs.lean` reports no error and no `sorry`, then reply with one line:
DONE or FALSE."""


def sandbox(fp: fp_export.FPFile, b: bound.Bound, name: str) -> Path:
    """A fresh sandbox with the compiled statements module and the file to edit."""
    root = SANDBOXES / f"{name}-{fp.key(b)}-{int(time.time())}"
    root.mkdir(parents=True, exist_ok=True)
    src = root / f"{fpbound.STATEMENTS}.lean"
    src.write_text(fp.statements(b), encoding="utf-8")
    out, _ = prove.run_lean(src, olean=root / f"{fpbound.STATEMENTS}.olean")
    errs = [m for m in prove.parse_messages(out) if m.severity == "error"]
    if errs:
        raise prove.LeanError(f"the statements do not compile: {errs[0].text[:300]}")
    template = fp.proofs(b, "-- helper lemmas here", "sorry", fpbound.STATEMENTS)
    (root / f"{fpbound.PROOFS}.lean").write_text(template, encoding="utf-8")
    return root


def _wrapper() -> Path:
    """A `lean` that runs the toolchain's Lean with the judge's memory limit, first on PATH,
    through :data:`GUARD`."""
    bindir = SANDBOXES / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    real = Path(prove.lean_env()["LEAN"])
    mem = fpbound.LEAN_MEMORY_MB
    guard = bindir / "guard.py"
    guard.write_text(GUARD, encoding="utf-8")
    py = Path(sys.executable)
    sh = f'#!/bin/sh\nexec "{py.as_posix()}" "{guard.as_posix()}" "{real.as_posix()}" {mem} "$@"\n'
    cmd = f'@"{py}" "{guard}" "{real}" {mem} %*\r\n'
    (bindir / "lean").write_text(sh, encoding="utf-8", newline="\n")
    (bindir / "lean.cmd").write_text(cmd, encoding="utf-8", newline="")
    return bindir


def _env(root: Path) -> dict[str, str]:
    env = dict(prove.lean_env())
    env["LEAN_PATH"] = os.pathsep.join([str(root), env["LEAN_PATH"]])
    env["PATH"] = str(_wrapper()) + os.pathsep + env.get("PATH", "")
    return env


def extract(text: str) -> tuple[str, str]:
    """The helper lemmas and the proof of ``fp_bound`` in the agent's file."""
    head = text.index("theorem fp_bound :")
    opened = text.rfind("\nopen ", 0, head)
    start = text.index("\n", opened + 1) if opened >= 0 else 0
    helpers = text[start:head].strip()
    body_start = text.index(":= by", head) + len(":= by")
    end = text.index("theorem fp_bound_holds", body_start)
    return helpers or "-- none", text[body_start:end].strip("\n")


def run_agent(prompt: str, root: Path, model: str, max_turns: int, timeout: float) -> dict:
    exe = shutil.which("claude") or "claude"
    cmd = [
        exe,
        "-p",
        "--output-format",
        "json",
        "--model",
        model,
        "--tools",
        "Read,Write,Edit,Bash",
        "--allowedTools",
        "Read,Write,Edit,Bash(lean:*)",
        "--disallowedTools",
        denied_tools(),
        "--permission-mode",
        "acceptEdits",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--setting-sources",
        "",
        "--max-turns",
        str(max_turns),
        "--system-prompt",
        SYSTEM,
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=root,
            env=_env(root),
        )
        out = json.loads(proc.stdout) if proc.stdout.strip().startswith("{") else {}
        error = "" if out else (proc.stderr[-500:] or proc.stdout[-500:])
    except subprocess.TimeoutExpired:
        out, error = {}, f"timeout after {timeout:.0f} s"
    return {
        "seconds": round(time.time() - t0, 1),
        "cost_usd": float(out.get("total_cost_usd") or 0.0),
        "turns": out.get("num_turns"),
        "result": str(out.get("result", ""))[-1500:],
        "subtype": out.get("subtype"),
        "error": error,
    }


def prove_agentic(
    module: str,
    btext: str,
    out: Path,
    models: tuple[str, ...] = ("sonnet", "opus"),
    max_turns: int = 150,
    timeout: float = 3600.0,
) -> dict:
    """Run the agent on the statement of ``module`` with bound ``btext``; judge what it leaves."""
    lib = {m: (p, prog) for m, p, prog in fptable.library()}
    pname, prog = lib[module]
    b = bound.parse(btext, pname)
    fp = fp_export.export(prog, problems.PROBLEMS[pname], module)
    return prove_file(fp, b, module, out, models, max_turns, timeout)


def prove_file(
    fp: fp_export.FPFile,
    b: bound.Bound,
    module: str,
    out: Path,
    models: tuple[str, ...] = ("sonnet", "opus"),
    max_turns: int = 150,
    timeout: float = 3600.0,
    library_text: bool = True,
) -> dict:
    """Run the agent on the statement ``fp`` (with ``b`` when it has a bound); judge the result."""
    out.mkdir(parents=True, exist_ok=True)
    prompt = fpbound.proof_prompt(fp, b, library_text) + "\n\n" + RULES
    (out / "prompts").mkdir(exist_ok=True)
    (out / "prompts" / f"{module}-agent.txt").write_text(prompt, encoding="utf-8")
    record: dict = {"program": module, "B": b.text, "attempts": [], "proved": None}
    for model in models:
        root = sandbox(fp, b, module)
        run = run_agent(prompt, root, model, max_turns, timeout)
        text = (root / f"{fpbound.PROOFS}.lean").read_text(encoding="utf-8")
        attempt = {"model": model, "sandbox": str(root), **run}
        try:
            helpers, proof = extract(text)
            v = fpbound.judge(fp, b, helpers, proof, prove.WORKDIR / "fp-agent" / module)
            attempt.update({"stage": v.stage, "ok": v.ok, "errors": v.errors[:3]})
        except (ValueError, fp_export.FPExportError) as e:
            attempt.update({"stage": "extract", "ok": False, "errors": [str(e)]})
            v = None
        record["attempts"].append(attempt)
        if v is not None and v.ok:
            record["proved"] = {"model": model, "axioms": v.axioms, "B": b.text, "lean": b.lean()}
            (out / f"{module}-proof.lean").write_text(
                fp.proofs(b, helpers, proof, fpbound.STATEMENTS), encoding="utf-8"
            )
            (out / f"{module}-statements.lean").write_text(fp.statements(b), encoding="utf-8")
            break
    record["cost_usd"] = round(sum(a["cost_usd"] for a in record["attempts"]), 4)
    (out / f"{module}-agent.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def propose_and_prove(
    module: str,
    out: Path,
    proposals: int = 3,
    models: tuple[str, ...] = ("sonnet", "opus"),
    max_turns: int = 150,
    timeout: float = 3600.0,
) -> dict:
    """The whole path for one library program: the model proposes B, the adversary attacks it,
    and a surviving B goes to the agent; a B the agent does not prove goes back to the proposer
    with the agent's own last words (its counterexample, when it gives one)."""
    lib = {m: (p, prog) for m, p, prog in fptable.library()}
    pname, prog = lib[module]
    problem = problems.PROBLEMS[pname]
    fp = fp_export.export(prog, problem, module)
    model = llm.from_spec("claude:sonnet")
    history: list[str] = []
    rec: dict = {"program": module, "proposals": []}
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    for k in range(proposals):
        prompt = fpbound.bound_prompt(fp, problem, history)
        (out / "prompts" / f"{module}-v{fp_export.VERSION}-bound-{k + 1}.txt").write_text(
            prompt, encoding="utf-8"
        )
        reply = model(prompt, fpbound.SYSTEM)
        try:
            b = fpbound.parse_bound_reply(reply, pname)
        except bound.BoundError as e:
            history.append(f"- proposal {k + 1}: not in the grammar ({e})")
            rec["proposals"].append({"k": k + 1, "rejected": str(e)})
            continue
        found = fpbound.attack(prog, b, 150, problem=pname)
        item = {"k": k + 1, "B": b.text, "ratio": found["ratio"]}
        rec["proposals"].append(item)
        if found["ratio"] > 1:
            history.append(
                f"- proposal {k + 1}: B = {b.text}. A float32 run breaks it: measured "
                f"{fpbound.QUANTITY[pname]} = {found['error']:.4g} where B = {found['B']:.4g} "
                f"(n = {found['n']:.0f}, R = {found['R']:.6g}, M = {found['M']:.6g}, "
                f"{found['order']} tree of depth {found['d']:.0f}, blocks of {found['block']})."
            )
            continue
        item["tightness_attack"] = 1.0 / found["ratio"] if found["ratio"] > 0 else None
        proof = prove_file(
            fp, b, f"{module}-v{fp_export.VERSION}-{k + 1}", out, models, max_turns, timeout
        )
        item["proof"] = {k2: proof[k2] for k2 in ("proved", "cost_usd")}
        if proof.get("proved"):
            rec["proved"] = {"B": b.text, **proof["proved"]}
            break
        said = (proof["attempts"][-1].get("result") or "")[-600:] if proof["attempts"] else ""
        history.append(f"- proposal {k + 1}: B = {b.text} was not proved. The prover said: {said}")
    rec["proposer_cost_usd"] = float(llm.usage(model).get("cost_usd", 0.0))
    (out / f"{module}-v{fp_export.VERSION}.json").write_text(
        json.dumps(rec, indent=1), encoding="utf-8"
    )
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--program", required=True)
    ap.add_argument("--bound", help="B in the grammar of discovery.bound; without it, proposed")
    ap.add_argument("--out", default="discovery/results/m3/agent")
    ap.add_argument("--models", nargs="+", default=["sonnet", "opus"])
    ap.add_argument("--max-turns", type=int, default=150)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--proposals", type=int, default=3)
    args = ap.parse_args(argv)
    if args.bound is None:
        rec = propose_and_prove(
            args.program,
            Path(args.out),
            args.proposals,
            tuple(args.models),
            args.max_turns,
            args.timeout,
        )
    else:
        rec = prove_agentic(
            args.program,
            args.bound,
            Path(args.out),
            tuple(args.models),
            args.max_turns,
            args.timeout,
        )
    print(json.dumps(rec, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
