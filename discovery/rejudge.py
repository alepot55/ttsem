"""M6: every proof the session accepted, judged again by the fixed judge.

The review of the judge (diary, 12:35 UTC) found two holes: the deny list read Unicode
identifiers unlike Lean (`λdef` passed as one word), and the statements module sat at a path an
agent could predict while its Lean could write files. Both are fixed in `fpbound.judge` and
`prove.banned`. This module takes every accepted proof under `results/`, reads the program and
the namespace from the saved statements, generates the statements again with the current
exporter, checks that they say what the saved ones said once comments are removed (the review
also changed two docstrings), and judges the saved proof again, in a fresh directory.

    python -m discovery.rejudge          # writes results/m6/rejudge.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from discovery import agentprove, bound, fp_export, fpbound, problems, prove

RESULTS = Path(__file__).parent / "results"
OUT = RESULTS / "m6" / "rejudge.json"

# (the saved proofs module, its record holding B, the problem, the kind): kind None is the
# library export of M3, otherwise the policy export of that kind (M4, and M5's elites)
PROOFS = [
    ("m3/agent/SoftmaxOnline-proof.lean", "m3/agent/SoftmaxOnline-agent.json", "softmax", None),
    ("m3/agent/SoftmaxLse-proof.lean", "m3/agent/SoftmaxLse-agent.json", "softmax", None),
    ("m3/agent/SoftmaxLse-proof-sonnet.lean", "m3/agent/SoftmaxLse-rejudged.json", "softmax", None),
    ("m3/agent/VarianceSumsq-proof.lean", "m3/agent/VarianceSumsq-agent.json", "variance", None),
    (
        "m3/agent/AttentionOnline-v4-1-proof.lean",
        "m3/agent/AttentionOnline-v4-1-agent.json",
        "attention",
        None,
    ),
    (
        "m4/SoftmaxConditional-exact-proof.lean",
        "m4/SoftmaxConditional-exact-agent.json",
        "policy",
        "exact",
    ),
    (
        "m4/SoftmaxConditional-overflow-proof.lean",
        "m4/SoftmaxConditional-overflow-agent.json",
        "policy",
        "overflow",
    ),
    (
        "m4/nounderflow/SoftmaxConditional-fp-nounderflow-proof.lean",
        "m4/nounderflow/SoftmaxConditional-fp-nounderflow-agent.json",
        "policy",
        "fp-nounderflow",
    ),
    ("m5/proofs/Cand4f04f56628-proof.lean", "m5/proofs/Cand4f04f56628-agent.json", "policy", "fp"),
    ("m5/proofs/Cand4fb632176b-proof.lean", "m5/proofs/Cand4fb632176b-agent.json", "policy", "fp"),
    ("m5/proofs/Candfbc0a9179c-proof.lean", "m5/proofs/Candfbc0a9179c-agent.json", "policy", "fp"),
]


def strip_comments(text: str) -> str:
    """Lean source without its comments (block comments nest), whitespace collapsed."""
    out, i, depth = [], 0, 0
    while i < len(text):
        if text.startswith("/-", i):
            depth, i = depth + 1, i + 2
        elif depth and text.startswith("-/", i):
            depth, i = depth - 1, i + 2
        elif depth:
            i += 1
        elif text.startswith("--", i):
            i = text.find("\n", i)
            i = len(text) if i < 0 else i
        else:
            out.append(text[i])
            i += 1
    return " ".join("".join(out).split())


def saved_program(statements: str) -> tuple[str, str, str]:
    """The namespace, the program text and the digest prefix the saved statements name."""
    ns = re.search(r"^namespace Discovery\.FP\.(\S+)", statements, re.M)
    text = re.search(r"```\n(\(program .*?)\n```", statements, re.S)
    digest = re.search(r"\(digest ([0-9a-f]+)\)", statements)
    if not (ns and text and digest):
        raise ValueError("the saved statements do not name their program")
    return ns.group(1), text.group(1), digest.group(1)


def _first_difference(a: str, b: str) -> str:
    k = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
    return f"saved …{a[max(0, k - 60) : k + 60]}… now …{b[max(0, k - 60) : k + 60]}…"


def rejudge_one(proof_path: str, record_path: str, problem: str, kind: str | None) -> dict:
    proof_file = RESULTS / proof_path
    stmts_file = proof_file.with_name(
        proof_file.name.replace("-proof-sonnet.lean", "-statements.lean").replace(
            "-proof.lean", "-statements.lean"
        )
    )
    saved = stmts_file.read_text(encoding="utf-8")
    ns, text, digest = saved_program(saved)
    record = json.loads((RESULTS / record_path).read_text(encoding="utf-8"))
    b = bound.parse(record["B"], problem)
    parse = problems.SOFTMAX.parse if problem == "policy" else problems.PROBLEMS[problem].parse
    prog = parse(text)
    item: dict = {
        "proof": proof_path,
        "namespace": ns,
        "kind": kind or "library",
        "B": record["B"],
        "digest_matches": prog.digest().startswith(digest),
    }
    if kind is None:
        fp = fp_export.export(prog, problems.PROBLEMS[problem], ns)
    else:
        fp = fp_export.export_policy(prog, problems.SOFTMAX, ns, kind)
    now = fp.statements(b)
    a, c = strip_comments(saved), strip_comments(now)
    item["same_statement"] = a == c
    if a != c:
        item["difference"] = _first_difference(a, c)
    helpers, body = agentprove.extract(proof_file.read_text(encoding="utf-8"))
    work = prove.WORKDIR / "rejudge" / proof_file.stem
    t0 = time.time()
    v = fpbound.judge(fp, b, helpers, body, work)
    item.update(
        {
            "ok": v.ok,
            "stage": v.stage,
            "errors": v.errors[:3],
            "axioms": v.axioms,
            "seconds": round(time.time() - t0, 1),
        }
    )
    return item


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--only", nargs="*", help="substrings of the proofs to judge")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in PROOFS:
        if args.only and not any(s in entry[0] for s in args.only):
            continue
        try:
            row = rejudge_one(*entry)
        except Exception as exc:  # noqa: BLE001  recorded, and the next proof is judged
            row = {"proof": entry[0], "ok": False, "stage": "error", "errors": [repr(exc)]}
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("proof", "ok", "stage", "same_statement")}))
        out.write_text(json.dumps({"proofs": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    ok = sum(1 for r in rows if r.get("ok") and r.get("same_statement"))
    print(f"{ok}/{len(rows)} accepted again, with the statement unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
