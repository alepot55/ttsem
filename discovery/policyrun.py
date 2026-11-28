"""M4: a rescaling policy's exactness, overflow and floating-point bound, proved; its worst cases.

For a policy of :mod:`discovery.policies` (the conditional one: rescale only when the maximum
grows by more than τ):

* ``--kind exact``: over ℝ, the normalizer the final state encodes is exact for every merge tree
  (the state's reference need not be the maximum);
* ``--kind overflow``: over ℝ, the sum field is at most n·exp(τ·ρ), ρ the tree's right depth;
* ``--kind fp``: the model proposes ``B`` (atoms of :mod:`discovery.bound` plus ``tau`` and
  ``rd``), the adversary attacks it on float32 runs for τ in {0, 2, 8}, blocks of 64 and 256,
  left, right, balanced and random trees (runs that overflow are outside the model and skipped),
  and a surviving ``B`` goes to the agentic prover;
* ``--kind worst``: the adversarial lower bounds: the largest lag of the reference behind the
  maximum per tree shape and τ (and the overflow it causes), and the largest error found.

Every kind proves with :mod:`discovery.agentprove` (the model, with Lean in a sandbox).

    python -m discovery.policyrun --policy conditional --kind exact
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np

from discovery import (
    adversary,
    agentprove,
    bound,
    fp_export,
    fpbound,
    fsem,
    ir,
    llm,
    policies,
    problems,
)

TAUS = (0.0, 2.0, 8.0)
ORDERS = ("left", "right", "balanced", "random")
BLOCKS = (64, 256)


def measure(prog: ir.Program, x: np.ndarray, block: int, order: str, tau: float) -> tuple:
    """|log Ẑ − log Z| of one float32 run (inf when it overflows), and the atoms of the run."""
    F = fsem.FORMATS["float32"]
    xs = np.asarray(F.value(x), dtype=np.float64)
    r = fsem.run(prog, {"x": xs[:, None]}, "float32", block=block, order=order, params={"tau": tau})
    m = float(xs.max())
    zhat = fsem._normalizer(prog, r, m)
    D_ref = math.fsum(np.exp(xs - m).tolist())
    state_finite = all(np.all(np.isfinite(np.asarray(v))) for v in r.finished.values())
    ok = zhat > 0 and math.isfinite(zhat) and state_finite
    err = abs(math.log(zhat) - math.log(D_ref)) if ok else math.inf
    lo, hi = float(xs.min()), float(xs.max())
    env = dict(fpbound.ENV32)
    env.update(
        {
            "D": float(math.ceil(math.log2(min(block, xs.size)))),
            "d": float(r.depth),
            "rd": float(r.rdepth),
            "n": float(xs.size),
            "R": hi - lo,
            "M": max(abs(lo), abs(hi)),
            "tau": tau,
            # how far a reference field `m` lags the maximum, for the states that have one
            "lag": m - float(next((v for k, v in r.finished.items() if k.endswith(".m")), m)),
        }
    )
    return err, env


def ratio(prog: ir.Program, b: bound.Bound, x: np.ndarray) -> tuple[float, dict]:
    """The largest error / B over τ, blockings and trees; overflowing runs are skipped."""
    worst, info = -1.0, {}
    for tau in TAUS:
        for block in BLOCKS:
            for order in ORDERS:
                err, env = measure(prog, x, block, order, tau)
                if not math.isfinite(err):
                    continue
                B = b.value(env)
                q = err / B if B > 0 else math.inf
                if q > worst:
                    worst = q
                    info = {"block": block, "order": order, "error": err, "B": B, **env}
    return worst, info


def _lag_family(rng: np.random.Generator, n: int, block: int, step: float) -> np.ndarray:
    """Block maxima rising by ``step``: each merge keeps a stale reference when step ≤ τ."""
    x = 0.1 * rng.standard_normal(n)
    for b, s in enumerate(range(0, n, block)):
        x[s : s + block] += b * step
    return x


def attack(prog: ir.Program, b: bound.Bound, steps: int = 100, seeds=(0, 1)) -> dict:
    best: dict = {"ratio": -1.0}
    F = fsem.FORMATS["float32"]
    starts = {
        f: (lambda rng, f=f: adversary.generate("softmax", f, rng, 4096, 256)["x"][:, 0])
        for f in adversary.families("softmax")
    }
    for step in (1.9, 7.8):
        starts[f"lag{step}"] = lambda rng, step=step: _lag_family(rng, 4096, 256, step)
    for name, make in starts.items():
        for seed in seeds:
            rng = np.random.default_rng(9000 + seed)
            x = np.asarray(F.value(make(rng)), dtype=np.float64)
            cur, info = ratio(prog, b, x)
            for _ in range(steps):
                cand = adversary.mutate("softmax", {"x": x[:, None]}, F, rng, 256)["x"][:, 0]
                q, inf2 = ratio(prog, b, cand)
                if math.isfinite(q) and q > cur:
                    x, cur, info = cand, q, inf2
            if cur > best["ratio"]:
                best = {"ratio": cur, "family": name, "seed": seed, **info}
    return best


def worst(prog: ir.Program) -> dict:
    """The lag of the reference and the overflow, per tree shape and τ, on rising block maxima."""
    rows = []
    for tau in (2.0, 4.0, 8.0):
        x = _lag_family(np.random.default_rng(0), 4096, 256, 0.98 * tau)
        for order in ("left", "right", "balanced"):
            err, env = measure(prog, x, 256, order, tau)
            rows.append(
                {
                    "tau": tau,
                    "order": order,
                    "depth": env["d"],
                    "rdepth": env["rd"],
                    "lag": env["lag"],
                    "lag_over_tau": env["lag"] / tau,
                    "overflow": not math.isfinite(err),
                    "error_u": err / fpbound.U32 if math.isfinite(err) else None,
                }
            )
    return {"lag": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", default="conditional", choices=sorted(policies.POLICIES))
    kinds = [*fp_export.POLICY_KINDS, "worst"]
    ap.add_argument("--kind", required=True, choices=kinds)
    ap.add_argument("--out", default="discovery/results/m4")
    ap.add_argument("--proposals", type=int, default=3)
    ap.add_argument("--models", nargs="+", default=["sonnet", "opus"])
    ap.add_argument("--bound", help="fp-nounderflow: the bound B that survived the adversary")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prog = policies.program(args.policy)
    module = "Softmax" + args.policy.capitalize()
    if args.kind == "worst":
        rec = worst(prog)
        (out / f"{module}-worst.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(json.dumps(rec, indent=1))
        return 0
    suffix = "".join(w.capitalize() for w in args.kind.split("-"))  # a Lean name: no hyphen
    fp = fp_export.export_policy(prog, problems.SOFTMAX, f"{module}{suffix}", args.kind)
    name = f"{module}-{args.kind}"
    if args.kind in ("exact", "overflow"):
        rec = agentprove.prove_file(fp, bound.parse("u"), name, out, tuple(args.models))
        print(json.dumps(rec, indent=1)[:3000])
        return 0
    if args.kind == "fp-nounderflow":
        # the weakened statement, with the bound that survived the adversary on all runs
        rec = agentprove.prove_file(
            fp, bound.parse(args.bound, "policy"), name, out, tuple(args.models)
        )
        print(json.dumps(rec, indent=1)[:3000])
        return 0
    # the floating-point bound: propose, attack, prove
    model = llm.from_spec("claude:sonnet")
    problem = dataclasses.replace(problems.SOFTMAX, name="policy")
    history: list[str] = []
    record: dict = {"policy": args.policy, "proposals": []}
    for k in range(args.proposals):
        prompt = fpbound.bound_prompt(fp, problem, history)
        (out / "prompts").mkdir(exist_ok=True)
        (out / "prompts" / f"{name}-bound-{k + 1}.txt").write_text(prompt, encoding="utf-8")
        reply = model(prompt, fpbound.SYSTEM)
        item: dict = {"k": k + 1}
        record["proposals"].append(item)
        try:
            b = fpbound.parse_bound_reply(reply, "policy")
        except bound.BoundError as e:
            item["rejected"] = str(e)
            history.append(f"- proposal {k + 1}: not in the grammar ({e})")
            continue
        item["B"] = b.text
        found = attack(prog, b)
        item["attack"] = found
        if found["ratio"] > 1:
            history.append(
                f"- proposal {k + 1}: B = {b.text}. A float32 run breaks it: n = {found['n']:.0f}, "
                f"R = {found['R']:.6g}, M = {found['M']:.6g}, tau = {found['tau']:g}, blocks of "
                f"{found['block']} (D = {found['D']:.0f}), {found['order']} tree of depth "
                f"d = {found['d']:.0f} and right depth rd = {found['rd']:.0f}; measured "
                f"|log(znorm st) - log Σ exp x| = {found['error']:.4g}, where B = {found['B']:.4g}."
            )
            continue
        item["tightness"] = 1.0 / found["ratio"] if found["ratio"] > 0 else math.inf
        item["proof"] = agentprove.prove_file(fp, b, name, out, tuple(args.models))
        if item["proof"].get("proved"):
            break
        history.append(f"- proposal {k + 1}: B = {b.text} survived the runs but was not proved.")
    record["cost_usd_proposer"] = float(llm.usage(model).get("cost_usd", 0.0))
    (out / f"{name}.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    print(json.dumps(record, indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
