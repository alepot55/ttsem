"""M5: a search under a certified floating-point bound, and its two baselines.

Three arms, the same proposer (sonnet), the same number of proposals, no lemma in the prompts:

* ``cert``: the archive search of :func:`discovery.loop.run`, where a correct candidate enters
  the archive only with a certificate and the certificate is the fitness: the model proposes a
  bound B for it (the statement of :mod:`discovery.fp_export`, policy form, so parameters such as
  a threshold are allowed), the adversary of :mod:`discovery.policyrun` attacks B, and a B it
  cannot break is the certificate; its value at a reference point is the fitness, lower better;
* ``iid``: independent samples from the seed's prompt, certified the same way;
* ``plain``: the search of PR 12 with no certificate (fitness: the cost model's time); its final
  elites are certified afterwards, for the comparison.

A certificate here is a bound that survived the adversary; the proof of a final elite's bound
is a separate step (``--prove``, the agentic prover), and an elite whose bound is not proved is
not counted as certified in the report. Certificates are cached by digest, across arms and runs.

The reference point: n = 4,096, blocks of 256 (D = 8), the kernel's left fold (d = 15, rd = 1),
R = 10, M = 100, tau = the program's own value (8 for FlashAttention-4's threshold), numpy's float32
rounding. Values are in units of u = 2^-24.

    python -m discovery.certsearch --arm cert --seed 0 --iterations 6
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import threading
from pathlib import Path

from discovery import bound, fp_export, fpbound, llm, loop, policyrun, problems, propose

OUT = Path("discovery/results/m5")
REF = {**fpbound.ENV32, "D": 8.0, "d": 15.0, "rd": 1.0, "n": 4096.0, "R": 10.0, "M": 100.0}
_LOCK = threading.Lock()


def _cache_path(out: Path) -> Path:
    return out / "certificates.json"


def _load(out: Path) -> dict:
    p = _cache_path(out)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _store(out: Path, digest: str, cert: dict) -> None:
    with _LOCK:
        cache = _load(out)
        cache[digest] = cert
        _cache_path(out).write_text(json.dumps(cache, indent=1), encoding="utf-8")


def certifier(out: Path = OUT, proposals: int = 2, steps: int = 60):
    """A function of a correct verdict: its certificate {"value", "B", ...} or None."""
    model = llm.from_spec("claude:sonnet")
    out.mkdir(parents=True, exist_ok=True)

    def certify(verdict) -> dict | None:
        cached = _load(out).get(verdict.digest)
        if cached is not None:
            return cached or None
        cert: dict = {"name": verdict.name, "text": verdict.text}
        try:
            prog = problems.SOFTMAX.parse(verdict.text)
            fp = fp_export.export_policy(prog, problems.SOFTMAX, f"Cand{verdict.digest[:10]}", "fp")
        except (fp_export.FPExportError, Exception) as e:  # noqa: BLE001  any program it cannot state
            cert["why"] = f"no statement: {e}"
            _store(out, verdict.digest, cert)
            return None
        kind = "policy" if prog.params else "softmax"
        problem = dataclasses.replace(problems.SOFTMAX, name=kind)
        history: list[str] = []
        for k in range(proposals):
            reply = model(fpbound.bound_prompt(fp, problem, history), fpbound.SYSTEM)
            try:
                b = fpbound.parse_bound_reply(reply, kind)
            except bound.BoundError as e:
                history.append(f"- proposal {k + 1}: not in the grammar ({e})")
                continue
            found = policyrun.attack(prog, b, steps=steps, seeds=(0,))
            if found["ratio"] > 1:
                history.append(
                    f"- proposal {k + 1}: B = {b.text}. A float32 run breaks it: "
                    f"R = {found['R']:.6g}, M = {found['M']:.6g}, tau = {found['tau']:g}, "
                    f"{found['order']} tree of depth {found['d']:.0f}, measured "
                    f"{found['error']:.4g} where B = {found['B']:.4g}."
                )
                continue
            ref = {**REF, "tau": float(prog.params.get("tau", 0.0))}
            cert.update(
                value=b.value(ref) / fpbound.U32,
                B=b.text,
                kind=kind,
                proposals=k + 1,
                tightness=1.0 / found["ratio"] if found["ratio"] > 0 else None,
            )
            _store(out, verdict.digest, cert)
            return cert
        cert["why"] = "no bound survived the adversary"
        _store(out, verdict.digest, cert)
        return None

    certify.model = model
    return certify


def run_arm(arm: str, seed: int, iterations: int, out: Path = OUT) -> dict:
    model = llm.from_spec("claude:sonnet")
    where = out / f"{arm}-s{seed}"
    certify = None
    if arm in ("cert", "iid"):
        certify = certifier(out)
        propose.MEASURED = "certified error bound {:.4g} u"
    mode = "iid" if arm == "iid" else "archive"
    summary = loop.run(
        problems.SOFTMAX,
        model,
        iterations=iterations,
        mode=mode,
        seed=seed,
        out=where,
        library=None,
        certify=certify,
    )
    if certify is not None:
        summary["certifier_usage"] = llm.usage(certify.model)
        (where / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=["cert", "iid", "plain"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    s = run_arm(args.arm, args.seed, args.iterations, Path(args.out))
    print(json.dumps({k: s[k] for k in ("cells", "statuses", "usage") if k in s}, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
