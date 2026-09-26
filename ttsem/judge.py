"""``python -m ttsem judge``: a KernelBench answer written with Triton, judged without a GPU.

    python -m ttsem judge TASK.py ANSWER.py [--json] [--timeout S] [--scale auto|none]
                          [--rule kernelbench|v3] [--precision fp32|keep] [--gpu NAME] [--out DIR]
    python -m ttsem judge --manifest M.jsonl --out DIR [--jobs N] [--timeout S] [--full-max N]

A task is a KernelBench problem file (`Model`, `get_inputs`, `get_init_inputs`); an answer is the
file that defines `ModelNew` (or, in the KernelBench-v3 release, a replacement `Model`). Each
answer runs in its own process (``ttsem._judge_runner``) under `bwrap`: read-only filesystem
except the output directory, a private /tmp, no network, its own pid namespace, a timeout and an
address-space cap (`TTSEM_MEM_GB`, default 6). There the reference and the answer are compared
by KernelBench's correctness rule (or, with `--rule v3`, by the KernelBench-v3 harness's own
rule), with every Triton launch executed by ttsem on the CPU.

One verdict per pass:

- verified: every trial within atol/rtol, no fault
- wrong: values or shape differ from the reference in some trial, or under another autotune config
- unsafe: out-of-bounds access, race between program instances, or use of memory nobody wrote,
  with the kernel and its source line
- no_kernel: the answer launches no Triton kernel (PyTorch does the work: a known reward hack)
- error: the answer does not import, build or run
- too_slow: the timeout expired
- not_judged: the tool cannot say (an op ttsem does not model, its own error, the memory cap)

`--gpu` sets only the shared-memory limit: every launch is compiled for sm_90a, so whether an
autotune config compiles (the autotuner skips one that raises `CompileTimeAssertionFailure` or
`PTXASError`) is decided for that target, whatever the GPU.

The `scaled` pass always runs (with `--scale auto`): every size constant of the task is divided
by one power of two until its largest tensor holds at most 2**17 elements (`judge_shapes`). The
`full` pass runs too when the task's largest tensor at its real shape has at most `--full-max`
elements and its reference model's parameters at most 32 times that, and always after a
`no_kernel` verdict (an answer may launch its kernel only at the benchmark's own shape).

One pair prints one line per pass that ran (usually one) and exits 0 only if every pass says
`verified`. A manifest line is a JSON object with `id`, `task`, `answer` (relative paths are
taken from the manifest's directory), optionally `rule`, `precision` and `gpu`, and any labels
to carry along; the reports go to OUT/<id>.<pass>.json (a rerun reads them back), one row per
answer to OUT/rows.jsonl, and the counts per verdict to the terminal.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

VERDICTS = ("verified", "wrong", "unsafe", "no_kernel", "error", "too_slow", "not_judged")
RUNNER = "ttsem._judge_runner"
# the directory this `ttsem` is imported from, so the sandboxed process imports this very copy
ROOT = Path(__file__).resolve().parent.parent

KINDS = {
    "oob_read": "out-of-bounds read",
    "oob_write": "out-of-bounds write",
    "race": "race between program instances",
    "poison": "use of a value nobody wrote",
}

NO_BWRAP = """\
ttsem judge: {problem}. The answers are model-written code, and the judge runs them only
inside a bwrap sandbox (read-only filesystem, no network, own pid namespace). Install
bubblewrap (apt install bubblewrap, dnf install bubblewrap), or pass --no-sandbox to run
them unsandboxed, with your user's permissions, at your own risk.
"""


def sandbox_prefix(out: Path, cache: Path, keep: list[Path] | None = None) -> list[str]:
    """The `bwrap` command line every answer runs under: the filesystem read-only except OUT
    (where the run writes its report), a private /tmp, no network, its own pid namespace. The
    paths in `keep` that the private /tmp would hide (the task, the answer, this package, the
    Python running it) stay visible, read-only."""
    cmd = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
    for path in dict.fromkeys(p.resolve() for p in keep or []):
        if path.is_relative_to("/tmp"):
            cmd += ["--ro-bind", str(path), str(path)]
    return cmd + [
        "--bind", str(out), str(out), "--unshare-net", "--unshare-pid", "--die-with-parent",
        "--chdir", str(out), "--setenv", "TRITON_CACHE_DIR", str(cache),
        "--setenv", "OMP_NUM_THREADS", "1", "--setenv", "TMPDIR", "/tmp",
    ]  # fmt: skip


def sandbox_problem() -> str | None:
    """Why the sandbox cannot run here (no `bwrap` on PATH, or one that cannot create the
    namespaces, as where unprivileged user namespaces are off), or None."""
    if shutil.which("bwrap") is None:
        return "bwrap is not on PATH"
    probe = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
    probe += ["--unshare-net", "--unshare-pid", "--die-with-parent", "true"]
    done = subprocess.run(probe, capture_output=True, text=True, check=False)
    if done.returncode:
        why = (done.stderr or "").strip().splitlines()[-1:] or [f"exit {done.returncode}"]
        return f"bwrap cannot create a sandbox here ({why[0]})"
    return None


def judge_one(
    task: Path,
    answer: Path,
    report: Path,
    timeout: int,
    scale: str = "auto",
    extra: list[str] | None = None,
    sandbox: bool = True,
) -> dict[str, Any]:
    """One answer, in its own process, with a timeout. The report is kept: a rerun reads it
    back."""
    if report.exists():
        return dict(json.loads(report.read_text()))
    out = report.parent.resolve()
    cache = out / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    keep = [task, answer, ROOT, Path(sys.prefix), Path(sys.base_prefix)]
    cmd = sandbox_prefix(out, cache, keep) if sandbox else []
    cmd += [
        sys.executable, "-m", RUNNER, str(task.resolve()), str(answer.resolve()),
        str(report.resolve()), "--scale", scale, *(extra or []),
    ]  # fmt: skip
    path = os.pathsep.join(p for p in (str(ROOT), os.environ.get("PYTHONPATH", "")) if p)
    env = {
        **os.environ,
        "PYTHONPATH": path,
        "TRITON_CACHE_DIR": str(cache),
        "OMP_NUM_THREADS": "1",
    }
    try:
        done = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=out, check=False
        )
        if not report.exists():
            tail = (done.stderr or "").strip().splitlines()[-1:] or [""]
            died = {"answer": answer.name, "verdict": "error", "why": "crash", "message": tail[0]}
            report.write_text(json.dumps(died, indent=1))
    except subprocess.TimeoutExpired:
        slow = {"answer": answer.name, "verdict": "too_slow", "seconds": timeout}
        report.write_text(json.dumps(slow, indent=1))
    return dict(json.loads(report.read_text()))


def judge_item(
    item: dict[str, Any], out: Path, timeout: int, full_max: int, sandbox: bool = True
) -> dict[str, Any]:
    """The scaled pass, and the full pass where the rule above calls for it."""
    task, answer = Path(item["task"]), Path(item["answer"])
    extra = [
        "--precision",
        item.get("precision", "fp32"),
        "--rule",
        item.get("rule", "kernelbench"),
        "--gpu",
        item.get("gpu", ""),  # its shared-memory limit bounds the autotune configs judged
    ]
    small = out / f"{item['id']}.scaled.json"
    scaled = judge_one(task, answer, small, timeout, "auto", extra, sandbox)
    row = {**item, "scaled": scaled}
    note = scaled.get("scaling", {})
    footprint = note.get("footprint_full")
    full = out / f"{item['id']}.full.json"
    if scaled["verdict"] == "no_kernel" and note.get("factor", 1) > 1:
        # PyTorch alone is quick at any size, and an answer that launches its kernel only at the
        # benchmark's own shape (`if x.shape[1] != 256: return torch.sum(...)`) shows it there
        row["full"] = judge_one(task, answer, full, timeout, "none", extra, sandbox)
    elif (
        full_max
        and footprint is not None
        and note.get("factor", 1) > 1
        and footprint <= full_max
        # parameters cost memory more than time: a looser bound for them
        and note.get("params_full", 0) <= 32 * full_max
    ):
        row["full"] = judge_one(task, answer, full, timeout, "none", extra, sandbox)
    return row


def describe(report: dict[str, Any]) -> str:
    """One pass's verdict in a line: for `wrong` the trial and the largest difference, for
    `unsafe` the kind of fault, the kernel and the answer's own line."""
    verdict = str(report["verdict"])
    trials = report.get("trials") or []
    configs = report.get("configs") or {}
    if verdict == "verified":
        diff = max((t.get("max_abs_diff", 0.0) for t in trials), default=0.0)
        text = f"verified: {len(trials)} trial(s), max abs diff {diff:.3g}"
        if configs.get("checked"):
            text += f", and under {configs['checked']} other autotune config(s)"
        return text
    if verdict == "wrong":
        if "failed" in configs:  # right under the autotuner's first config only
            bad = configs["failed"]
            text = f"wrong under autotune config {bad['index']} ({bad['config']})"
        else:
            bad = trials[-1] if trials else {}
            text = f"wrong in trial {len(trials)} (seed {bad.get('seed')})"
        if "shape" in bad:
            want, got = bad["shape"]
            text += f": output shape {got}, the reference's is {want}"
        elif "why" in bad:  # the only one: not a tensor
            text += f": the output is {bad['why']}"
        else:
            diff = bad.get("max_abs_diff", float("nan"))
            text += f": max abs diff {diff:.3g}"
            if math.isnan(diff) and not bad.get("ref_nan"):
                text += " (NaN: the output holds memory nobody wrote)"
        if report.get("not_copied"):  # the v3 harness copies weights by name: these stayed its own
            text += f"; weights not copied by name: {', '.join(report['not_copied'][:4])}"
        return text
    # a fault or an error while the sweep forced a config: which one
    forcing = configs.get("forcing")
    under = f" under autotune config {forcing['index']} ({forcing['config']})" if forcing else ""
    if verdict == "unsafe":
        kind = KINDS.get(str(report.get("kind")), str(report.get("kind")))
        where = f"{report.get('file') or report.get('answer')}:{report.get('line')}"
        buffer = f" (`{report['buffer']}`)" if report.get("buffer") else ""
        source = f": {report['source']}" if report.get("source") else ""
        kernel = f"kernel `{report.get('kernel')}`{buffer}"
        return f"unsafe: {kind} in {kernel}{under}, {where}{source}"
    if verdict == "no_kernel":
        ops = report.get("torch_ops") or {}
        text = "no_kernel: the forward launches no Triton kernel"
        if report.get("launches_init"):
            text += " (it launches at init only)"
        if ops:
            text += "; PyTorch does " + ", ".join(sorted(ops))
        return text
    if verdict == "too_slow":
        return f"too_slow: no verdict within {report.get('seconds')} s"
    stage = report.get("why") or report.get("stage") or ""
    text = f"{verdict} ({stage})" if stage else verdict
    if report.get("message"):
        text += f": {report['message']}"
    if report.get("where"):
        text += f" at {report['where']}"
    return text + under


def lines_of(row: dict[str, Any]) -> list[str]:
    """One line per pass that ran, saying at which shapes."""
    found = []
    if "scaled" in row:
        factor = (row["scaled"].get("scaling") or {}).get("factor", 1)
        at = f"sizes / {factor}" if factor > 1 else "real sizes"
        found.append(f"{describe(row['scaled'])}  [{at}]")
    if "full" in row:
        found.append(f"{describe(row['full'])}  [real sizes]")
    return found


def tally(rows: list[dict[str, Any]]) -> list[str]:
    """Counts per verdict: of the scaled pass (every answer), then of the full pass (where it
    ran)."""
    found = []
    for key in ("scaled", "full"):
        counts = Counter(str(r[key]["verdict"]) for r in rows if key in r)
        if counts:
            each = ", ".join(f"{v} {counts[v]}" for v in VERDICTS if counts[v])
            found.append(f"{key} pass, {sum(counts.values())} answer(s): {each}")
    return found


def load_manifest(manifest: Path) -> list[dict[str, Any]]:
    items = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    for item in items:
        for key in ("task", "answer"):
            item[key] = str(manifest.parent / item[key])  # an absolute path stays as it is
    return items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ttsem judge",
        description="A KernelBench answer written with Triton, judged without a GPU.",
    )
    ap.add_argument("task", type=Path, nargs="?", help="the KernelBench problem file")
    ap.add_argument("answer", type=Path, nargs="?", help="the file that defines ModelNew")
    ap.add_argument("--manifest", type=Path, help="judge every answer of this JSONL file")
    ap.add_argument("--out", type=Path, help="where the reports go (a temporary one otherwise)")
    ap.add_argument("--json", action="store_true", help="print the verdict record as JSON")
    ap.add_argument("--scale", choices=("auto", "none"), default="auto")
    ap.add_argument("--rule", choices=("kernelbench", "v3"), default="kernelbench")
    ap.add_argument("--precision", choices=("fp32", "keep"), default="fp32")
    ap.add_argument("--gpu", default="", help="the dataset's GPU (H100, B200, RTX3090, ...)")
    ap.add_argument("--timeout", type=int, default=None, help="seconds per pass (600; 300 batch)")
    ap.add_argument("--full-max", type=int, default=1 << 20)
    ap.add_argument("--jobs", type=int, default=2, help="answers judged at once (batch)")
    ap.add_argument("--limit", type=int, default=0, help="the first N answers only (batch)")
    ap.add_argument(
        "--no-sandbox",
        action="store_true",
        help="run the answers without bwrap (model-written code, with your permissions)",
    )
    args = ap.parse_args(argv)
    if args.manifest is None and (args.task is None or args.answer is None):
        ap.error("give TASK.py ANSWER.py, or --manifest M.jsonl --out DIR")
    if args.manifest is not None and args.out is None:
        ap.error("--manifest needs --out DIR")
    if args.manifest is not None and args.json:
        ap.error("--json is for one pair; a batch writes its records to OUT/rows.jsonl")
    sandbox = not args.no_sandbox
    if sandbox:
        problem = sandbox_problem()
        if problem is not None:
            sys.stderr.write(NO_BWRAP.format(problem=problem))
            return 2
    else:
        sys.stderr.write(
            "ttsem judge: warning: --no-sandbox: model-written code is about to run unsandboxed, "
            "with your permissions\n"
        )
    if args.manifest is not None:
        return batch(args, sandbox)
    return single(args, sandbox)


def single(args: argparse.Namespace, sandbox: bool) -> int:
    item = {
        "id": args.answer.stem,
        "task": str(args.task),
        "answer": str(args.answer),
        "rule": args.rule,
        "precision": args.precision,
        "gpu": args.gpu,
    }
    timeout = args.timeout or 600
    with contextlib.ExitStack() as stack:
        out = args.out or Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="ttsem-")))
        out.mkdir(parents=True, exist_ok=True)
        for tag in ("scaled", "full"):  # one pair is judged afresh, not read back
            (out / f"{item['id']}.{tag}.json").unlink(missing_ok=True)
        if args.scale == "auto":
            row = judge_item(item, out, timeout, args.full_max, sandbox)
        else:
            extra = ["--precision", args.precision, "--rule", args.rule, "--gpu", args.gpu]
            report = out / f"{item['id']}.full.json"
            full = judge_one(args.task, args.answer, report, timeout, "none", extra, sandbox)
            row = {**item, "full": full}
    if args.json:
        print(json.dumps(row, indent=1))
    else:
        print("\n".join(lines_of(row)))
    passes = [row[key]["verdict"] for key in ("scaled", "full") if key in row]
    return 0 if all(v == "verified" for v in passes) else 1


def batch(args: argparse.Namespace, sandbox: bool) -> int:
    items = load_manifest(args.manifest)
    if args.limit:
        items = items[: args.limit]
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    timeout = args.timeout or 300
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(args.jobs) as pool, (out / "rows.jsonl").open("w") as sink:
        work = functools.partial(
            judge_item, out=out, timeout=timeout, full_max=args.full_max, sandbox=sandbox
        )
        for row in pool.map(work, items):
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            print(f"{row['id']}: " + "; ".join(lines_of(row)), flush=True)
    print("\n".join(tally(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
