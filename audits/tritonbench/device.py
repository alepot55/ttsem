"""The TritonBench audit's control: the same answers on a GPU, judged by value the same way.

    python audits/tritonbench/device.py MANIFEST.jsonl OUT_DIR [--jobs N] [--timeout S] [--limit N]

For each line of a manifest of `manifest.py`, the file the benchmark runs (the answer, the
separator, the task's test: ``ttsem.judge_tritonbench.build``) and the task's own file each run on
the GPU in their own process (`device_runner.py`, no ttsem), inside the judge's `bwrap` sandbox
with the GPU's device nodes let in. As in the judge, an answer's sandbox sees neither the task nor
the reference runs, and the references run once per task on a Triton cache of their own. The
answer's results are then compared with the reference's by the judge's own rules
(``ttsem.judge_tritonbench.decide``), minus what only the semantics can see: no fault is reported
on the device, and there is no second reference run under another poison pattern.

One record per answer in OUT/<id>.device.json (a rerun reads it back), one row per answer in
OUT/rows.jsonl (the manifest's fields and `device`), the counts on the terminal. OUT must not be a
directory the judge writes to: both keep their reference runs in OUT/reference. `agree.py` then
sets the device's verdicts next to the judge's.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import os
import subprocess
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ttsem import judge
from ttsem import judge_tritonbench as tb

RUNNER = Path(__file__).resolve().parent / "device_runner.py"
RECORD = "device"  # the key of a row's record, and the suffix of its file
GPU_NODES = ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools")
NOT_RUN: dict[str, Any] = {"run": "not_run"}


def gpu_nodes() -> list[str]:
    """The `bwrap` arguments that let the GPU's device nodes into the sandbox."""
    nodes = [Path(p) for p in GPU_NODES] + sorted(Path("/dev").glob("nvidia[0-9]*"))
    found: list[str] = []
    for node in nodes:
        if node.exists():
            found += ["--dev-bind", str(node), str(node)]
    return found


def run_file(
    script: Path, report: Path, out: Path, timeout: int, cache: Path, extra: list[str]
) -> dict[str, Any]:
    """`script` run by `device_runner.py` in its own process, with a timeout, in the judge's
    sandbox (OUT the only directory it may write) plus the GPU, `extra` added (what an answer
    must not see). The report is kept: a rerun reads it back."""
    if report.exists():
        return tb.read_report(report, script)
    out = out.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    keep = [script, Path(sys.prefix), Path(sys.base_prefix)]
    cmd = judge.sandbox_prefix(out, cache, keep) + gpu_nodes() + extra
    cmd += ["--ro-bind", str(RUNNER), str(RUNNER)]  # whatever /tmp or `extra` hide
    npz = report.with_suffix(".npz")
    cmd += [sys.executable, "-P", str(RUNNER), str(script.resolve()), str(report.resolve())]
    cmd += [str(npz.resolve())]
    env = {**os.environ, "TRITON_CACHE_DIR": str(cache)}
    try:
        done = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=out, check=False
        )
        if not report.exists():
            tail = (done.stderr or "").strip().splitlines()[-1:] or [""]
            report.write_text(json.dumps({"file": script.name, "run": "crash", "message": tail[0]}))
    except subprocess.TimeoutExpired:
        report.write_text(json.dumps({"file": script.name, "run": "timeout", "seconds": timeout}))
    return tb.read_report(report, script)


_LOCKS: dict[Path, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def reference(task: Path, out: Path, timeout: int) -> tuple[dict[str, Any], Path]:
    """The task file's own run on the GPU, once per task content, in OUT/reference with the
    Triton cache OUT/reference/.cache, which no answer sees: its report and where its arrays are.
    Two answers of one task wait for one run."""
    digest = hashlib.sha256(task.read_bytes()).hexdigest()[:12]
    report = out / "reference" / f"{task.stem}-{digest}.{RECORD}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    cache = (out / "reference" / ".cache").resolve()
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(report.resolve(), threading.Lock())
    with lock:
        run = run_file(task, report, out, timeout, cache, [])
    return run, report.with_suffix(".npz")


def judge_item(item: dict[str, Any], out: Path, timeout: int) -> dict[str, Any]:
    """One answer on the GPU: its record, read back if it is already in OUT."""
    record_path = out / f"{item['id']}.{RECORD}.json"
    kept = tb.read_record(record_path) if record_path.exists() else None
    if kept is not None:
        return {**item, RECORD: kept}
    task, answer = Path(item["task"]), Path(item["answer"])
    script = out / f"{item['id']}.{RECORD}.py"
    try:
        tb.build(task, answer, script)
    except tb.BadTask as exc:
        bad = {"answer": answer.name, "verdict": "not_judged", "class": "bad_task"}
        return {**item, RECORD: {**bad, "launches": 0, "message": str(exc)}}
    gold = reference(task, out, timeout)
    report = script.with_suffix(".run.json")
    hide = tb.answer_sandbox(task, out)
    run = run_file(script, report, out, timeout, (out / ".cache").resolve(), hide)
    random = tb.draws_random(
        tb.operator_part(task.read_text(encoding="utf-8")),
        tb.operator_part(answer.read_text(encoding="utf-8")),
    )

    def no_twin() -> tuple[dict[str, Any], Path]:  # the device has no poison pattern to vary
        return NOT_RUN, out / "reference" / "no-twin.npz"

    try:
        record = tb.decide(run, script.with_suffix(".run.npz"), gold, no_twin, random)
        record = {"answer": answer.name, **record, "seconds": run.get("seconds")}
    except tb.MALFORMED as exc:  # written by the answer's own process, which may write anything
        message = f"malformed run report: {type(exc).__name__}: {exc}"[:300]
        record = {"answer": answer.name, "verdict": "error", "class": "crash", "launches": 0}
        record["message"] = message
    with contextlib.suppress(OSError):
        record_path.write_text(json.dumps(record, indent=1))
    return {**item, RECORD: record}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("manifest", type=Path, help="a manifest of manifest.py")
    ap.add_argument("out", type=Path, help="where the runs and records go")
    ap.add_argument("--jobs", type=int, default=2, help="answers run at once (2)")
    ap.add_argument("--timeout", type=int, default=300, help="seconds per run (300)")
    ap.add_argument("--limit", type=int, default=0, help="the first N answers only")
    args = ap.parse_args(argv)
    problem = judge.sandbox_problem()
    if problem is not None:
        sys.stderr.write(f"device.py: {problem}; the answers run only inside bwrap\n")
        return 2
    if not gpu_nodes():
        sys.stderr.write("device.py: no NVIDIA device node under /dev: this is the GPU control\n")
        return 2
    out: Path = args.out
    if any(out.glob(f"*.{tb.RECORD}.json")):
        sys.stderr.write(f"device.py: {out} holds the judge's records: give the device its own\n")
        return 2
    try:
        items = judge.unique(judge.load_manifest(args.manifest))
    except ValueError as exc:
        sys.stderr.write(f"device.py: {exc}\n")
        return 2
    if args.limit:
        items = items[: args.limit]
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    work = functools.partial(judge_item, out=out, timeout=args.timeout)
    with ThreadPoolExecutor(args.jobs) as pool, (out / "rows.jsonl").open("w") as sink:
        for row in pool.map(work, items):
            records.append(row[RECORD])
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            print(f"{row['id']}: {row[RECORD]['verdict']} ({row[RECORD].get('class')})", flush=True)
    counts = Counter(str(r["verdict"]) for r in records)
    print(f"{len(records)} answer(s): " + ", ".join(f"{v} {n}" for v, n in counts.most_common()))
    print("columns: " + ", ".join(f"{k} {n}" for k, n in tb.columns(records).items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
