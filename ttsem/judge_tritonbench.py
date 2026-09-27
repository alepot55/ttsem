"""``python -m ttsem judge --tritonbench``: a TritonBench answer, judged without a GPU.

    python -m ttsem judge --tritonbench TASK.py ANSWER.py [--json] [--timeout S] [--out DIR]
    python -m ttsem judge --tritonbench --manifest M.jsonl --out DIR [--jobs N] [--timeout S]

A task is a file of TritonBench's own data (`data/TritonBench_G_v1/*.py`,
`data/TritonBench_T_v1/*.py`): the reference operator, a line of 146 `#`, and a test that calls
the operator on its test cases and keeps what it returns. An answer is the code a model wrote for
it, as the benchmark runs it (for G only its imports and functions, for T its last fenced block:
`EVAL/eval_*/0_call_acc.py`); a file that already has the separator and a test after it (the
benchmark's own runnable files) counts only up to the separator. The judge puts the task's test
after the answer, as the benchmark does, and runs both files, the task's and the answer's, each in
its own `bwrap` sandbox (``ttsem._tritonbench_runner``: seed 0, every Triton launch executed by
ttsem on the CPU). TritonBench's own check compares the stdout of the two runs, and its tests
print nothing, so it asks only that the file runs; here what the two tests kept is compared by
value (`same`).

The verdicts are those of ``python -m ttsem judge``, and `class` keeps the finer one:

- verified: the test's results are the reference's within rtol 1e-4 / atol 1e-5 (`verified`) or
  rtol 1e-2 / atol 1e-3 (`verified_lenient`); with `tolerance: reduced`, only within rtol and atol
  5e-2, what a float16 accumulator does to a float32 result (`close_reduced_precision`). The
  absolute tolerance scales with the reference's largest finite value, when that is below 1, down
  to 1e-6.
- wrong: other values, another structure, shape or dtype, or an exception the test swallowed into
  its results (`wrong_result`)
- unsafe: out-of-bounds access, race between program instances, or use of memory nobody wrote,
  with the kernel and its source line (`oob_read`, `oob_write`, `race`, `poison`)
- no_kernel: the results are right, at any of the three tolerances, and no Triton kernel was
  launched (`verified_no_kernel`)
- error: the file raised (`call_error`: a module that exists nowhere is the answer's error too),
  or its process died or left a report that cannot be read (`crash`)
- too_slow: the timeout expired (`timeout`)
- not_judged: something this machine lacks (`environment`: torch or a known package missing, a
  function torch removed, an operator CPU torch does not have, or an error the reference file
  raises word for word too), an op ttsem does not model or its own error (`unsupported`, `error`,
  `tool_error`), the memory cap (`memory`), a task file with no separator (`bad_task`); or a task
  that cannot be compared by value: the reference run is not `ok` or keeps objects that are not
  values (`ran_reference_unjudged`), keeps nothing (`ran_reference_returns_nothing`), two
  reference runs under two poison patterns differ, so it returns memory it never wrote
  (`ran_reference_unwritten`), the task or the answer draws random numbers inside the operator
  (`ran_random_task`), or the answer faults where the task's own test drives the reference into a
  fault too (`fault_also_in_reference`)

The reference runs once per task (and a second time, with the second poison pattern, when an
answer's results are to be compared with it), into OUT/reference/, with a Triton cache of its own
there. An answer's sandbox does not see OUT/reference/, nor the task file and the tree it sits in,
nor the tree OUT sits in but OUT itself (`answer_sandbox`). One pair prints one line and exits 0
only on `verified`. A manifest line is a JSON object with `id`, `task`, `answer` (relative paths
are taken from the manifest's directory) and any labels to carry along; the record goes to
OUT/<id>.tritonbench.json (a rerun reads it back), one row per answer to OUT/rows.jsonl, and the
counts to the terminal, per verdict and in columns (`columns`).
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import io
import json
import os
import re
import site
import subprocess
import sys
import tempfile
import threading
import tokenize
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ttsem import judge

RUNNER = "ttsem._tritonbench_runner"
SEP = "#" * 146
RECORD = "tritonbench"  # the key of a row's record, and the suffix of its file

STRICT = {"rtol": 1e-4, "atol": 1e-5}
LENIENT = {"rtol": 1e-2, "atol": 1e-3}
CLOSE = {"rtol": 5e-2, "atol": 5e-2}  # what a float16 accumulator does to a float32 result
NOISE = 1e-6  # about eight units in the last place of a float32 one
FAULTS = ("oob_read", "oob_write", "race", "poison")
# what a run's report may say besides `ok`, a fault, `call_error`, `crash` and `timeout`: the
# runner's own words for what the tool cannot judge (any other word is a report the runner did
# not write)
UNJUDGED_RUNS = ("unsupported", "error", "memory", "tool_error")
# the task's operator, read as text
RNG = re.compile(r"dropout|tl\.rand|torch\.rand|torch\.bernoulli|random\.|manual_seed|\bseed\b")
# the answer's operator, its code only (`code_only`): what draws random numbers, or reseeds the
# generator the test draws its later inputs from; a name `seed` alone draws nothing
ANSWER_RNG = re.compile(r"dropout|tl\.rand|torch\.rand|torch\.bernoulli|random\.|manual_seed")
ENVIRONMENT = re.compile(
    r"is now removed|not implemented on the CPU|not implemented for 'CPU'|compiled without support"
    r"|OutOfResources|Cannot get CUDA generator"
)
MISSING = re.compile(r"ModuleNotFoundError: No module named '([\w.]+)'")
# the packages that exist and that an answer may rightly import, which this machine may lack; a
# module missing that is none of these (nor torch's) is the answer's own error
KNOWN_PACKAGES = (
    "torch", "flag_gems", "paddle", "vllm", "deepspeed", "fla", "scipy", "jax", "flash_attn",
    "einops", "mamba_ssm", "xformers", "modelscope", "equitriton", "flashnn", "xopes", "apex",
    "transformers",
)  # fmt: skip
# what a report the answer's own process could have written may break when read
MALFORMED = (
    ValueError, KeyError, IndexError, TypeError, AttributeError, RecursionError, OSError,
    zipfile.BadZipFile,
)  # fmt: skip

# the columns (`columns`), by class
UNSAFE = FAULTS
NOT_JUDGED = (
    "unsupported", "timeout", "memory", "tool_error", "crash", "error", "environment", "bad_task",
)  # fmt: skip
NOT_COMPARED = (
    "ran_random_task",
    "ran_reference_unjudged",
    "ran_reference_returns_nothing",
    "ran_reference_unwritten",
    "fault_also_in_reference",
    "verified_no_kernel",
)


class BadTask(ValueError):
    """A task file that is not one of TritonBench's: no separator between operator and test."""


def operator_part(text: str) -> str:
    """What comes before the separator: the operator of a task, or the code of an answer."""
    return text.split(SEP)[0]


def test_part(text: str) -> str:
    return text.split(SEP)[-1]


def task_test(task: Path) -> str:
    """The task's test, what follows its separator."""
    text = task.read_text(encoding="utf-8")
    if SEP not in text:
        raise BadTask(f"{task}: no TritonBench separator (a line of 146 '#')")
    return test_part(text)


_LITERALS = {tokenize.COMMENT, tokenize.STRING} | {
    getattr(tokenize, name)
    for name in ("FSTRING_MIDDLE", "TSTRING_MIDDLE")
    if hasattr(tokenize, name)
}


def code_only(text: str) -> str:
    """`text` without its comments and string literals (a line commented out, a docstring's
    example), its tokens joined so that a dotted name stays whole; `text` itself where it does
    not tokenize."""
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        kept = [t.string for t in tokens if t.type not in _LITERALS]
    except (tokenize.TokenError, SyntaxError):
        return text
    return re.sub(r" ?\. ?", ".", " ".join(kept))


def draws_random(task_code: str, answer_code: str) -> bool:
    """Whether the task's operator or the answer's draws random numbers: the later inputs of the
    test then differ between the two runs. The answer's comments and strings do not count."""
    return bool(RNG.search(task_code) or ANSWER_RNG.search(code_only(answer_code)))


def environment(message: str) -> bool:
    """Whether a `call_error` is something this machine lacks, not the file's own error."""
    missing = MISSING.search(message)
    if missing:
        return missing.group(1).split(".")[0] in KNOWN_PACKAGES
    return ENVIRONMENT.search(message) is not None


def same(gold: Any, pred: Any, tol: dict[str, float]) -> tuple[bool, str]:
    """Whether the results `pred` are the reference's `gold` within `tol`, and where they part
    if not: the same structure, shapes and dtype kinds, and values within the tolerance, whose
    absolute part shrinks with the reference (against a softmax over 2,048 columns, values near
    5e-4, a fixed atol would accept all zeros) and stops at the noise of a float32 cancellation."""
    if isinstance(gold, dict):
        if not isinstance(pred, dict) or set(gold) != set(pred):
            return False, "different test cases"
        for key in gold:
            ok, why = same(gold[key], pred[key], tol)
            if not ok:
                return False, f"{key}: {why}"
        return True, ""
    if isinstance(gold, list):
        if not isinstance(pred, list) or len(gold) != len(pred):
            return False, "different number of outputs"
        for i, (g, p) in enumerate(zip(gold, pred, strict=True)):
            ok, why = same(g, p, tol)
            if not ok:
                return False, f"[{i}] {why}"
        return True, ""
    if isinstance(gold, np.ndarray):
        if not isinstance(pred, np.ndarray):
            return False, "not a tensor"
        if gold.shape != pred.shape:
            return False, f"shape {pred.shape} for {gold.shape}"
        if gold.dtype.kind != pred.dtype.kind:
            return False, f"dtype {pred.dtype} for {gold.dtype}"
        if gold.dtype.kind in "fc":
            finite = np.abs(gold[np.isfinite(gold)])
            scale = min(1.0, float(finite.max())) if finite.size else 1.0
            tol = {"rtol": tol["rtol"], "atol": max(tol["atol"] * scale, NOISE)}
            if np.allclose(pred, gold, equal_nan=True, **tol):
                return True, ""
            bad = ~np.isclose(pred, gold, equal_nan=True, **tol)
            return False, f"{int(bad.sum())} of {gold.size} values differ"
        if np.array_equal(gold, pred):
            return True, ""
        return False, f"{int((gold != pred).sum())} of {gold.size} values differ"
    if isinstance(gold, float) and isinstance(pred, float):
        return bool(np.isclose(pred, gold, equal_nan=True, **tol)), "scalar differs"
    if isinstance(pred, np.ndarray) or type(gold) is not type(pred):
        return False, f"{type(pred).__name__} for {type(gold).__name__}"
    return bool(gold == pred), "value differs"


def results_of(run: dict[str, Any], npz: Path) -> Any:
    """What a run's test kept, rebuilt from its report and its arrays (no pickle)."""
    with np.load(npz, allow_pickle=False) as data:
        arrays = [data[f"arr_{i}"] for i in range(len(data.files))]

    def build(node: Any) -> Any:
        if isinstance(node, dict):
            if "array" in node:
                return arrays[node["array"]]
            if "dict" in node:
                return {k: build(v) for k, v in node["dict"]}
            if "list" in node:
                return [build(v) for v in node["list"]]
            return f"<{node.get('object')}>"
        return node

    return build(run.get("results"))


def opaque(node: Any) -> list[str]:
    """The type names of the objects a report's results keep that are not values (the runner
    writes them as `{"object": name}`), which no comparison can tell apart."""
    if isinstance(node, dict):
        if "object" in node:
            return [str(node["object"])]
        if "dict" in node:
            return [n for pair in node["dict"] for n in opaque(pair[1])]
        if "list" in node:
            return [n for v in node["list"] for n in opaque(v)]
    return []


def read_report(report: Path, script: Path) -> dict[str, Any]:
    """A run's report as the host reads it. One that is not a JSON object (cut short by a kill,
    or written by the code under test, which runs in the runner's process) is rewritten as the
    crash it stands for, so that a rerun does not trip on it again."""
    try:
        found = json.loads(report.read_text(encoding="utf-8"))
    except (ValueError, RecursionError, OSError):
        found = None
    if not isinstance(found, dict):
        found = {"file": script.name, "run": "crash", "message": "unreadable report"}
        with contextlib.suppress(OSError):
            report.write_text(json.dumps(found))
    return found


def run_file(
    script: Path,
    report: Path,
    out: Path,
    timeout: int,
    sandbox: bool,
    cache: Path,
    poison: str = "",
    extra: Iterable[str] = (),
) -> dict[str, Any]:
    """`script` run by ``ttsem._tritonbench_runner`` in its own process, with a timeout, under
    the sandbox of ``ttsem judge`` (OUT the only directory it may write) and `cache` as its
    Triton cache; `extra` is added to the sandbox's command line (what an answer must not see).
    Python leaves the working directory, OUT, off the path (`-P`): a module a run writes there is
    not what a later run imports. The report is kept: a rerun reads it back."""
    if report.exists():
        return read_report(report, script)
    out = out.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = []
    if sandbox:
        keep = [script, judge.ROOT, Path(sys.prefix), Path(sys.base_prefix)]
        cmd = judge.sandbox_prefix(out, cache, keep) + list(extra)
    npz = report.with_suffix(".npz")
    cmd += [sys.executable, "-P", "-m", RUNNER, str(script.resolve()), str(report.resolve())]
    cmd += [str(npz.resolve())]
    path = os.pathsep.join(p for p in (str(judge.ROOT), os.environ.get("PYTHONPATH", "")) if p)
    env = {**os.environ, "PYTHONPATH": path, "TRITON_CACHE_DIR": str(cache), "OMP_NUM_THREADS": "1"}
    env.pop("TTSEM_POISON", None)
    if poison:
        env["TTSEM_POISON"] = poison
    try:
        done = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=out, check=False
        )
        if not report.exists():
            tail = (done.stderr or "").strip().splitlines()[-1:] or [""]
            report.write_text(json.dumps({"file": script.name, "run": "crash", "message": tail[0]}))
    except subprocess.TimeoutExpired:
        report.write_text(json.dumps({"file": script.name, "run": "timeout", "seconds": timeout}))
    return read_report(report, script)


def hidden_tree(path: Path) -> Path | None:
    """The highest directory above `path` that holds nothing the runs need (this package, the
    Python running it, the directories on PYTHONPATH, the home directory), or None: the tree a
    task or an output sits in, hidden from an answer whole (a TritonBench checkout, the outputs
    of earlier runs next to OUT). Under /tmp there is nothing to hide: the sandbox's /tmp is its
    own."""
    needed = [judge.ROOT, Path(sys.prefix), Path(sys.base_prefix), Path.home()]
    needed += [Path(p) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    needed += [Path(p) for p in [*site.getsitepackages(), site.getusersitepackages()]]
    needed = [p.resolve() for p in needed]
    top = None
    for parent in path.resolve().parents:
        if parent.is_relative_to("/tmp") or any(p.is_relative_to(parent) for p in needed):
            break
        top = parent
    return top


def answer_sandbox(task: Path, out: Path) -> list[str]:
    """What an answer's sandbox adds to the judge's: the trees the task and OUT sit in under an
    empty tmpfs (the task's file itself under /dev/null where its tree cannot go), then OUT
    again, writable, and OUT/reference/, the reference runs and their Triton cache, under an
    empty tmpfs. The answer sees neither the reference operator nor its results, and cannot
    leave anything a reference run reads."""
    out = out.resolve()
    cmd: list[str] = []
    trees = {tree for tree in (hidden_tree(task), hidden_tree(out)) if tree is not None}
    for tree in sorted(trees):
        cmd += ["--tmpfs", str(tree)]
    if not any(task.resolve().is_relative_to(tree) for tree in trees):
        cmd += ["--ro-bind", "/dev/null", str(task.resolve())]
    return cmd + ["--bind", str(out), str(out), "--tmpfs", str(out / "reference")]


_LOCKS: dict[Path, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def reference(
    task: Path, out: Path, timeout: int, sandbox: bool, second: bool = False
) -> tuple[dict[str, Any], Path]:
    """The task file's own run (with `second`, under the second poison pattern), once per task
    content, in OUT/reference with the Triton cache OUT/reference/.cache, which no answer sees:
    its report and where its arrays are. Two answers of one task wait for one run."""
    digest = hashlib.sha256(task.read_bytes()).hexdigest()[:12]
    report = out / "reference" / f"{task.stem}-{digest}.{'second' if second else 'first'}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    cache = (out / "reference" / ".cache").resolve()
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(report.resolve(), threading.Lock())
    with lock:
        poison = "second" if second else ""
        run = run_file(task, report, out, timeout, sandbox, cache, poison=poison)
    return run, report.with_suffix(".npz")


def fault_fields(run: dict[str, Any]) -> dict[str, Any]:
    keys = ("kernel", "buffer", "fault_file", "line", "source", "message")
    return {k: run[k] for k in keys if k in run}


def decide(
    run: dict[str, Any],
    answer_npz: Path,
    gold: tuple[dict[str, Any], Path],
    second: Callable[[], tuple[dict[str, Any], Path]],
    random: bool,
) -> dict[str, Any]:
    """The verdict of one answer's run against the reference's: `gold` is the reference's first
    run and where its arrays are, `second` gives the second run the same way (called only when
    the values are to be compared). A report of the answer's that cannot be read raises one of
    `MALFORMED`; one of the reference's leaves the answer unjudged."""
    kind = str(run.get("run"))
    gold_run, gold_npz = gold
    launches = run.get("launches", 0)

    def verdict(value: str, name: str, **extra: Any) -> dict[str, Any]:
        return {"verdict": value, "class": name, "launches": launches, **extra}

    def unjudged(message: str) -> dict[str, Any]:
        return verdict("not_judged", "ran_reference_unjudged", message=message)

    if kind == "ok":
        if gold_run.get("run") != "ok" or not gold_npz.exists():
            return unjudged(f"the reference run is {gold_run.get('run')}")
        objects = opaque(gold_run.get("results"))
        if objects:
            names = ", ".join(sorted(set(objects)))
            return unjudged(f"the reference's test keeps objects that are not values: {names}")
        if random:
            return verdict(
                "not_judged",
                "ran_random_task",
                message="the task or the answer draws random numbers in the operator",
            )
        try:
            want = results_of(gold_run, gold_npz)
        except MALFORMED as exc:
            return unjudged(f"the reference's report cannot be read: {type(exc).__name__}")
        if want is None:
            return verdict(
                "not_judged",
                "ran_reference_returns_nothing",
                message="the reference's test keeps nothing to compare",
            )
        again, again_npz = second()
        if again.get("run") == "ok" and again_npz.exists():
            try:
                twin = results_of(again, again_npz)
            except MALFORMED as exc:
                return unjudged(f"the reference's report cannot be read: {type(exc).__name__}")
            if not same(want, twin, STRICT)[0]:
                return verdict(
                    "not_judged",
                    "ran_reference_unwritten",
                    message="the reference returns memory it never wrote: its runs "
                    "under two poison patterns differ",
                )
        got = results_of(run, answer_npz)
        why = ""
        for name, tolerance, tol in (
            ("verified", "strict", STRICT),
            ("verified_lenient", "lenient", LENIENT),
            ("close_reduced_precision", "reduced", CLOSE),
        ):
            ok, why = same(want, got, tol)
            if ok:
                if not launches:
                    return verdict("no_kernel", "verified_no_kernel", tolerance=tolerance)
                return verdict("verified", name, tolerance=tolerance)
        return verdict("wrong", "wrong_result", message=why)
    if kind in FAULTS:
        if gold_run.get("run") in FAULTS:  # the task's own test drives the reference into one
            return verdict(
                "not_judged",
                "fault_also_in_reference",
                kind=kind,
                reference=gold_run.get("run"),
                **fault_fields(run),
            )
        return verdict("unsafe", kind, kind=kind, **fault_fields(run))
    message = str(run.get("message", ""))
    where = {k: run[k] for k in ("detail", "where") if k in run}
    if kind == "call_error":
        if environment(message):
            return verdict("not_judged", "environment", message=message, **where)
        if gold_run.get("run") == "call_error" and gold_run.get("message") == message:
            # the reference file raises it too: the task's test, or this machine, not the answer
            return verdict("not_judged", "environment", message=message, reference=kind, **where)
        return verdict("error", "call_error", message=message, **where)
    if kind == "crash":
        return verdict("error", "crash", message=message)
    if kind == "timeout":
        return verdict("too_slow", "timeout", seconds=run.get("seconds"))
    if kind in UNJUDGED_RUNS:
        return verdict("not_judged", kind, message=message, **where)
    raise ValueError(f"a run the runner does not report: {kind[:40]!r}")


def build(task: Path, answer: Path, dest: Path) -> None:
    """The file TritonBench runs for an answer: its code, the separator, the task's test."""
    test = task_test(task)
    code = operator_part(answer.read_text(encoding="utf-8"))
    body = code.rstrip("\n") + "\n" + SEP + "\n" + test
    if not dest.exists() or dest.read_text(encoding="utf-8") != body:
        dest.write_text(body, encoding="utf-8")


def read_record(path: Path) -> dict[str, Any] | None:
    """A record judged before, or None where there is none that reads (cut short by a kill)."""
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, RecursionError, OSError):
        return None
    return found if isinstance(found, dict) and "verdict" in found else None


def judge_item(item: dict[str, Any], out: Path, timeout: int, sandbox: bool) -> dict[str, Any]:
    """One answer: its record, read back if it is already in OUT."""
    record_path = out / f"{item['id']}.{RECORD}.json"
    kept = read_record(record_path) if record_path.exists() else None
    if kept is not None:
        return {**item, RECORD: kept}
    task, answer = Path(item["task"]), Path(item["answer"])
    script = out / f"{item['id']}.{RECORD}.py"
    try:
        build(task, answer, script)
    except BadTask as exc:
        bad = {"answer": answer.name, "verdict": "not_judged", "class": "bad_task"}
        return {**item, RECORD: {**bad, "launches": 0, "message": str(exc)}}
    gold = reference(task, out, timeout, sandbox)
    report = script.with_suffix(".run.json")
    cache = (out / ".cache").resolve()
    hide = answer_sandbox(task, out) if sandbox else []
    run = run_file(script, report, out, timeout, sandbox, cache, extra=hide)
    random = draws_random(
        operator_part(task.read_text(encoding="utf-8")),
        operator_part(answer.read_text(encoding="utf-8")),
    )
    second = functools.partial(reference, task, out, timeout, sandbox, True)
    try:
        record = decide(run, script.with_suffix(".run.npz"), gold, second, random)
        record = {"answer": answer.name, **record, "seconds": run.get("seconds")}
        # a line of the file run is the answer's own, or of the task's test after it
        lines = operator_part(answer.read_text(encoding="utf-8")).rstrip("\n").count("\n") + 1
        owner = {True: answer.name, False: f"{task.name} (its test)"}
        if record.pop("fault_file", None) == script.name:
            record["file"] = owner[int(record["line"]) <= lines]
        file, _, line = str(record.get("where", "")).rpartition(":")
        if file == script.name and line.isdigit():
            record["where"] = f"{owner[int(line) <= lines]}:{line}"
    except MALFORMED as exc:  # written by the answer's own process, which may write anything
        message = f"malformed run report: {type(exc).__name__}: {exc}"[:300]
        record = {"answer": answer.name, "verdict": "error", "class": "crash", "launches": 0}
        record["message"] = message
    with contextlib.suppress(OSError):
        record_path.write_text(json.dumps(record, indent=1))
    return {**item, RECORD: record}


def describe(record: dict[str, Any]) -> str:
    """One answer's verdict in a line."""
    verdict, name = str(record["verdict"]), str(record.get("class"))
    if verdict == "verified":
        if record.get("tolerance") == "reduced":
            return (
                "verified at reduced precision: the test's results are the reference's only "
                "within rtol = atol = 5e-2 (a float16 accumulator)"
            )
        tol = STRICT if record.get("tolerance") == "strict" else LENIENT
        return f"verified: the test's results are the reference's (rtol {tol['rtol']:g})"
    if verdict == "wrong":
        return f"wrong: the test's results differ from the reference's: {record.get('message')}"
    if verdict == "no_kernel":
        return "no_kernel: the test's results are right and no Triton kernel was launched"
    if verdict == "too_slow":
        return f"too_slow: no verdict within {record.get('seconds')} s"
    if name in FAULTS:
        kind = judge.KINDS.get(name, name)
        buffer = f" (`{record['buffer']}`)" if record.get("buffer") else ""
        source = f": {record['source']}" if record.get("source") else ""
        text = f"{kind} in kernel `{record.get('kernel')}`{buffer}, "
        text += f"{record.get('file')}:{record.get('line')}{source}"
        if verdict == "unsafe":
            return f"unsafe: {text}"
        return f"not_judged ({name}): {text}; the task's own test faults the reference too"
    text = f"{verdict} ({name})"
    if record.get("message"):
        text += f": {record['message']}"
    if record.get("detail") and record["detail"] not in str(record.get("message")):
        text += f" ({record['detail']})"  # a compilation error says what went wrong last
    if record.get("where"):
        text += f" at {record['where']}"
    if record.get("reference") == "call_error":
        text += "; the reference file raises it too"
    return text


def columns(records: list[dict[str, Any]]) -> dict[str, int]:
    """The answers in columns: answers; verified (strict or lenient); reduced precision; wrong;
    unsafe; not compared (a random task, a reference that cannot be compared with, a fault the
    task's test causes in the reference too, right values with no kernel); not judged (what the
    machine or the tool lacks, a timeout, a crash); ran (the file ran to the end, or stopped for
    a fault a GPU runs past: all but the last two and the answers that raised)."""
    tally = Counter(str(r.get("class")) for r in records)
    count = {
        "answers": len(records),
        "verified": tally["verified"] + tally["verified_lenient"],
        "reduced": tally["close_reduced_precision"],
        "wrong": tally["wrong_result"],
        "unsafe": sum(tally[c] for c in UNSAFE),
        "not_compared": sum(tally[c] for c in NOT_COMPARED),
        "not_judged": sum(tally[c] for c in NOT_JUDGED),
    }
    count["ran"] = sum(count[k] for k in ("verified", "reduced", "wrong", "unsafe", "not_compared"))
    return count


def tally(rows: list[dict[str, Any]]) -> list[str]:
    records = [r[RECORD] for r in rows]
    counts = Counter(str(r["verdict"]) for r in records)
    found = [
        f"{len(records)} answer(s): "
        + ", ".join(f"{v} {counts[v]}" for v in judge.VERDICTS if counts[v])
    ]
    col = columns(records)
    found.append("columns: " + ", ".join(f"{k} {n}" for k, n in col.items()))
    judged = col["verified"] + col["reduced"] + col["wrong"] + col["unsafe"]
    if judged:
        bad = col["wrong"] + col["unsafe"]
        found.append(
            f"of the {judged} that ran and could be judged, {bad} ({100 * bad / judged:.0f}%) "
            "return wrong values or are unsafe"
        )
    return found


def single(args: argparse.Namespace, sandbox: bool) -> int:
    item = {"id": args.answer.stem, "task": str(args.task), "answer": str(args.answer)}
    try:
        task_test(args.task)
    except (BadTask, OSError) as exc:
        sys.stderr.write(f"ttsem judge: {exc}\n")
        return 2
    timeout = args.timeout or 600
    with contextlib.ExitStack() as stack:
        out = args.out or Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="ttsem-")))
        out.mkdir(parents=True, exist_ok=True)
        for stale in out.glob(f"{item['id']}.{RECORD}.*"):  # one pair is judged afresh
            stale.unlink()
        row = judge_item(item, out, timeout, sandbox)
    if args.json:
        print(json.dumps(row, indent=1))
    else:
        print(describe(row[RECORD]))
    return 0 if row[RECORD]["verdict"] == "verified" else 1


def batch(args: argparse.Namespace, sandbox: bool) -> int:
    items = judge.load_manifest(args.manifest)
    try:
        items = judge.unique(items)
    except ValueError as exc:
        sys.stderr.write(f"ttsem judge: {exc}\n")
        return 2
    if args.limit:
        items = items[: args.limit]
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    timeout = args.timeout or 300
    rows: list[dict[str, Any]] = []
    work = functools.partial(judge_item, out=out, timeout=timeout, sandbox=sandbox)
    with ThreadPoolExecutor(args.jobs) as pool, (out / "rows.jsonl").open("w") as sink:
        for row in pool.map(work, items):
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            print(f"{row['id']}: {describe(row[RECORD])}", flush=True)
    print("\n".join(tally(rows)))
    return 0
