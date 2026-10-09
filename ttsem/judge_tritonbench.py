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
  its results (`wrong_result`). Memory nobody wrote is filled with NaN, so a kernel that reads it
  shows up here, as NaN in the values
- unsafe: an out-of-bounds access or a race between program instances (at the kernel's line), or
  an operation with an undefined result (a division by zero, MIN / -1, two lanes of one store
  writing different values), with the kernel and its source line (`oob_read`, `oob_write`, `race`,
  `poison`)
- no_kernel: the results are right, at any of the three tolerances, and no Triton kernel was
  launched (`verified_no_kernel`)
- error: the file raised (`call_error`: a module that exists nowhere is the answer's error too),
  or its process died or left a report that cannot be read (`crash`)
- too_slow: the timeout expired (`timeout`)
- not_judged: something this machine lacks (`environment`: torch or a known package missing, a
  function torch removed, an operator CPU torch does not have, or an error the reference file
  raises word for word too), an op ttsem does not model or its own error (`unsupported`, `error`,
  `tool_error`), the memory cap (`memory`), a task file with no separator (`bad_task`), the answer
  reaching into the judge or not finishing through it (`tampered`); or a task that cannot be
  compared by value: the reference run is not `ok` or keeps objects that are not values
  (`ran_reference_unjudged`), keeps nothing (`ran_reference_returns_nothing`), two reference runs
  under two poison patterns differ, so it returns memory it never wrote (`ran_reference_unwritten`),
  the task or the answer draws random numbers inside the operator (`ran_random_task`), the
  answer faults where the task's own test drives the reference into a fault too
  (`fault_also_in_reference`), or its results are right but a launch was too large to be checked
  for races between program instances (`races_unchecked`: more memory accesses than
  `ttsem.memory.ACCESS_BUDGET`, and address ranges of two instances that meet); such a run is
  never `verified`, and is `wrong` or `unsafe` still where its values differ or a fault was found

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
import re
import sys
import tempfile
import tokenize
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ttsem import _guard, judge
from ttsem.judge_io import Ran, load_npz, printable, read_json, run_sandboxed

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
UNJUDGED_RUNS = ("unsupported", "error", "memory", "tool_error", "tampered")
# what a run that stopped at a launch the semantics cannot judge says (`sanitize.Unjudged`): a
# launch the semantics does not model, or a file that swapped the checker's own code (`tampered`)
STOPPED_RUNS = ("unsupported", "error", "tampered")
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
MALFORMED = judge.MALFORMED

# the columns (`columns`), by class
UNSAFE = FAULTS
NOT_JUDGED = (
    "unsupported", "timeout", "memory", "tool_error", "crash", "error", "environment", "bad_task",
    "tampered", "forged",
)  # fmt: skip
NOT_COMPARED = (
    "ran_random_task",
    "ran_reference_unjudged",
    "ran_reference_returns_nothing",
    "ran_reference_unwritten",
    "fault_also_in_reference",
    "verified_no_kernel",
    "races_unchecked",
)


class BadTask(ValueError):
    """A task file that is not one of TritonBench's: no separator between operator and test."""


def tamper_reason(answer: Path) -> str | None:
    """Why this answer must not be run, read off its code (before the separator) without running
    it (`_guard`), or None: an answer that reaches into the judge is `not_judged`, before it runs.
    Only the answer's own code is scanned; the task's test, added after the separator, is the
    benchmark's."""
    try:
        source = operator_part(answer.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None
    reason = _guard.tamper_scan(source, judge.FORBIDDEN_MODULES)
    return None if reason is None else f"the answer reaches into the judge: {reason}"


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
    """What a run's test kept, rebuilt from its report and its arrays (no pickle, every array
    read as untrusted: `judge_io.load_npz`)."""
    data = load_npz(npz)
    arrays = [data[f"arr_{i}"] for i in range(len(data))]

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


def run_report(ran: Ran, work: Path, script: Path, timeout: int) -> dict[str, Any]:
    """A run's report as the judge takes it: the `stop` the run sent (a fault, or a launch the
    semantics cannot judge: sent before the code under test could catch it), else the report it
    wrote if it said `done`, else what it stands for: a timeout, or a crash (a process that died,
    or a report that is not a JSON object)."""
    stop = ran.stop()
    if stop is not None:
        fields = ("run", "kernel", "buffer", "fault_file", "line", "source", "message", "ops")
        found = {key: stop[key] for key in fields if key in stop}
        if found.get("run") not in (*FAULTS, *STOPPED_RUNS):
            return {"file": script.name, "run": "crash", "message": "a stop that says no fault"}
        counts = {key: stop.get(key, 0) for key in ("launches", "races_unchecked")}
        return {"file": script.name, **found, **counts}
    if ran.code is None:
        return {"file": script.name, "run": "timeout", "seconds": timeout}
    found = read_json(work / "run.json")
    if not ran.done():
        if isinstance(found, dict):
            # a report is there, but the runner's own token-bearing `done` never arrived: the file
            # left the process itself (os._exit), having written or forged the report
            return {
                "file": script.name, "run": "tampered",
                "message": "the run wrote a report but did not finish through the judge",
            }  # fmt: skip
        return {"file": script.name, "run": "crash", "message": ran.tail}
    if not isinstance(found, dict):
        return {"file": script.name, "run": "crash", "message": "unreadable report"}
    confirmed = ran.confirmed_launches()
    claimed = found.get("launches")
    if confirmed is not None and isinstance(claimed, int) and claimed != confirmed:
        return {
            "file": script.name, "run": "tampered",
            "message": f"the run claims {claimed} launch(es); the judge counted {confirmed} over "
            "its own channel",
        }  # fmt: skip
    skipped, said = ran.confirmed_unchecked(), found.get("races_unchecked")
    if skipped is not None and isinstance(said, int) and said != skipped:
        return {
            "file": script.name, "run": "tampered",
            "message": f"the run claims {said} launch(es) too large to be checked for races; "
            f"the judge counted {skipped} over its own channel",
        }  # fmt: skip
    # the count the judge trusts: the channel's, or, from a runner that sent none, the report's
    return {**found, "races_unchecked": judge.unchecked_count(found, ran)}


def run_file(
    script: Path,
    work: Path,
    timeout: int,
    sandbox: bool,
    poison: str = "",
    hide: Sequence[Path] = (),
    extra: Iterable[str] = (),
    strict: bool = False,
) -> dict[str, Any]:
    """`script` run by ``ttsem._tritonbench_runner`` in its own process, with a timeout, under
    the sandbox of ``ttsem judge``: `work` is its own directory, the only one it may write (its
    report, its arrays and its Triton cache go there), `hide` is hidden from it besides the
    judge's own, and `extra` is added to the sandbox's command line. Python leaves the working
    directory, `work`, off the path (`-P`): a module the run writes there is not what the runner
    imports. Returns the run's report (`run_report`); its arrays are in work/run.npz."""
    work = work.resolve()
    cache = work / ".cache"
    cache.mkdir(parents=True, exist_ok=True)  # a run may write there first
    # the caller's environment stays out (an answer could copy a token from it into its
    # report), and so does its TTSEM_POISON: the pattern is the run's own
    env = judge.run_env(cache, sandbox, {"TTSEM_POISON": poison} if poison else None)
    cmd: list[str] = []
    if sandbox:
        cmd = judge.sandbox_prefix(work, cache, [script], env, hide, strict) + list(extra)
    cmd += [sys.executable, "-P", "-m", RUNNER, str(script.resolve())]
    cmd += [str(work / "run.json"), str(work / "run.npz")]
    return run_report(run_sandboxed(cmd, env, work, timeout), work, script, timeout)


def answer_sandbox(task: Path, out: Path) -> tuple[list[Path], list[str]]:
    """What an answer's sandbox hides besides the judge's own: the trees the task and OUT sit in
    (`judge.hidden_tree`; OUT itself where its tree holds what the runs need), so the answer sees
    neither the reference operator, nor the reference runs and their results, nor the other
    answers' runs; and the task's file itself, under /dev/null, where no hidden tree holds it.
    Returns (the directories to hide, the arguments that cover the file)."""
    trees = {tree for tree in (judge.hidden_tree(task), *judge.out_tree(out)) if tree is not None}
    extra: list[str] = []
    if not any(task.resolve().is_relative_to(tree) for tree in trees):
        extra = ["--ro-bind", "/dev/null", str(task.resolve())]
    return sorted(trees), extra


def reference(
    task: Path, out: Path, timeout: int, sandbox: bool, second: bool = False, strict: bool = False
) -> tuple[dict[str, Any], Path]:
    """The task file's own run (with `second`, under the second poison pattern), once per task
    content, in OUT/reference/<task>-<digest>.<first|second>/ with its own Triton cache, which no
    answer sees: its report and where its arrays are. The report is kept by this process in
    OUT/reference/<task>-<digest>.<first|second>.json, and a rerun reads it back. Two answers of
    one task wait for one run."""
    digest = hashlib.sha256(task.read_bytes()).hexdigest()[:12]
    name = f"{task.stem}-{digest}.{'second' if second else 'first'}"
    report = out / judge.REFERENCE / f"{name}.json"
    work = out / judge.REFERENCE / name
    with judge.locked(report):
        run = read_json(report)
        if not isinstance(run, dict):
            judge.fresh(work)
            poison = "second" if second else ""
            run = run_file(task, work, timeout, sandbox, poison, judge.out_tree(out), (), strict)
            report.write_text(json.dumps(run))
    return run, work / "run.npz"


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
    unchecked = run.get("races_unchecked", 0)
    unchecked = unchecked if isinstance(unchecked, int) and unchecked > 0 else 0
    counts = {"launches": launches} | ({"races_unchecked": unchecked} if unchecked else {})

    def verdict(value: str, name: str, **extra: Any) -> dict[str, Any]:
        return {"verdict": value, "class": name, **counts, **extra}

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
                if unchecked:  # right values, but a race was not looked for
                    message = judge.unchecked_message(unchecked)
                    return verdict(
                        "not_judged", "races_unchecked", tolerance=tolerance, message=message
                    )
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
    """A record judged before (this process's own: no run writes in OUT), or None where there
    is none that reads (cut short by a kill)."""
    found = read_json(path)
    return found if isinstance(found, dict) and "verdict" in found else None


def record_of(
    run: dict[str, Any],
    npz: Path,
    gold: tuple[dict[str, Any], Path],
    second: Callable[[], tuple[dict[str, Any], Path]],
    random: bool,
    task: Path,
    answer: Path,
    script: str,
) -> dict[str, Any]:
    """An answer's record from its run report (`decide`), the lines of the file it ran put back
    on the answer's own file or the task's test (`script` is that file's name). A report the
    answer's process broke (it may write anything) is its crash."""
    try:
        record = decide(run, npz, gold, second, random)
        record = {"answer": answer.name, **record, "seconds": run.get("seconds")}
        # a line of the file run is the answer's own, or of the task's test after it
        lines = operator_part(answer.read_text(encoding="utf-8")).rstrip("\n").count("\n") + 1
        owner = {True: answer.name, False: f"{task.name} (its test)"}
        if record.pop("fault_file", None) == script:
            record["file"] = owner[int(record["line"]) <= lines]
        file, _, line = str(record.get("where", "")).rpartition(":")
        if file == script and line.isdigit():
            record["where"] = f"{owner[int(line) <= lines]}:{line}"
    except MALFORMED as exc:  # written by the answer's own process, which may write anything
        message = f"malformed run report: {type(exc).__name__}: {exc}"[:300]
        record = {"answer": answer.name, "verdict": "error", "class": "crash", "launches": 0}
        record["message"] = message
    return record


def judge_item(
    item: dict[str, Any], out: Path, timeout: int, sandbox: bool, strict: bool = False
) -> dict[str, Any]:
    """One answer: its record, read back if it is already in OUT. The answer runs in a directory
    of its own, OUT/work/<id>.tritonbench/, the only one its sandbox may write, with the file
    TritonBench runs for it (`build`); no other run sees it."""
    record_path = out / f"{item['id']}.{RECORD}.json"
    kept = read_record(record_path)
    if kept is not None:
        return {**item, RECORD: kept}
    task, answer = Path(item["task"]), Path(item["answer"])
    reason = tamper_reason(answer)
    if reason is not None:
        bad = {"answer": answer.name, "verdict": "not_judged", "class": "tampered"}
        record = {**bad, "launches": 0, "message": reason}
        with contextlib.suppress(OSError):
            record_path.write_text(json.dumps(record, indent=1))
        return {**item, RECORD: record}
    work = judge.fresh(out / judge.WORK / f"{item['id']}.{RECORD}")
    script = work / f"{item['id']}.{RECORD}.py"
    try:
        build(task, answer, script)
    except BadTask as exc:
        bad = {"answer": answer.name, "verdict": "not_judged", "class": "bad_task"}
        return {**item, RECORD: {**bad, "launches": 0, "message": str(exc)}}
    gold = reference(task, out, timeout, sandbox, strict=strict)
    hide, extra = answer_sandbox(task, out) if sandbox else ([], [])
    run = run_file(script, work, timeout, sandbox, "", hide, extra, strict)
    random = draws_random(
        operator_part(task.read_text(encoding="utf-8")),
        operator_part(answer.read_text(encoding="utf-8")),
    )
    second = functools.partial(reference, task, out, timeout, sandbox, True, strict)
    record = record_of(run, work / "run.npz", gold, second, random, task, answer, script.name)
    judge.tidy(work)
    with contextlib.suppress(OSError):
        record_path.write_text(json.dumps(record, indent=1))
    return {**item, RECORD: record}


def describe(record: dict[str, Any]) -> str:
    """One answer's verdict in a line, and, for a run with launches too large to be checked for
    races, how many."""
    text = verdict_line(record)
    n = record.get("races_unchecked")
    if isinstance(n, int) and n > 0 and record.get("class") != "races_unchecked":
        text += f"; {n} launch(es) too large to be checked for races"
    return text


def verdict_line(record: dict[str, Any]) -> str:
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
    task's test causes in the reference too, right values with no kernel or from a launch too large
    to be checked for races); not judged (what the
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
    strict = bool(getattr(args, "strict", False))
    with contextlib.ExitStack() as stack:
        out = args.out or Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="ttsem-")))
        out.mkdir(parents=True, exist_ok=True)
        for stale in out.glob(f"{item['id']}.{RECORD}.*"):  # one pair is judged afresh
            stale.unlink()
        row = judge_item(item, out, timeout, sandbox, strict)
    if args.json:
        print(json.dumps(row, indent=1))
    else:
        print(printable(describe(row[RECORD])))
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
    strict = bool(getattr(args, "strict", False))
    rows: list[dict[str, Any]] = []
    work = functools.partial(judge_item, out=out, timeout=timeout, sandbox=sandbox, strict=strict)
    with ThreadPoolExecutor(args.jobs) as pool, (out / "rows.jsonl").open("w") as sink:
        for row in pool.map(work, items):
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            print(printable(f"{row['id']}: {describe(row[RECORD])}"), flush=True)
    print("\n".join(tally(rows)))
    return 0
