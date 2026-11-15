"""The reward's worker: one process in the judge's own venv (ttsem with its `triton` extra: Triton
and torch CPU) that judges one pass per request, every answer and every reference in its own
`bwrap` sandbox, as `python -m ttsem judge` does.

    TTSEM_REWARD_TTSEM_ROOT=<a ttsem checkout> python -I worker.py [--jobs N] [--mem-gb G]

`Judge` starts it by this file's path with the interpreter of `TTSEM_REWARD_PYTHON`, so the venv
needs `ttsem.judge` and its dependencies, not `ttsem.reward`. The worker reads one JSON request
per line on stdin and writes one JSON reply per line on stdout, in the order the passes finish
(each reply carries its request's `id`). Trusted code of ours: it never imports a task or an
answer, it hands their paths to `ttsem.judge`, whose runner imports them inside the sandbox. The
trainer's process imports neither torch nor Triton: this one does, through the judge, in its own
venv.

Requests (`op`):

- `hello`: the versions and whether the sandbox works here.
- `kernelbench`: one pass of `ttsem.judge.judge_one` over a KernelBench-style task (`task`,
  `answer`, `out`, `report`, `scale`, `budget`, `trials`, `settings`, `timeout`, `strict`). The
  runner flags `--trials` and `--budget`, which `judge_one` does not expose, are added to this
  thread's sandboxed commands only, the reference run's too, so the reference and the answer
  always run at the same sizes. `out` must be one directory per (budget, trials): the judge keys
  its reference cache by task, scale, rule and precision only.
- `tritonbench`: `ttsem.judge_tritonbench.judge_item` over a TritonBench task and answer.

Inside every sandbox the runner is started by a few lines of ours (`METER`) that wait for it and
send the CPU seconds of its process tree (user and system time of every descendant waited for) as
one more event on the judge's pipe, `{"event": "cpu"}`, which the judge's own reading ignores (it
carries no token). A reply gives each pass's wall time and its process-seconds (`answer_cpu_s`,
`reference_cpu_s`): the second stays honest on a machine shared with other work. It is telemetry,
not a verdict: an answer could forge the event, never the verdict. A run killed at its timeout
counts its wall time.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT_ENV = "TTSEM_REWARD_TTSEM_ROOT"
# started by the sandbox in place of the runner: runs it, then reports its tree's CPU seconds
METER = (
    "import json,os,resource,subprocess,sys\n"
    "fd=int(sys.argv[sys.argv.index('--events')+1])\n"
    "code=subprocess.run(sys.argv[1:],close_fds=False).returncode\n"
    "u=resource.getrusage(resource.RUSAGE_CHILDREN)\n"
    "os.write(fd,(json.dumps({'event':'cpu','cpu_s':u.ru_utime+u.ru_stime})+'\\n').encode())\n"
    "os.kill(os.getpid(),-code) if code<0 else sys.exit(code)\n"
)

_LOCAL = threading.local()


def metered(cmd: Sequence[str], runners: Sequence[Sequence[str]]) -> list[str] | None:
    """`cmd` with the runner's own command started through `METER`, or None where `cmd` holds
    none of `runners` (the judge's command for each runner: `python_module(executable, runner)`,
    which is `python -P -m RUNNER` on 3.11+ and a `-c` bootstrap on 3.10)."""
    cmd = list(cmd)
    for at in range(len(cmd)):
        if any(cmd[at : at + len(runner)] == list(runner) for runner in runners):
            return [*cmd[:at], sys.executable, "-I", "-c", METER, *cmd[at:]]
    return None


def _import_judge() -> tuple[Any, Any]:
    """`ttsem.judge` and `ttsem.judge_tritonbench`, from TTSEM_REWARD_TTSEM_ROOT when it is set
    (a checkout of ttsem ahead of whatever the venv installs), with the thread-local runner flags
    and the CPU meter installed."""
    root = os.environ.get(ROOT_ENV, "")
    if root:
        sys.path.insert(0, str(Path(root).resolve()))
    import ttsem.judge as judge
    import ttsem.judge_tritonbench as tritonbench
    from ttsem._compat import python_module

    original = judge.run_sandboxed  # the same function both modules import from judge_io
    runners = [python_module(sys.executable, m.RUNNER) for m in (judge, tritonbench)]

    def run_sandboxed(cmd: Sequence[str], env: Any, cwd: Path, timeout: int) -> Any:
        start = time.monotonic()
        # a command without the runner in it runs unmetered, and counts its wall time
        wrapped = metered(cmd, runners) or list(cmd)
        ran = original([*wrapped, *getattr(_LOCAL, "flags", ())], env, cwd, timeout)
        sent = [e.get("cpu_s") for e in ran.events if e.get("event") == "cpu"]
        cpu = sent[-1] if sent and isinstance(sent[-1], (int, float)) else None
        # killed at the timeout, or a run that never reported: its wall time
        _LOCAL.cpu = getattr(_LOCAL, "cpu", 0.0) + (
            float(cpu) if cpu is not None else time.monotonic() - start
        )
        return ran

    # judge_one and reference_run call it by this name, judge_tritonbench by its own
    judge.run_sandboxed = run_sandboxed
    tritonbench.run_sandboxed = run_sandboxed
    return judge, tritonbench


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _timed(call: Callable[[], Any]) -> tuple[Any, float, float]:
    """`call()`, its wall seconds, and the CPU seconds of the sandboxed runs it made."""
    _LOCAL.cpu = 0.0
    start = time.monotonic()
    value = call()
    return value, round(time.monotonic() - start, 2), round(_LOCAL.cpu, 2)


class Worker:
    def __init__(self, judge: Any, tritonbench: Any) -> None:
        self.judge = judge
        self.tb = tritonbench  # ttsem.judge_tritonbench (the method `tritonbench` uses it)

    def hello(self, request: dict[str, Any]) -> dict[str, Any]:
        strict = bool(request.get("strict", True))
        return {
            "ttsem": _version("ttsem"),
            "root": str(self.judge.ROOT),
            "triton": _version("triton"),
            "torch": _version("torch"),
            "python": sys.version.split()[0],
            "sandbox_problem": self.judge.sandbox_problem(strict)
            or self._interpreter_problem(strict),
        }

    def _interpreter_problem(self, strict: bool) -> str | None:
        """The sandbox starts, but does this worker's own Python start inside it? A venv whose
        interpreter is a symlink into a directory the sandbox hides (a Python that a tool keeps
        under the home) does not: every run would then fail and, with `on_abstain="keep"`, the
        gate would pass everything without saying so."""
        with tempfile.TemporaryDirectory(prefix="ttsem-probe-") as tmp:
            work = Path(tmp)
            env = {"PATH": os.environ.get("PATH", os.defpath)}
            cmd = self.judge.sandbox_prefix(work, work, [], env, strict=strict)
            cmd += [sys.executable, "-c", "pass"]
            done = subprocess.run(
                cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False
            )
        if done.returncode:
            why = (done.stderr or "").strip().splitlines()[-1:] or [f"exit {done.returncode}"]
            return (
                f"the judge's Python ({sys.executable}) cannot start in the sandbox ({why[0]}): "
                "make the judge's venv from an interpreter that sits outside the home directory"
            )
        return None

    def kernelbench(self, request: dict[str, Any]) -> dict[str, Any]:
        judge = self.judge
        task, answer = Path(request["task"]), Path(request["answer"])
        out = Path(request["out"])
        out.mkdir(parents=True, exist_ok=True)
        settings = {**judge.DEFAULTS, **request.get("settings", {})}
        scale, timeout = str(request["scale"]), int(request["timeout"])
        strict = bool(request.get("strict", True))
        _LOCAL.flags = ("--trials", str(int(request["trials"])), "--budget", str(request["budget"]))
        try:
            # the reference first, timed apart: judge_one reads it back from OUT/reference/
            _, reference_s, reference_cpu = _timed(
                lambda: judge.reference_run(task, out, scale, settings, timeout, True, strict)
            )
            report = out / f"{request['report']}.json"
            record, answer_s, answer_cpu = _timed(
                lambda: judge.judge_one(
                    task, answer, report, timeout, scale, settings, True, strict
                )
            )
        finally:
            _LOCAL.flags = ()
        return {
            "record": record,
            "line": judge.describe(record),
            "answer_s": answer_s,
            "reference_s": reference_s,
            "answer_cpu_s": answer_cpu,
            "reference_cpu_s": reference_cpu,
        }

    def tritonbench(self, request: dict[str, Any]) -> dict[str, Any]:
        tb = self.tb
        task, out = Path(request["task"]), Path(request["out"])
        out.mkdir(parents=True, exist_ok=True)
        timeout, strict = int(request["timeout"]), bool(request.get("strict", True))
        _, reference_s, reference_cpu = _timed(
            lambda: tb.reference(task, out, timeout, True, strict=strict)
        )
        item = {"id": request["report"], "task": str(task), "answer": request["answer"]}
        row, answer_s, answer_cpu = _timed(lambda: tb.judge_item(item, out, timeout, True, strict))
        record = row[tb.RECORD]
        return {
            "record": record,
            "line": tb.describe(record),
            "answer_s": answer_s,
            "reference_s": reference_s,
            "answer_cpu_s": answer_cpu,
            "reference_cpu_s": reference_cpu,
        }

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("op")
        handler = {
            "hello": self.hello,
            "kernelbench": self.kernelbench,
            "tritonbench": self.tritonbench,
        }.get(str(op))
        if handler is None:
            return {"id": request.get("id"), "error": f"unknown op {op!r}"}
        try:
            return {"id": request.get("id"), **handler(request)}
        except Exception as exc:  # noqa: BLE001  the judge's own failure: the reward abstains
            return {"id": request.get("id"), "error": f"{type(exc).__name__}: {exc}"[:500]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--mem-gb", type=float, default=None, help="TTSEM_MEM_GB of every run")
    args = ap.parse_args(argv)
    if args.mem_gb is not None:
        os.environ["TTSEM_MEM_GB"] = f"{args.mem_gb:g}"  # read by the judge's run_env per run
    # the protocol owns stdout: anything else printed in this process goes to stderr
    channel = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    sys.stdout = sys.stderr
    worker = Worker(*_import_judge())
    lock = threading.Lock()

    def answer(request: dict[str, Any]) -> None:
        reply = worker.handle(request)
        with lock:
            channel.write(json.dumps(reply) + "\n")
            channel.flush()

    with ThreadPoolExecutor(max(1, args.jobs)) as pool:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                with lock, contextlib.suppress(OSError):
                    channel.write(json.dumps({"id": None, "error": f"bad request: {exc}"}) + "\n")
                continue
            pool.submit(answer, request)
    return 0


if __name__ == "__main__":
    sys.exit(main())
