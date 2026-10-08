"""``python -m ttsem judge``: a KernelBench answer written with Triton, judged without a GPU.

    python -m ttsem judge TASK.py ANSWER.py [--json] [--timeout S] [--scale auto|none]
                          [--rule kernelbench|v3] [--precision fp32|keep] [--gpu NAME] [--out DIR]
                          [--strict]
    python -m ttsem judge --manifest M.jsonl --out DIR [--jobs N] [--timeout S] [--full-max N]
                          [--rule ...] [--precision ...] [--gpu ...]  (for lines without them)
    python -m ttsem judge --tritonbench TASK.py ANSWER.py | --manifest M.jsonl --out DIR
                          (a TritonBench answer: ``ttsem.judge_tritonbench``)

A task is a KernelBench problem file (`Model`, `get_inputs`, `get_init_inputs`); an answer is the
file that defines `ModelNew` (or, in the KernelBench-v3 release, a replacement `Model`). The answer
and the task's reference each run in their own process (``ttsem._judge_runner``) under `bwrap`
(`sandbox_prefix`), with a timeout and an address-space cap (`TTSEM_MEM_GB`, default 6). There the
answer is run by KernelBench's correctness rule (or, with `--rule v3`, by the KernelBench-v3
harness's own rule), with every Triton launch executed by ttsem on the CPU, and this process
compares what it returned with what the reference returned (`verdict_of`): the process that ran
the answer never writes a verdict.

One verdict per pass:

- verified: every trial within atol/rtol, no fault
- wrong: values or shape differ from the reference in some trial, or under another autotune config
  (one the autotuner could pick: not one it skips, nor one its `prune_configs_by` drops)
- unsafe: out-of-bounds access, race between program instances, or use of memory nobody wrote,
  with the kernel and its source line
- no_kernel: the answer launches no Triton kernel (PyTorch does the work: a known reward hack)
- error: the answer does not import, build or run (`shared_memory`: every config of one of its
  autotuners was skipped, and the one it falls back to needs more shared memory than the GPU
  has, by Triton 3.8.0's count, which can disagree with labels produced under another Triton; a
  launch outside an autotuner is not checked against the limit), or its process died (`crash`)
- too_slow: the timeout expired
- not_judged: the tool cannot say (an op ttsem does not model, its own error, the memory cap,
  a full /tmp, a task that does not run, a run that stopped before the trials the rule asks for)

`--gpu` sets only the shared-memory limit: every launch is compiled for sm_90a, so whether an
autotune config compiles (the autotuner skips one that raises `CompileTimeAssertionFailure` or
`PTXASError`) is decided for that target, whatever the GPU.

The `scaled` pass always runs (with `--scale auto`): every size constant of the task is divided
by one power of two until its largest tensor holds at most 2**17 elements (`judge_shapes`). The
`full` pass runs too when the task's largest tensor at its real shape has at most `--full-max`
elements and its reference model's parameters at most 32 times that, and always after a
`no_kernel` verdict (an answer may launch its kernel only at the benchmark's own shape).

What the sandbox gives a run: no capabilities, no network, its own pid namespace and session (no
controlling terminal, stdin closed), none of the caller's environment but what the run needs
(`run_env`), a private /tmp, and one writable directory of its own. It hides the caller's home
directory, the XDG directories, /run (the host's sockets), and the whole tree OUT sits in: the
reference runs, the other answers' runs and the judge's records. The rest of the host's filesystem
stays readable, read-only: system files, other users' world-readable files, other checkouts. With
`--strict` it sees only what a run needs: /usr and the links into it, the few files of /etc the
dynamic loader and the C library read, the CPU's description in /sys, the Python running the judge
with its site-packages, this package, and the task and the answer.

One pair prints one line per pass that ran (usually one) and exits 0 only if every pass says
`verified`. A manifest line is a JSON object with `id`, `task`, `answer` (relative paths are
taken from the manifest's directory), optionally `rule`, `precision` and `gpu`, and any labels
to carry along; `--rule`, `--precision` and `--gpu` given with `--manifest` apply to the lines
that do not set that field, or set it to null or "" (KernelBench's defaults otherwise). An id is
judged once: a later line with an id already seen is dropped, with a warning, if it names the
same task, answer and settings, and is an error otherwise. The reports go to
OUT/<id>.<pass>.json (a rerun reads back those judged under the same rule, precision and GPU:
only this process writes there), one row per answer to OUT/rows.jsonl, and the counts per verdict
to the terminal. The runs keep their files in OUT/work/<id>.<pass>/ and OUT/reference/.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import importlib.util
import json
import math
import os
import shutil
import site
import subprocess
import sys
import tempfile
import threading
import zipfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ttsem.judge_io import (
    Ran,
    compare,
    load_npz,
    printable,
    read_json,
    run_sandboxed,
    tensor_of,
)

VERDICTS = ("verified", "wrong", "unsafe", "no_kernel", "error", "too_slow", "not_judged")
RUNNER = "ttsem._judge_runner"
# the directory this `ttsem` is imported from, so the sandboxed process imports this very copy
ROOT = Path(__file__).resolve().parent.parent
# under OUT: the reference runs, and each answer's own directory
REFERENCE, WORK = "reference", "work"

KINDS = {
    "oob_read": "out-of-bounds read",
    "oob_write": "out-of-bounds write",
    "race": "race between program instances",
    "poison": "use of a value nobody wrote",
}

NO_BWRAP = """\
ttsem judge: {problem}. The answers are model-written code, and the judge runs them only
inside a bwrap sandbox (read-only filesystem, home hidden, no network, own pid namespace). Install
bubblewrap (apt install bubblewrap, dnf install bubblewrap), or pass --no-sandbox to run
them unsandboxed, with your user's permissions, at your own risk.
"""

# what a run's account may be broken in, once the answer's process wrote it
MALFORMED = (
    ValueError, KeyError, IndexError, TypeError, AttributeError, RecursionError, OSError,
    zipfile.BadZipFile,
)  # fmt: skip

# What a run is given of the caller's environment, where set. Nothing else of it reaches the
# answer, which could otherwise copy a token from it into the report the judge prints.
PASSED_ENV = (
    "PATH", "PYTHONPATH", "LANG", "LC_ALL", "PYTHONHASHSEED", "TTSEM_MEM_GB", "TTSEM_TRACEBACK",
    "TTSEM_BLOCK_CAP",
)  # fmt: skip
# HOME inside the sandbox: an empty directory of its private /tmp
SANDBOX_HOME = "/tmp/home"
# Every run: no capabilities (run by root, bwrap keeps them, and CAP_SYS_ADMIN would let the run
# unmount the tmpfs over a hidden directory), a session of its own (no controlling terminal, so
# no TIOCSTI into the caller's shell), no network, its own pid namespace, gone with its parent.
# A user namespace adds nothing here: bwrap makes one by itself for a caller without privileges,
# and for root it would map the run's uid back to root's.
SANDBOX_FLAGS = (
    "--cap-drop", "ALL", "--new-session", "--unshare-net", "--unshare-pid", "--die-with-parent",
)  # fmt: skip
# With --strict, of /etc only what the dynamic loader and the C library read
STRICT_ETC = (
    "/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d", "/etc/alternatives",
    "/etc/localtime", "/etc/passwd", "/etc/group", "/etc/nsswitch.conf",
)  # fmt: skip


def run_env(
    cache: Path, sandbox: bool = True, extra: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The whole environment one run starts with: `PASSED_ENV` from the caller's, where set,
    this package first on PYTHONPATH, `cache` as the Triton cache, one OpenMP thread, and
    `extra`. In the sandbox HOME is `SANDBOX_HOME` and TMPDIR its private /tmp; without it, the
    caller's own."""
    env = {key: os.environ[key] for key in PASSED_ENV if key in os.environ}
    env.setdefault("PATH", os.defpath)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), env.get("PYTHONPATH", "")) if p)
    env |= {"TRITON_CACHE_DIR": str(cache), "OMP_NUM_THREADS": "1"}
    if sandbox:
        env |= {"HOME": SANDBOX_HOME, "TMPDIR": "/tmp"}
    else:
        env |= {key: os.environ[key] for key in ("HOME", "TMPDIR") if key in os.environ}
    return env | dict(extra or {})


def private_dirs() -> list[Path]:
    """The directories the sandbox hides under an empty tmpfs: the caller's home directory (by
    HOME and by the password database), the XDG config, data, cache and state directories where
    they are set apart from it, the session's runtime directory, and /run (and /var/run where it
    is not a link to it), where the host's daemons keep their sockets: a socket on the
    filesystem is reachable without a network (the D-Bus buses, an ssh-agent, the Docker
    daemon, which would run anything as root). Never `/`, nor one under /tmp, which the sandbox
    replaces anyway; one inside another is left to the outer one."""
    names = [os.environ.get(key, "") for key in ("HOME", "XDG_RUNTIME_DIR")]
    names += [os.environ.get(f"XDG_{key}_HOME", "") for key in ("CONFIG", "DATA", "CACHE", "STATE")]
    names += ["/run", "/var/run"]
    with contextlib.suppress(ImportError, KeyError, AttributeError):
        import pwd

        names.append(pwd.getpwuid(os.getuid()).pw_dir)
    found: list[Path] = []
    paths = {Path(name).resolve() for name in names if name and os.path.isabs(name)}
    for path in sorted(paths, key=lambda p: (len(p.parts), str(p))):
        if path == Path("/") or path.is_relative_to("/tmp") or not path.is_dir():
            continue
        if not any(path.is_relative_to(outer) for outer in found):
            found.append(path)
    return found


def runtime_paths() -> list[Path]:
    """What every run reads: this package, the Python running it, its site-packages and
    PYTHONPATH, and where numpy, torch and Triton are installed (an editable install of one can
    sit anywhere, a home directory included)."""
    paths = [ROOT, Path(sys.prefix), Path(sys.base_prefix)]
    paths += [Path(p) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    paths += [Path(p) for p in site.getsitepackages()]
    if site.ENABLE_USER_SITE:
        paths.append(Path(site.getusersitepackages()))
    for name in ("numpy", "torch", "triton"):
        with contextlib.suppress(ImportError, ValueError):
            spec = importlib.util.find_spec(name)
            paths += [Path(p) for p in (spec.submodule_search_locations or [])] if spec else []
    return paths


def hidden_tree(path: Path) -> Path | None:
    """The highest directory above `path` that holds nothing the runs need (this package, the
    Python running it, the directories on PYTHONPATH, the home directory), or None: the tree a
    task or an output sits in, hidden from an answer whole (a benchmark's checkout, the outputs
    of earlier runs next to OUT). Under /tmp there is nothing to hide: the sandbox's /tmp is its
    own."""
    needed = [p.resolve() for p in [*runtime_paths(), Path.home()]]
    top = None
    for parent in path.resolve().parents:
        if parent.is_relative_to("/tmp") or any(p.is_relative_to(parent) for p in needed):
            break
        top = parent
    return top


def out_tree(out: Path) -> list[Path]:
    """What hides OUT from every run: the tree it sits in (`hidden_tree`), or OUT itself where
    that tree holds what the runs need; nothing under /tmp, which the sandbox replaces. An OUT
    that holds what the runs need (this package, the Python running it) is refused (ValueError):
    shown to the runs, it would show them the reference runs too."""
    out = out.resolve()
    held = [path for path in runtime_paths() if path.resolve().is_relative_to(out)]
    if held:
        raise ValueError(f"the output directory {out} holds {held[0]}: give it one of its own")
    if out.is_relative_to("/tmp"):
        return []
    return [hidden_tree(out) or out]


def strict_root() -> list[str]:
    """With --strict, the filesystem a run starts from: /usr read-only and the links into it
    (/bin, /lib, ...) as the host has them, the few files of /etc the dynamic loader and the C
    library read, and the CPU's description in /sys (torch reads it to choose its kernels)."""
    cmd = ["--ro-bind", "/usr", "/usr"]
    for name in ("bin", "sbin", "lib", "lib32", "lib64", "libx32"):
        path = Path("/") / name
        if path.is_symlink():
            cmd += ["--symlink", os.readlink(path), str(path)]
        elif path.is_dir():
            cmd += ["--ro-bind", str(path), str(path)]
    for name in STRICT_ETC:
        cmd += ["--ro-bind-try", name, name]
    return cmd + ["--ro-bind-try", "/sys/devices/system/cpu", "/sys/devices/system/cpu"]


def sandbox_prefix(
    out: Path,
    cache: Path,
    keep: list[Path] | None = None,
    env: Mapping[str, str] | None = None,
    hide: Sequence[Path] = (),
    strict: bool = False,
) -> list[str]:
    """The `bwrap` command line every run goes through (`SANDBOX_FLAGS`): the filesystem
    read-only except OUT (the run's own directory), a private /tmp, the caller's home and the
    host's sockets (`private_dirs`) and the directories in `hide` under an empty tmpfs, and no
    environment but `env` (`run_env`'s by default). The paths in `keep` and those every run reads
    (`runtime_paths`: this package, the Python running it) stay visible, read-only, where /tmp
    or a hidden directory would hide them; one that holds a hidden directory whole stays hidden.
    An OUT that holds one is refused (ValueError): the run would see it, and could write there.
    With `strict`, the run starts from an empty root instead of the host's (`strict_root`), and
    sees of the rest only `keep` and the runtime paths."""
    out = out.resolve()
    private = private_dirs()
    held = [path for path in private if path.is_relative_to(out)]
    if held:
        raise ValueError(f"the output directory {out} holds {held[0]}: give it one of its own")
    hidden = [Path("/tmp"), *private, *(Path(p).resolve() for p in hide)]
    shown = list(dict.fromkeys(p.resolve() for p in [*(keep or []), *runtime_paths()]))
    shown = [p for p in shown if p.exists()]
    cmd = ["bwrap", *SANDBOX_FLAGS]
    cmd += strict_root() if strict else ["--ro-bind", "/", "/"]
    cmd += ["--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
    if strict:  # nothing of the host but these, before what hides part of them
        for path in sorted(shown, key=lambda p: len(p.parts)):
            cmd += ["--ro-bind", str(path), str(path)]
    for path in hidden[1:]:
        cmd += ["--tmpfs", str(path)]
    for path in shown:
        if any(path.is_relative_to(top) for top in hidden) and not any(
            inner.is_relative_to(path) for inner in private
        ):
            cmd += ["--ro-bind", str(path), str(path)]
    cmd += ["--bind", str(out), str(out), "--dir", SANDBOX_HOME, "--chdir", str(out), "--clearenv"]
    for key, value in (run_env(cache) if env is None else env).items():
        cmd += ["--setenv", key, value]
    return cmd


def sandbox_problem(strict: bool = False) -> str | None:
    """Why the sandbox cannot run here (no `bwrap` on PATH, or one that cannot create the
    namespaces, as where unprivileged user namespaces are off, or that does not know one of
    `SANDBOX_FLAGS`), or None."""
    if shutil.which("bwrap") is None:
        return "bwrap is not on PATH"
    with tempfile.TemporaryDirectory(prefix="ttsem-probe-") as tmp:
        work = Path(tmp)
        env = {"PATH": os.environ.get("PATH", os.defpath)}
        probe = sandbox_prefix(work, work, [], env, strict=strict) + ["true"]
        done = subprocess.run(
            probe, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False
        )
    if done.returncode:
        why = (done.stderr or "").strip().splitlines()[-1:] or [f"exit {done.returncode}"]
        return f"bwrap cannot create a sandbox here ({why[0]})"
    return None


def fresh(path: Path) -> Path:
    """`path` as an empty directory of its own: a run never starts from what an earlier one
    left there."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def tidy(work: Path) -> None:
    """A run's directory once its outputs are read: its arrays and its Triton cache go (the
    account and the files it ran stay, for whoever looks)."""
    with contextlib.suppress(OSError):
        for path in work.iterdir():
            if path.name == ".cache" and path.is_dir() and not path.is_symlink():
                shutil.rmtree(path, ignore_errors=True)
            elif path.suffix == ".npz":
                path.unlink()


_LOCKS: dict[Path, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


@contextlib.contextmanager
def locked(path: Path) -> Iterator[None]:
    """One run at a time for `path` in this process: two answers of one task wait for one
    reference run."""
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(path.resolve(), threading.Lock())
    with lock:
        yield


def settings_flags(settings: Mapping[str, str]) -> list[str]:
    return [arg for key, value in settings.items() for arg in (f"--{key}", value)]


def reference_run(
    task: Path,
    out: Path,
    scale: str,
    settings: Mapping[str, str],
    timeout: int,
    sandbox: bool,
    strict: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """The task's reference run for one pass (``ttsem._judge_runner`` without an answer), once per
    task content, scale, rule and precision, in OUT/reference/<key>/ (its Triton cache inside),
    which no answer's sandbox sees. Returns (the reference, None): its trials (seed, precision,
    `want`), the sweep's inputs' output, the scaling; or (None, the record of what kept it from
    running), which is the answer's record too. What it came to is kept, by this process, in
    OUT/reference/<key>.json, and a rerun reads it back (a timeout aside)."""
    digest = hashlib.sha256(task.read_bytes()).hexdigest()[:12]
    key = f"{task.stem}-{digest}.{scale}.{settings['rule']}.{settings['precision']}"
    base = out / REFERENCE
    work, summary = base / key, base / f"{key}.json"
    with locked(summary):
        kept = read_json(summary)
        if not isinstance(kept, dict):
            fresh(work)
            cache = work / ".cache"
            cache.mkdir(parents=True, exist_ok=True)  # a run may write there first
            env = run_env(cache, sandbox)
            cmd = sandbox_prefix(work, cache, [task], env, out_tree(out), strict) if sandbox else []
            cmd += [sys.executable, "-P", "-m", RUNNER, str(task.resolve()), str(work.resolve())]
            cmd += ["--scale", scale, "--rule", settings["rule"]]
            cmd += ["--precision", settings["precision"]]
            ran = run_sandboxed(cmd, env, work, timeout)
            kept = reference_outcome(ran, work, timeout)
            if kept.get("verdict") != "too_slow":
                summary.write_text(json.dumps(kept, indent=1))
    if "verdict" in kept:
        return None, kept
    try:
        return reference_of(read_json(work / "run.json"), load_npz(work / "run.npz")), {}
    except MALFORMED as exc:
        unread = f"the reference run's outputs cannot be read: {type(exc).__name__}: {exc}"
        return None, {"verdict": "not_judged", "why": "task_error", "message": unread[:300]}


def reference_outcome(ran: Ran, work: Path, timeout: int) -> dict[str, Any]:
    """What a reference run came to: `{}` when it ran to the end, the record of the pass
    otherwise (a task that does not run, or stops, is not the answer's to answer for)."""
    account = read_json(work / "run.json")
    context = {}
    if isinstance(account, dict):
        context = {key: account[key] for key in ("scaling", "shapes") if key in account}
    stop = ran.stop()
    if stop is not None:
        message = f"the reference run stopped: {stop.get('fault') or stop.get('why')}"
        if stop.get("message"):
            message += f": {stop['message']}"
        return {"verdict": "not_judged", "why": "task_error", "message": message[:600], **context}
    if ran.code is None:
        return {"verdict": "too_slow", "seconds": timeout, "stage": "reference"}
    if not isinstance(account, dict) or not ran.done():
        message = f"the reference run died: {ran.tail}" if ran.tail else "the reference run died"
        return {"verdict": "not_judged", "why": "task_error", "message": message, **context}
    failure = account.get("failure")
    if isinstance(failure, dict):
        return {**context, "stage": account.get("stage"), **failure}
    return {}


def reference_of(account: Any, arrays: Mapping[str, Any]) -> dict[str, Any]:
    """The reference run's account with its outputs as tensors (`want`)."""
    if not isinstance(account, dict):
        raise TypeError("the reference run's account is not an object")
    trials = []
    for index, trial in enumerate(account["trials"]):
        want = tensor_of(trial["output"], arrays, f"trial{index}")
        trials.append({"seed": trial["seed"], "precision": str(trial["precision"]), "want": want})
    sweep = account["sweep"]
    found: dict[str, Any] = {
        "trials": trials,
        "sweep": {
            "precision": str(sweep["precision"]),
            "want": tensor_of(sweep["output"], arrays, "sweep"),
        },
    }
    return found | {key: account[key] for key in ("scaling", "shapes") if key in account}


def judge_one(
    task: Path,
    answer: Path,
    report: Path,
    timeout: int,
    scale: str = "auto",
    settings: dict[str, str] | None = None,
    sandbox: bool = True,
    strict: bool = False,
) -> dict[str, Any]:
    """One answer, in its own process, with a timeout, under `settings` (`rule`, `precision`,
    `gpu`; `DEFAULTS` for those missing), against the task's reference run (`reference_run`).
    The run writes only in its own directory, OUT/work/<report's name>/, which no other run
    sees; the report is this process's, and a rerun reads it back, unless it was judged under
    other settings."""
    settings = {**DEFAULTS, **(settings or {})}
    kept = read_json(report)
    if isinstance(kept, dict) and "verdict" in kept and kept.get("settings") == settings:
        return kept
    report.unlink(missing_ok=True)  # judged under another rule, precision or GPU: judged again
    out = report.parent.resolve()
    head = {"answer": answer.name, "scale": scale, "launches": 0, "settings": settings}
    reference, problem = reference_run(task, out, scale, settings, timeout, sandbox, strict)
    if reference is None:
        record = {**head, **problem}
    else:
        work = fresh(out / WORK / report.stem)
        cache = work / ".cache"
        cache.mkdir(parents=True, exist_ok=True)  # a run may write there first
        # the caller's environment stays out: an answer could copy a token from it into its report
        env = run_env(cache, sandbox)
        cmd = (
            sandbox_prefix(work, cache, [task, answer], env, out_tree(out), strict)
            if sandbox
            else []
        )
        # -P: the working directory, the run's own, stays off the path, so a module the answer
        # leaves there is not what the runner imports as ttsem
        cmd += [sys.executable, "-P", "-m", RUNNER, str(task.resolve()), str(work.resolve())]
        cmd += ["--answer", str(answer.resolve()), "--scale", scale, *settings_flags(settings)]
        ran = run_sandboxed(cmd, env, work, timeout)
        record = verdict_of(head, reference, ran, work, settings["rule"], timeout)
        tidy(work)
    report.write_text(json.dumps(record, indent=1))
    return record


# what of a run's account goes into its report as it is (the run's own counts and notes)
ACCOUNT = (
    "answer_overrides", "entry", "not_copied", "launches_init", "launches", "races_unchecked",
    "gpu", "shared_unknown", "torch_ops", "atomics", "autotune", "seconds",
)  # fmt: skip


def verdict_of(
    head: dict[str, Any],
    reference: dict[str, Any],
    ran: Ran,
    work: Path,
    rule: str,
    timeout: int,
) -> dict[str, Any]:
    """The pass's report, decided here from what the answer's run left: a `stop` it sent (a
    fault: unsafe; a launch the semantics cannot judge: not_judged), else, for a run that said
    `done`, its failure or its outputs compared with the reference's by `rule`. A run that sent
    neither died (error, `crash`) or was killed at the timeout (too_slow). Whatever the run
    wrote is read as untrusted (`MALFORMED`: the answer's crash)."""
    report = dict(head)
    report |= {key: reference[key] for key in ("scaling", "shapes") if key in reference}
    account = read_json(work / "run.json")
    account = account if isinstance(account, dict) else None
    stop = ran.stop()
    if stop is None and ran.code is None:
        return {**head, "verdict": "too_slow", "seconds": timeout}
    if stop is None and (account is None or not ran.done()):
        return {**head, "verdict": "error", "why": "crash", "message": ran.tail}
    try:
        if account is not None:
            arrays = load_npz(work / "run.npz") if (work / "run.npz").exists() else {}
            outcome = compared(report, account, arrays, reference, rule)
        else:
            outcome = {}
    except MALFORMED as exc:
        if stop is None:
            message = f"malformed run report: {type(exc).__name__}: {exc}"[:300]
            return {**head, "verdict": "error", "why": "crash", "message": message}
        outcome = {}
    if stop is not None:
        return report | stopped(stop) | stage_of(account)
    failure = account.get("failure") if account is not None else None
    if failure is None and account is not None and not isinstance(account.get("trials"), list):
        message = "malformed run report: no trials and no failure"
        return {**head, "verdict": "error", "why": "crash", "message": message}
    if failure is not None:
        if not isinstance(failure, dict) or failure.get("verdict") not in ("error", "not_judged"):
            message = "malformed run report: a failure that is no error"
            return {**head, "verdict": "error", "why": "crash", "message": message}
        return report | stage_of(account) | failure
    return report | outcome


def stage_of(account: dict[str, Any] | None) -> dict[str, Any]:
    """Where a run that did not end was: its stage, as it said it."""
    if account is None or "stage" not in account:
        return {}
    return {"stage": str(account["stage"])}


def stopped(stop: dict[str, Any]) -> dict[str, Any]:
    """The verdict of a `stop` (sent before the answer could catch anything)."""
    if stop.get("kind") == "fault":
        fields = ("kernel", "buffer", "file", "line", "source", "message")
        found = {key: stop.get(key) for key in fields}
        return {"verdict": "unsafe", "kind": str(stop.get("fault")), **found}
    why = str(stop.get("why") or "error")
    return {"verdict": "not_judged", "why": why, "message": str(stop.get("message", ""))[:300]}


def compared(
    report: dict[str, Any],
    account: dict[str, Any],
    arrays: Mapping[str, Any],
    reference: dict[str, Any],
    rule: str,
) -> dict[str, Any]:
    """The answer's outputs against the reference's, trial by trial, then config by config, into
    `report` (`trials`, `configs`, and the account's own counts); returns the verdict they come to
    when the run ended on its own. A comparison that raises (a dtype the reference's does not
    take) is the answer's error, as it was when the runner compared."""
    report |= {key: account[key] for key in ACCOUNT if key in account}
    expected = reference["trials"]
    trials: list[dict[str, Any]] = []
    report["trials"] = trials
    stage = "forward"
    try:
        ran = account.get("trials") or []
        for index, (mine, theirs) in enumerate(zip(ran, expected, strict=False)):
            got = tensor_of(mine["output"], arrays, f"trial{index}")
            row = compare(theirs["want"], got, theirs["precision"], rule)
            row.update(seed=theirs["seed"], launches=int(mine["launches"]))
            row["seconds"] = mine.get("seconds")
            trials.append(row)
            if not row["pass"]:
                break
        passed = len(trials) == len(expected) and all(t["pass"] for t in trials)
        sweep = account.get("sweep")
        if passed and sweep is not None:
            stage = "configs"
            report["configs"] = configs_of(sweep, arrays, reference["sweep"], rule, account)
    except MALFORMED:
        raise
    except Exception as exc:  # noqa: BLE001  a comparison torch refuses, as `torch.allclose` does
        text = str(exc).strip().splitlines()
        message = f"{type(exc).__name__}: {text[0] if text else ''}"[:300]
        return {"verdict": "error", "stage": stage, "message": message}
    if trials and all(t["pass"] for t in trials) and len(trials) < len(expected):
        short = f"the run stopped after {len(trials)} of the rule's {len(expected)} trials"
        return {"verdict": "not_judged", "why": "incomplete", "message": short}
    if passed and int(account.get("tuners", 0)) and "configs" not in report:
        return {"verdict": "not_judged", "why": "incomplete", "message": "no autotune sweep"}
    if not any(t["launches"] for t in trials):  # a launch at init only does not do the task
        return {"verdict": "no_kernel"}
    if passed and "failed" not in report.get("configs", {}):
        return {"verdict": "verified"}
    return {"verdict": "wrong"}


def configs_of(
    sweep: dict[str, Any],
    arrays: Mapping[str, Any],
    reference: dict[str, Any],
    rule: str,
    account: dict[str, Any],
) -> dict[str, Any]:
    """The sweep's outcome (`checked`, the config that `failed` first, those `not_applicable`):
    each config's output against the reference's on the same inputs."""
    done: dict[str, Any] = {"tuners": int(account.get("tuners", 0)), "checked": 0}
    last = math.inf
    for run in sweep["runs"]:
        got = tensor_of(run["output"], arrays, f"config{int(run['index'])}")
        row = compare(reference["want"], got, reference["precision"], rule)
        done["checked"] += 1
        if not row["pass"]:
            rest = {k: v for k, v in row.items() if k != "pass"}
            done["failed"] = {"index": int(run["index"]), "config": str(run["config"]), **rest}
            last = int(run["index"])
            break
    skipped = [entry for entry in sweep.get("not_applicable", []) if entry["index"] <= last]
    if skipped:
        done["not_applicable"] = skipped
    if "forcing" in sweep:
        done["forcing"] = sweep["forcing"]
    return done


def judge_item(
    item: dict[str, Any],
    out: Path,
    timeout: int,
    full_max: int,
    sandbox: bool = True,
    strict: bool = False,
) -> dict[str, Any]:
    """The scaled pass, and the full pass where the rule above calls for it."""
    task, answer = Path(item["task"]), Path(item["answer"])
    # the GPU's shared-memory limit bounds the configs judged
    settings = {key: item[key] for key in DEFAULTS}
    small = out / f"{item['id']}.scaled.json"
    scaled = judge_one(task, answer, small, timeout, "auto", settings, sandbox, strict)
    row = {**item, "scaled": scaled}
    note = scaled.get("scaling", {})
    footprint = note.get("footprint_full")
    full = out / f"{item['id']}.full.json"
    if scaled["verdict"] == "no_kernel" and note.get("factor", 1) > 1:
        # PyTorch alone is quick at any size, and an answer that launches its kernel only at the
        # benchmark's own shape (`if x.shape[1] != 256: return torch.sum(...)`) shows it there
        row["full"] = judge_one(task, answer, full, timeout, "none", settings, sandbox, strict)
    elif (
        full_max
        and footprint is not None
        and note.get("factor", 1) > 1
        and footprint <= full_max
        # parameters cost memory more than time: a looser bound for them
        and note.get("params_full", 0) <= 32 * full_max
    ):
        row["full"] = judge_one(task, answer, full, timeout, "none", settings, sandbox, strict)
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
        description="A KernelBench or TritonBench answer in Triton, judged without a GPU.",
    )
    ap.add_argument("task", type=Path, nargs="?", help="the KernelBench problem file")
    ap.add_argument("answer", type=Path, nargs="?", help="the file that defines ModelNew")
    ap.add_argument(
        "--tritonbench",
        action="store_true",
        help="TASK.py is a TritonBench task (reference, separator, test), ANSWER.py its answer",
    )
    ap.add_argument("--manifest", type=Path, help="judge every answer of this JSONL file")
    ap.add_argument("--out", type=Path, help="where the reports go (a temporary one otherwise)")
    ap.add_argument("--json", action="store_true", help="print the verdict record as JSON")
    ap.add_argument("--scale", choices=("auto", "none"), help="(auto)")
    # no default here: with --manifest, a flag given applies to the lines that do not set it
    ap.add_argument("--rule", choices=("kernelbench", "v3"), help="(kernelbench)")
    ap.add_argument("--precision", choices=("fp32", "keep"), help="(fp32)")
    ap.add_argument("--gpu", help="the dataset's GPU (H100, B200, RTX3090, ...)")
    ap.add_argument("--timeout", type=int, default=None, help="seconds per pass (600; 300 batch)")
    ap.add_argument("--full-max", type=int, help="(2**20)")
    ap.add_argument("--jobs", type=int, default=2, help="answers judged at once (batch)")
    ap.add_argument("--limit", type=int, default=0, help="the first N answers only (batch)")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="show the runs only what they need of the filesystem, not the host's read-only",
    )
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
    if args.strict and args.no_sandbox:
        ap.error("--strict narrows the sandbox: not with --no-sandbox")
    for path in (args.manifest,) if args.manifest is not None else (args.task, args.answer):
        if not path.is_file():
            ap.error(f"{path}: no such file")
    if args.tritonbench:
        given = [
            f"--{k.replace('_', '-')}" for k in KERNELBENCH_ONLY if getattr(args, k) is not None
        ]
        if given:
            ap.error(f"{', '.join(given)}: KernelBench only, not with --tritonbench")
    sandbox = not args.no_sandbox
    if sandbox:
        problem = sandbox_problem(args.strict)
        if problem is not None:
            sys.stderr.write(NO_BWRAP.format(problem=problem))
            return 2
    else:
        sys.stderr.write(
            "ttsem judge: warning: --no-sandbox: model-written code is about to run unsandboxed, "
            "with your permissions\n"
        )
    if args.tritonbench:
        from ttsem import judge_tritonbench as tritonbench

        if args.manifest is not None:
            return tritonbench.batch(args, sandbox)
        return tritonbench.single(args, sandbox)
    args.scale = args.scale or "auto"
    args.full_max = (1 << 20) if args.full_max is None else args.full_max
    if args.manifest is not None:
        return batch(args, sandbox)
    return single(args, sandbox)


# what a pair, or a manifest line, is judged under when neither the line nor a flag says
DEFAULTS = {"rule": "kernelbench", "precision": "fp32", "gpu": ""}
# the flags that mean nothing to a TritonBench answer (its test sets the shapes, the dtypes and
# the tolerance is TritonBench's)
KERNELBENCH_ONLY = ("scale", "rule", "precision", "gpu", "full_max")


def single(args: argparse.Namespace, sandbox: bool) -> int:
    item = {
        "id": args.answer.stem,
        "task": str(args.task),
        "answer": str(args.answer),
        **{key: getattr(args, key) or value for key, value in DEFAULTS.items()},
    }
    timeout = args.timeout or 600
    strict = bool(getattr(args, "strict", False))
    with contextlib.ExitStack() as stack:
        out = args.out or Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="ttsem-")))
        out.mkdir(parents=True, exist_ok=True)
        for tag in ("scaled", "full"):  # one pair is judged afresh, not read back
            (out / f"{item['id']}.{tag}.json").unlink(missing_ok=True)
        if args.scale == "auto":
            row = judge_item(item, out, timeout, args.full_max, sandbox, strict)
        else:
            settings = {key: item[key] for key in DEFAULTS}
            report = out / f"{item['id']}.full.json"
            full = judge_one(
                args.task, args.answer, report, timeout, "none", settings, sandbox, strict
            )
            row = {**item, "full": full}
    if args.json:
        print(json.dumps(row, indent=1))
    else:
        print(printable("\n".join(lines_of(row))))
    passes = [row[key]["verdict"] for key in ("scaled", "full") if key in row]
    return 0 if all(v == "verified" for v in passes) else 1


def differences(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """The fields that decide a verdict (the task, the answer, the settings) on which two
    manifest lines differ."""
    paths = [key for key in ("task", "answer") if Path(a[key]).resolve() != Path(b[key]).resolve()]
    return paths + [key for key in DEFAULTS if a.get(key) != b.get(key)]


def unique(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The manifest's lines with one per id, the first: a repeated id would be counted twice and,
    with --jobs above 1, judged twice at once into the same report. A repeat of the same task,
    answer and settings is dropped with a warning; an id that names anything else twice raises
    ValueError, since only one of the two would be judged."""
    seen: dict[str, tuple[int, dict[str, Any]]] = {}
    kept = []
    for line, item in enumerate(items, 1):
        key = str(item["id"])
        if key not in seen:
            seen[key] = (line, item)
            kept.append(item)
            continue
        first, was = seen[key]
        differ = differences(was, item)
        if differ:
            raise ValueError(
                f"id {key!r} on line {line} was already on line {first}, with another "
                f"{' and '.join(differ)}: give each answer its own id"
            )
        sys.stderr.write(
            f"ttsem judge: warning: id {key!r} on line {line} repeats line {first}: judged once\n"
        )
    return kept


def batch(args: argparse.Namespace, sandbox: bool) -> int:
    items = load_manifest(args.manifest)
    for item in items:  # a flag given applies where the line does not say (or says null or "")
        for key, value in DEFAULTS.items():
            item[key] = item.get(key) or getattr(args, key) or value
    try:
        items = unique(items)
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
    with ThreadPoolExecutor(args.jobs) as pool, (out / "rows.jsonl").open("w") as sink:
        work = functools.partial(
            judge_item,
            out=out,
            timeout=timeout,
            full_max=args.full_max,
            sandbox=sandbox,
            strict=strict,
        )
        for row in pool.map(work, items):
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            print(printable(f"{row['id']}: " + "; ".join(lines_of(row))), flush=True)
    print("\n".join(tally(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
