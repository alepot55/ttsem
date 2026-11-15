"""The exact judge as a reward for kernel RL (new in ttsem 0.2, prototype-grade).

`Judge.score(reference_src, answer_src)` returns a `Verdict`: a score (1.0, 0.0, or None for
"abstain"), the judge's verdict and its kind, one paragraph of feedback for the model, the tier
that decided, and the process-seconds the answer cost. This process imports neither torch nor
Triton: every pass goes to a worker in the judge's own venv (`worker.py`, JSON lines on a pipe),
which runs each answer and each reference in its own `bwrap` sandbox. Nothing here runs an answer.

Tiers for a KernelBench-style task (`Model`, `get_inputs`, `get_init_inputs`; the answer defines
`ModelNew`):

1. `scale`: the task's sizes divided by a power of two until its largest tensor holds at most
   2**13 elements, one trial. If it verifies and the answer has a launch whose grid is a ceiling
   division over a block constexpr (`cap.py`), the pass runs again on a copy with those blocks
   capped at `cap` (`scale_cap`), and only an `unsafe` verdict of that copy counts: the cap is
   there to show races between program instances that a one-instance test hides.
2. `medium`: 2**17 elements, one trial, only after `scale` verified (skipped when `scale` already
   ran the task at its own sizes). A launch too large for the race check here is accepted when
   `scale` checked races.
3. `real` (optional): the task's own sizes, when its largest tensor holds at most `real_max`
   elements; and, under `contract="task"`, to re-judge at the task's shape whatever a scaled tier
   rejected.

`contract="operator"` (default): an answer must be right at the task's shape and at the scaled
shapes, so a kernel specialised to the benchmark's shape gets 0. `contract="task"`: only the task's
shape counts; a rejection at a scaled shape stands only if the real shape confirms it, and it
abstains where the real shape is too large to run.

A TritonBench task (`format="tritonbench"`: the reference, a line of 146 `#`, the test) has one
tier, `test`, at the shapes its test sets, and the same cap.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ttsem.reward import cap as _cap

SCALE_BUDGET = 1 << 13
MEDIUM_BUDGET = 1 << 17
REAL_MAX = 1 << 20
TIERS = ("scale", "medium", "real")
MODES = ("gate", "cpu")
CONTRACTS = ("operator", "task")
FORMATS = ("kernelbench", "tritonbench")
# the answer reached into the checker: 0 in every mode, and counted as a hack attempt
HACKS = frozenset({"tampered", "forged"})
# errors that may be the drift between the trainer's Triton and the judge's, not the answer's
# fault (the first word of the judge's message: the exception's type)
VERSION_ERRORS = frozenset(
    {"CompilationError", "CompileTimeAssertionFailure", "PTXASError", "ImportError",
     "ModuleNotFoundError"}
)  # fmt: skip
TB_FAULTS = frozenset({"oob_read", "oob_write", "race", "poison"})
# ttsem's `judge_io.compare` calls `torch.allclose` under every rule, which raises this when the
# answer's output dtype is not the reference's; the KernelBench-v3 harness casts both to fp32
# before it compares, so under `v3` the error is the judge's, not the answer's (no `where`: no
# frame of the answer raised it). Under `kernelbench` it stands: KernelBench's own
# `torch.allclose` raises the same.
_DTYPE_CLASH = re.compile(r"^RuntimeError: \w+ did not match \w+$")
WORKER = Path(__file__).resolve().parent / "worker.py"

Runner = Callable[[dict[str, Any], float], dict[str, Any]]


@dataclass(frozen=True)
class Verdict:
    """One answer judged. `score` is the mode's: in gate mode the multiplier of the base reward,
    in CPU-only mode the reward itself; None is an abstention."""

    score: float | None
    verdict: str  # verified | wrong | unsafe | no_kernel | error | too_slow | not_judged
    kind: str | None  # oob_read | race | unwritten | values | autotune | version | tampered | ...
    feedback: str
    tier: str  # the tier that decided: scale | scale_cap | medium | real | test
    seconds: float  # CPU seconds of the answer's sandboxed runs, references excluded
    record: dict[str, Any] = field(repr=False)

    @property
    def abstain(self) -> bool:
        return self.score is None

    @property
    def hack(self) -> bool:
        """The answer reached into the checker (logged per step: a policy that learns to attack
        the checker shows up here first)."""
        return self.kind in HACKS


@dataclass(frozen=True)
class Pass:
    """One pass of the judge: its record, its line, and its cost (wall and CPU seconds of the
    answer's runs and of the reference's, which a cached reference makes 0)."""

    tier: str
    record: dict[str, Any]
    line: str
    answer_s: float
    reference_s: float
    answer_cpu_s: float = 0.0
    reference_cpu_s: float = 0.0

    @property
    def verdict(self) -> str:
        return str(self.record.get("verdict"))


# ---------------------------------------------------------------- verdict to reward


def kind_of(record: Mapping[str, Any], fmt: str = "kernelbench") -> str | None:
    """The finer reason of one pass's verdict, from the judge's record."""
    if fmt == "tritonbench":
        return _tb_kind(record)
    verdict = record.get("verdict")
    if verdict == "unsafe":
        return str(record.get("kind") or "fault")
    if verdict == "wrong":
        if "failed" in (record.get("configs") or {}):
            return "autotune"  # right under the first config only: another one the GPU may pick
        if record.get("not_copied"):
            return "weights"  # parameters under names the v3 harness does not copy
        trials = record.get("trials") or []
        bad = trials[-1] if trials else {}
        if "shape" in bad:
            return "shape"
        if "why" in bad:
            return "not_tensor"
        diff = bad.get("max_abs_diff")
        if isinstance(diff, float) and math.isnan(diff) and not bad.get("ref_nan"):
            return "unwritten"  # NaN: the output holds memory nobody wrote
        return "values"
    if verdict == "error":
        return _error_kind(record)
    if verdict == "not_judged":
        return _unjudged_kind(str(record.get("why") or "unknown"), record)
    if verdict == "too_slow":
        return "timeout"
    return None


def _error_kind(record: Mapping[str, Any]) -> str:
    why = record.get("why")
    if why in ("crash", "shared_memory"):
        return str(why)
    if record.get("stage") == "configs":
        # the sweep forces a config the autotuner would skip: not the answer's to answer for
        return "configs"
    message = str(record.get("message") or "")
    rule = (record.get("settings") or {}).get("rule")
    if rule == "v3" and not record.get("where") and _DTYPE_CLASH.match(message):
        return "compare_dtype"
    if message.split(":", 1)[0].strip() in VERSION_ERRORS:
        return "version"
    return "raised"


def _tb_kind(record: Mapping[str, Any]) -> str | None:
    name = str(record.get("class") or "")
    verdict = record.get("verdict")
    if verdict == "unsafe" or name in TB_FAULTS:
        return name
    if verdict == "wrong":
        return "values"
    if verdict == "verified":
        return "reduced_precision" if record.get("tolerance") == "reduced" else None
    if verdict == "error":
        if name == "crash":
            return "crash"
        message = str(record.get("message") or "")
        return "version" if message.split(":", 1)[0].strip() in VERSION_ERRORS else "raised"
    if verdict == "too_slow":
        return "timeout"
    if verdict == "no_kernel":
        return None
    return _unjudged_kind(name or "unknown", record)


def _unjudged_kind(why: str, record: Mapping[str, Any]) -> str:
    """`native_code` for an answer the scan refuses because it builds C++ or CUDA of its own
    (`torch.utils.cpp_extension`): out of the semantics' reach, not an attack on the checker."""
    if why == "tampered" and "`cpp_extension`" in str(record.get("message") or ""):
        return "native_code"
    return why


def scores(verdict: str, kind: str | None) -> tuple[float | None, float | None]:
    """(CPU-only reward, gate multiplier) of a final verdict; None abstains (the shape contract
    is applied before: see `Judge`)."""
    if verdict == "verified":
        return 1.0, 1.0
    if verdict in ("wrong", "unsafe", "no_kernel"):
        return 0.0, 0.0
    if verdict == "not_judged" and kind in HACKS:
        return 0.0, 0.0
    if verdict == "not_judged" and kind == "native_code":
        return 0.0, None  # CPU-only mode cannot credit what it cannot run; the gate abstains
    if verdict == "error":
        if kind in ("version", "shared_memory"):
            return 0.0, None  # the answer failed here, but maybe only under the judge's Triton
        if kind in ("crash", "configs", "compare_dtype"):
            return None, None  # the judge cannot tell whose fault it is, or it is the judge's
        return 0.0, 0.0  # raised by the answer (an `assert D == 1024` at a scaled shape too)
    return None, None  # too_slow, not_judged


def rejects(p: Pass, fmt: str) -> bool:
    """Whether this pass alone says the answer is not a right kernel (an abstention is not)."""
    cpu, gate = scores(p.verdict, kind_of(p.record, fmt))
    return cpu == 0.0 and gate == 0.0


def races_unchecked(p: Pass) -> bool:
    return p.verdict == "not_judged" and p.record.get("why") == "races_unchecked"


# ---------------------------------------------------------------- answers and files


# the names the judge gives the answer's copies: its file, its scaled and capped copies, the
# TritonBench file it builds (answer, separator, test)
_COPIES = re.compile(
    r"\b(?:t[0-9a-f]{16}\.)?a[0-9a-f]{16}(?:\.c\d+)?(?:_answer|\.tritonbench)?\.py\b"
)
_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", re.S)
_ANSWER_CLASS = re.compile(r"^class\s+ModelNew\b", re.M)


def extract_answer(completion: str) -> str | None:
    """The answer's code in a model's completion: the last fenced Python block that defines
    `ModelNew`, else the last fenced Python block, else the completion itself when it is Python
    that defines `ModelNew`; None when there is none."""
    blocks = [
        str(body)
        for lang, body in _FENCE.findall(completion)
        if str(lang).lower() in ("", "python", "py")
    ]
    for body in reversed(blocks):
        if _ANSWER_CLASS.search(body):
            return body
    if blocks:
        return blocks[-1]
    if _ANSWER_CLASS.search(completion):
        with contextlib.suppress(SyntaxError, ValueError):
            ast.parse(completion)
            return completion
    return None


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _put(path: Path, text: str) -> Path:
    """`text` at `path`, written once (content-addressed: the name carries its digest)."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    return path


def _span_text(code: str, span: tuple[int, int, int, int]) -> str:
    """The source text of an `ast` span (1-based lines, UTF-8 byte columns)."""
    lines = code.encode("utf-8").split(b"\n")
    sl, sc, el, ec = span
    if sl == el:
        return lines[sl - 1][sc:ec].decode("utf-8")
    parts = [lines[sl - 1][sc:], *lines[sl : el - 1], lines[el - 1][:ec]]
    return b"\n".join(parts).decode("utf-8")


def _below_autotune_floor(code: str, launch: Any, cap: int) -> bool:
    """True when the launch's block comes from `@triton.autotune` configs that all hold integer
    literals above `cap`: no config the answer's own tuner can pick launches a block that small, and
    the host code may size its buffers for the smallest listed one. Such a launch is not capped;
    the medium tier still judges it at the answer's own blocks."""
    if "autotune" not in launch.sources:
        return False
    values = []
    for span in launch.spans:
        try:
            values.append(ast.literal_eval(_span_text(code, span)))
        except (ValueError, SyntaxError):
            return False
    return bool(values) and all(isinstance(v, int) and v > cap for v in values)


def capped_source(source: str, cap: int) -> str | None:
    """The answer with the block constexpr of every qualifying launch capped at `cap` and the
    grid's divisor with it (`cap.rewrite`: no line added or removed), or None where no launch
    qualifies. A launch whose autotune configs all list blocks above `cap` is left as written
    (`_below_autotune_floor`): the cap stays within what the answer's own tuner can reach."""
    code = _cap.operator_part(source)
    try:
        launches, _ = _cap.analyse(code)
        launches = [launch for launch in launches if not _below_autotune_floor(code, launch, cap)]
        if not launches:
            return None
        spans = [s for launch in launches for s in launch.spans]
        new = _cap.rewrite(code, spans, cap, not _cap.shadows_min(code))
        ast.parse(new)
    except Exception:  # noqa: BLE001  an answer the rewrite does not handle is not capped
        return None
    return str(new) + source[len(code) :]


# ---------------------------------------------------------------- the worker


class WorkerPool:
    """One worker process (`worker.py` in the judge's venv) with `jobs` passes at a time,
    started on first use and again in a forked child (a trainer's reward processes)."""

    def __init__(
        self, python: Path, jobs: int, ttsem_root: str | None = None, mem_gb: float | None = None
    ) -> None:
        self.python, self.jobs, self.ttsem_root, self.mem_gb = python, jobs, ttsem_root, mem_gb
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._pid = 0
        self._pending: dict[int, Future[dict[str, Any]]] = {}
        self._ids = itertools.count(1)

    def _start(self) -> subprocess.Popen[str]:
        if self._proc is not None and self._pid == os.getpid() and self._proc.poll() is None:
            return self._proc
        env = dict(os.environ)
        if self.ttsem_root:
            env["TTSEM_REWARD_TTSEM_ROOT"] = self.ttsem_root
        cmd = [str(self.python), "-I", str(WORKER), "--jobs", str(self.jobs)]
        if self.mem_gb is not None:
            cmd += ["--mem-gb", f"{self.mem_gb:g}"]
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env, bufsize=1
        )
        self._proc, self._pid, self._pending = proc, os.getpid(), {}
        threading.Thread(target=self._read, args=(proc,), daemon=True).start()
        return proc

    def _read(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            with contextlib.suppress(json.JSONDecodeError):
                reply = json.loads(line)
                with self._lock:
                    future = self._pending.pop(reply.get("id"), None)
                if future is not None:
                    future.set_result(reply)
        with self._lock:
            waiting, self._pending = list(self._pending.values()), {}
        for future in waiting:
            future.set_exception(RuntimeError(f"the reward worker exited ({proc.poll()})"))

    def request(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        future: Future[dict[str, Any]] = Future()
        with self._lock:
            proc = self._start()
            ident = next(self._ids)
            self._pending[ident] = future
            assert proc.stdin is not None
            proc.stdin.write(json.dumps({**payload, "id": ident}) + "\n")
            proc.stdin.flush()
        try:
            return future.result(timeout=timeout)
        finally:
            with self._lock:
                self._pending.pop(ident, None)

    def close(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is not None and proc.stdin is not None:
            with contextlib.suppress(OSError):
                proc.stdin.close()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=30)


# ---------------------------------------------------------------- the judge


class Judge:
    """The reward's judge. `python` is the interpreter of the judge's venv (ttsem with its
    `triton` extra: the pinned Triton and torch CPU); `ttsem_root` a checkout of ttsem to import
    ahead of the venv's own; `cache_dir` holds the tasks and answers (content-addressed), the
    reference runs (one per task and tier, reused across a GRPO group and across epochs) and
    every pass's record; no answer's sandbox sees it but its own files. `rule`, `precision` and
    `gpu` are the judge's (`kernelbench`: `allclose` at 1e-2; `v3`: the KernelBench-v3 release's);
    `score` takes them per call too. `fail_value` is what gate mode gives a rejected answer when
    the base reward is above it (default 0: the base reward multiplied by the score)."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        python: Path | None = None,
        workers: int | None = None,
        mode: str = "gate",
        tiers: tuple[str, ...] = ("scale", "medium"),
        contract: str = "operator",
        rule: str = "kernelbench",
        precision: str = "fp32",
        gpu: str = "",
        timeout_s: int = 120,
        cap: int | None = 16,
        on_abstain: str = "keep",
        abstain_value: float | None = None,
        fail_value: float = 0.0,
        strict_fs: bool = True,
        real_max: int = REAL_MAX,
        ttsem_root: str | None = None,
        mem_gb: float | None = None,
        runner: Runner | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode {mode!r}: one of {MODES}")
        if contract not in CONTRACTS:
            raise ValueError(f"contract {contract!r}: one of {CONTRACTS}")
        if "scale" not in tiers or not set(tiers) <= set(TIERS):
            raise ValueError(f"tiers {tiers!r}: `scale` and any of {TIERS[1:]}")
        if on_abstain not in ("keep", "zero"):
            raise ValueError(f"on_abstain {on_abstain!r}: keep or zero")
        self.cache_dir = Path(cache_dir).resolve()
        self.workers = workers or os.cpu_count() or 1
        self.mode, self.tiers, self.contract = mode, tuple(tiers), contract
        self.settings = {"rule": rule, "precision": precision, "gpu": gpu}
        self.timeout_s, self.cap, self.real_max = timeout_s, cap, real_max
        self.on_abstain, self.abstain_value, self.fail_value = on_abstain, abstain_value, fail_value
        self.strict_fs = strict_fs
        self._pool: WorkerPool | None = None
        if runner is None:
            self._pool = WorkerPool(
                Path(python or sys.executable), self.workers, ttsem_root, mem_gb
            )
            runner = self._pool.request
        self._runner = runner
        self._checked = False
        self._check_lock = threading.Lock()
        # one judging at a time per (task, answer): a GRPO group repeats completions, and the
        # repeat reads the first one's records back instead of racing it for the same files
        self._pairs: dict[tuple[str, str], threading.Lock] = {}
        self._pairs_lock = threading.Lock()
        self.info: dict[str, Any] = {}

    @classmethod
    def from_env(cls, **overrides: Any) -> Judge:
        """A judge configured by environment variables (the verl adapter's way: no code):
        TTSEM_REWARD_CACHE, _PYTHON, _TTSEM_ROOT, _WORKERS, _MODE, _CONTRACT, _CAP (0: off),
        _TIERS (comma-separated), _TIMEOUT, _RULE, _PRECISION, _GPU, _MEM_GB."""
        env = os.environ.get
        kw: dict[str, Any] = {
            "cache_dir": Path(env("TTSEM_REWARD_CACHE", "~/.cache/ttsem_reward")).expanduser(),
            "mode": env("TTSEM_REWARD_MODE", "gate"),
            "contract": env("TTSEM_REWARD_CONTRACT", "operator"),
            "rule": env("TTSEM_REWARD_RULE", "kernelbench"),
            "precision": env("TTSEM_REWARD_PRECISION", "fp32"),
            "gpu": env("TTSEM_REWARD_GPU", ""),
            "ttsem_root": env("TTSEM_REWARD_TTSEM_ROOT") or None,
        }
        if env("TTSEM_REWARD_PYTHON"):
            kw["python"] = Path(env("TTSEM_REWARD_PYTHON", ""))
        if env("TTSEM_REWARD_WORKERS"):
            kw["workers"] = int(env("TTSEM_REWARD_WORKERS", "1"))
        if env("TTSEM_REWARD_CAP") is not None:
            kw["cap"] = int(env("TTSEM_REWARD_CAP", "0")) or None
        if env("TTSEM_REWARD_TIERS"):
            kw["tiers"] = tuple(t for t in env("TTSEM_REWARD_TIERS", "").split(",") if t)
        if env("TTSEM_REWARD_TIMEOUT"):
            kw["timeout_s"] = int(env("TTSEM_REWARD_TIMEOUT", "120"))
        if env("TTSEM_REWARD_MEM_GB"):
            kw["mem_gb"] = float(env("TTSEM_REWARD_MEM_GB", "6"))
        return cls(**{**kw, **overrides})

    # -------------------------------------------------- public

    def score(self, reference_src: str, answer_src: str | None, **overrides: Any) -> Verdict:
        """One answer against one task's reference. `overrides`: `rule`, `precision`, `gpu`,
        `format` (kernelbench or tritonbench), `contract`."""
        fmt = str(overrides.pop("format", "kernelbench"))
        contract = str(overrides.pop("contract", self.contract))
        if fmt not in FORMATS or contract not in CONTRACTS:
            raise ValueError(f"format {fmt!r}, contract {contract!r}")
        unknown = set(overrides) - set(self.settings)
        if unknown:
            raise ValueError(f"unknown settings {sorted(unknown)}: rule, precision, gpu")
        settings = {**self.settings, **{k: str(v) for k, v in overrides.items() if v is not None}}
        if not answer_src or not answer_src.strip():
            return self._final([], "error", "no_code", "", fmt, "the completion holds no code")
        self._check()
        task = _put(self.cache_dir / "src" / f"t{digest(reference_src)}.py", reference_src)
        answer = _put(self.cache_dir / "src" / f"a{digest(answer_src)}.py", answer_src)
        with self._pairs_lock:
            lock = self._pairs.setdefault((task.stem, answer.stem), threading.Lock())
        try:
            with lock:
                if fmt == "tritonbench":
                    return self._tritonbench(task, answer, answer_src)
                return self._kernelbench(task, answer, answer_src, settings, contract)
        except Exception as exc:  # noqa: BLE001  the worker's own failure: abstain, and say so
            text = f"the judge failed: {type(exc).__name__}: {exc}"[:300]
            return self._final([], "not_judged", "tool_error", "", fmt, text)

    def score_many(
        self, items: Iterable[tuple[str, str | None]], **overrides: Any
    ) -> list[Verdict]:
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(lambda it: self.score(it[0], it[1], **overrides), items))

    async def ascore(self, reference_src: str, answer_src: str | None, **overrides: Any) -> Verdict:
        return await asyncio.to_thread(self.score, reference_src, answer_src, **overrides)

    def gate(self, base: float, v: Verdict) -> float:
        """The gated reward: never above `base` (monotone), `base` itself when verified or, with
        on_abstain="keep", when the judge abstains."""
        if v.score is None:
            return base if self.on_abstain == "keep" else min(base, self.fail_value)
        return base if v.score >= 1.0 else min(base, self.fail_value)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()

    # -------------------------------------------------- the passes

    def _check(self) -> None:
        """Once: the worker answers, and its sandbox works (the reward never runs an answer
        without `bwrap`)."""
        with self._check_lock:
            if self._checked:
                return
            info = self._runner({"op": "hello", "strict": self.strict_fs}, 300.0)
            if info.get("error") or info.get("sandbox_problem"):
                why = info.get("error") or info.get("sandbox_problem")
                raise RuntimeError(f"ttsem.reward: the judge cannot run here: {why}")
            self.info = {k: v for k, v in info.items() if k != "id"}
            self._checked = True

    def _call(self, payload: dict[str, Any], tier: str) -> Pass:
        reply = self._runner(
            {**payload, "timeout": self.timeout_s, "strict": self.strict_fs},
            2.0 * self.timeout_s + 120.0,
        )
        if reply.get("error"):
            raise RuntimeError(str(reply["error"]))
        return Pass(
            tier,
            dict(reply["record"]),
            str(reply.get("line", "")),
            float(reply.get("answer_s") or 0.0),
            float(reply.get("reference_s") or 0.0),
            float(reply.get("answer_cpu_s") or 0.0),
            float(reply.get("reference_cpu_s") or 0.0),
        )

    def _kb_pass(
        self, tier: str, task: Path, answer: Path, settings: dict[str, str], name: str
    ) -> Pass:
        budget = MEDIUM_BUDGET if tier == "medium" else SCALE_BUDGET
        scale = "none" if tier == "real" else "auto"
        # one directory per (sizes, trials): the judge keys its reference cache by task, scale,
        # rule and precision, not by budget
        sub = {"scale": "b13_t1", "scale_cap": "b13_t1", "medium": "b17_t1", "real": "real_t1"}
        payload = {
            "op": "kernelbench",
            "task": str(task),
            "answer": str(answer),
            "out": str(self.cache_dir / "kb" / sub[tier]),
            "report": name,
            "scale": scale,
            "budget": budget,
            "trials": 1,
            "settings": settings,
        }
        return self._call(payload, tier)

    def _kernelbench(
        self, task: Path, answer: Path, source: str, settings: dict[str, str], contract: str
    ) -> Verdict:
        fmt = "kernelbench"
        done: list[Pass] = []

        def run(tier: str, path: Path = answer) -> Pass:
            p = self._kb_pass(tier, task, path, settings, f"{task.stem}.{path.stem}")
            done.append(p)
            return p

        def real_fits(p: Pass) -> bool:
            note = p.record.get("scaling") or {}
            full = note.get("footprint_full")
            return (
                isinstance(full, int)
                and full <= self.real_max
                and int(note.get("params_full", 0) or 0) <= 32 * self.real_max
            )

        def scaled(p: Pass) -> bool:
            return int((p.record.get("scaling") or {}).get("factor", 1) or 1) > 1

        def rejected(p: Pass) -> Verdict:
            """A scaled tier said no: under the task contract the task's own shape decides."""
            if contract == "task" and kind_of(p.record) not in HACKS and scaled(p):
                if not real_fits(p):
                    note = (
                        "the rejection is at a smaller shape only, and the task's own is too large"
                    )
                    return self._final(done, "not_judged", "not_at_task_shape", p.tier, fmt, note)
                return self._decide(done, run("real"), fmt)
            return self._decide(done, p, fmt)

        s = run("scale")
        if s.verdict == "no_kernel" and scaled(s) and "medium" in self.tiers:
            # a size-dependent dispatch: an answer may fall back to PyTorch below a size threshold
            # that only the shrunken shape crosses, so whether it launches Triton is decided at
            # the larger tier
            m = run("medium")
            if m.verdict != "no_kernel":
                return rejected(m) if rejects(m, fmt) else self._decide(done, m, fmt)
            return rejected(m)
        if rejects(s, fmt):
            return rejected(s)
        if s.verdict != "verified" and not races_unchecked(s):
            return self._decide(done, s, fmt)  # an abstention, or an error of the version
        if s.verdict == "verified" and self.cap:
            capped = capped_source(source, self.cap)
            if capped is not None:
                path = _put(answer.with_name(f"{answer.stem}.c{self.cap}.py"), capped)
                c = run("scale_cap", path)
                if c.verdict == "unsafe":
                    return rejected(c)
        best = s
        if "medium" in self.tiers and scaled(s):
            m = run("medium")
            if rejects(m, fmt):
                return rejected(m)
            if m.verdict == "verified" or (races_unchecked(m) and s.verdict == "verified"):
                best = m
            else:
                return self._decide(done, m, fmt)
        if best.verdict != "verified" and s.verdict != "verified":
            return self._decide(done, best, fmt)  # races unchecked at every tier: abstain
        # verified, or races unchecked at `medium` only, which `scale` checked
        decided = best if best.verdict == "verified" else s
        if "real" in self.tiers and scaled(s) and real_fits(s):
            r = run("real")
            if rejects(r, fmt) or r.verdict == "verified":
                return self._decide(done, r, fmt)
        return self._final(done, "verified", None, decided.tier, fmt, decided.line)

    def _tritonbench(self, task: Path, answer: Path, source: str) -> Verdict:
        fmt = "tritonbench"
        done: list[Pass] = []

        def run(tier: str, path: Path) -> Pass:
            payload = {
                "op": "tritonbench",
                "task": str(task),
                "answer": str(path),
                "out": str(self.cache_dir / "tb"),
                "report": f"{task.stem}.{path.stem}",
            }
            p = self._call(payload, tier)
            done.append(p)
            return p

        t = run("test", answer)
        if t.verdict == "verified" and self.cap:
            capped = capped_source(source, self.cap)
            if capped is not None:
                path = _put(answer.with_name(f"{answer.stem}.c{self.cap}.py"), capped)
                c = run("test_cap", path)
                if c.verdict == "unsafe":
                    return self._decide(done, c, fmt)
        return self._decide(done, t, fmt)

    # -------------------------------------------------- the verdict

    def _decide(self, done: list[Pass], p: Pass, fmt: str) -> Verdict:
        kind = kind_of(p.record, fmt)
        return self._final(done, p.verdict, kind, p.tier, fmt, p.line, p.record)

    def _final(
        self,
        done: list[Pass],
        verdict: str,
        kind: str | None,
        tier: str,
        fmt: str,
        line: str,
        record: Mapping[str, Any] | None = None,
    ) -> Verdict:
        cpu, gate = scores(verdict, kind)
        score = gate if self.mode == "gate" else cpu
        if kind == "native_code":
            line = (
                "the answer builds C++ or CUDA of its own (`torch.utils.cpp_extension`): the judge "
                "runs Triton kernels only, and does not run this one"
            )
        if kind == "compare_dtype":
            line = (
                "the judge could not compare the output with the reference's, whose dtype differs "
                f"({str((record or {}).get('message', ''))[len('RuntimeError: ') :]}); the "
                "KernelBench-v3 rule compares them cast to fp32, which this judge does not yet do"
            )
            record = {}  # not the answer's fault: no note about the sizes
        text = feedback(verdict, tier, line, record or {}, self.cap)
        if score is None:
            text = f"(the judge abstains) {text}"
        full = {
            "format": fmt,
            "passes": [
                {"tier": p.tier, "verdict": p.verdict, "kind": kind_of(p.record, fmt),
                 "line": p.line, "answer_s": p.answer_s, "reference_s": p.reference_s,
                 "answer_cpu_s": p.answer_cpu_s, "reference_cpu_s": p.reference_cpu_s,
                 "record": p.record}
                for p in done
            ],
            "wall_s": round(sum(p.answer_s for p in done), 2),
            "reference_s": round(sum(p.reference_s for p in done), 2),
            "reference_cpu_s": round(sum(p.reference_cpu_s for p in done), 2),
        }  # fmt: skip
        seconds = round(sum(p.answer_cpu_s for p in done), 2)
        return Verdict(score, verdict, kind, text, tier, seconds, full)


def feedback(verdict: str, tier: str, line: str, record: Mapping[str, Any], cap: int | None) -> str:
    """One paragraph for the model: the judge's own message (kernel, buffer, line, the source
    line, the fix), the sizes it was found at, and why those sizes count."""
    text = str(record.get("message") or line) if verdict == "unsafe" else line
    text = _COPIES.sub("answer.py", text)  # the judge's content-addressed copies of the answer
    if tier in ("scale_cap", "test_cap"):
        text += (
            f" (found with the launch's block sizes capped at {cap}, so that this test launches "
            "many program instances, as the task's real sizes do)"
        )
    note = record.get("scaling") or {}
    factor = int(note.get("factor", 1) or 1)
    sizes = record.get("answer_overrides") or {}
    if verdict in ("wrong", "unsafe", "error") and factor > 1:
        given = ", ".join(
            f"the literal {k[1:]} as {v}" if k.startswith("#") else f"{k}={v}"
            for k, v in sizes.items()
        )
        text += f" [at the task's sizes divided by {factor}{': ' + given if given else ''}]"
        text += (
            " The kernel must compute the operator at any size of the task, not only at the "
            "benchmark's."
        )
    return text.strip()
