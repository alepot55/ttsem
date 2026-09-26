"""Subprocess entry point for ``ttsem.judge``: one KernelBench answer, judged in this process.

Not meant to be imported: ``ttsem.judge`` runs it, under ``bwrap`` and with a timeout, as

    python -m ttsem._judge_runner TASK.py ANSWER.py OUT.json [--scale auto|none] [--budget N]
                                  [--trials 5] [--seed 42] [--atol 1e-2] [--rtol 1e-2]
                                  [--precision fp32|keep] [--rule kernelbench|v3] [--gpu NAME]

KernelBench's correctness rule (`src/kernelbench/eval.py`, `eval_kernel_against_ref` and
`run_and_check_correctness`, read at commit 423217d): seed 42 for the init inputs and both
models' weights, five trials whose seeds are drawn from `torch.manual_seed(42)`, fp32 inputs,
the reference first, then `ModelNew` on the same inputs, a shape check, then
`torch.allclose(atol, rtol)`; every trial must pass. The only change: every Triton launch is
compiled by the real frontend and executed by ttsem on CPU tensors (`sanitize.session`), which
stops the run at an out-of-bounds access, a race between program instances or a use of memory
nobody wrote, with the kernel's source line.

With `--rule v3`, the rule of the KernelBench-v3 harness instead (github.com/Infatoshi/
KernelBench-v3, `src/eval/benchmark.py`, read on 23 Sep 2026): the two models are built one after
the other from the same random stream (so they draw different weights), the solution then gets the
reference's weights by name (`load_state_dict(strict=False)`; what it holds under other names stays
its own, and is listed as `not_copied`), inputs keep the task's dtypes, seeds 42, 123, 456, 789
and 1337, outputs must be finite, and `max|diff| < atol + rtol * max|ref|` with atol = rtol =
1e-3 for fp32 inputs (1e-2 for fp16 and bf16, 0.1 and 0.05 for fp8). Its repeatability check
(two runs bitwise equal) is not repeated: under the semantics every run is deterministic, and a
race between program instances is reported as such.

Verdicts: verified | wrong | unsafe | no_kernel | error (with `why: shared_memory` when every
config of an autotuner was skipped and the one it falls back to needs more shared memory than the
GPU has, by Triton 3.8.0's count), and, when the tool cannot judge, not_judged (an op ttsem does
not model, an internal error, the memory cap, a full /tmp, an IR trace over `TTSEM_DUMP_MB`, a
broken task). The report records the rule, the precision and the GPU it was judged under
(`settings`).
"""

from __future__ import annotations

import argparse
import errno
import importlib.util
import json
import os
import resource
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.overrides import TorchFunctionMode
from triton.compiler.errors import CompileTimeAssertionFailure
from triton.runtime.errors import OutOfResources, PTXASError

from ttsem import judge_shapes as shapes
from ttsem import sanitize

PACKAGE = Path(__file__).resolve().parent
# the judge's own files: an exception raised in them is the answer's or the task's (a result
# that cannot be compared, a task without `Model`), not a failure of the semantics
JUDGE_FILES = {"_judge_runner.py", "judge_shapes.py"}

# the high-level PyTorch compute ops of KernelBench's own static checker
# (`kernel_static_checker.TORCH_COMPUTATION_OPS` and its `F.*` patterns), plus the matmul
# and reduction spellings a forward can reach them by
COMPUTE = {
    "mm", "bmm", "matmul", "__matmul__", "einsum", "addmm", "baddbmm", "linear", "conv1d",
    "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "avg_pool1d", "avg_pool2d", "avg_pool3d", "max_pool1d", "max_pool2d", "max_pool3d",
    "adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_avg_pool3d", "adaptive_max_pool1d",
    "adaptive_max_pool2d", "adaptive_max_pool3d", "relu", "hardtanh", "elu", "selu",
    "leaky_relu", "gelu", "softsign", "softplus", "softmax", "log_softmax", "tanh", "sigmoid",
    "hardsigmoid", "silu", "mish", "batch_norm", "group_norm", "layer_norm", "instance_norm",
    "rms_norm", "normalize", "cross_entropy", "kl_div", "mse_loss", "huber_loss",
    "smooth_l1_loss", "triplet_margin_loss", "cosine_similarity", "logsumexp", "clamp",
    "dropout", "scaled_dot_product_attention", "sum", "mean", "max", "min", "prod", "cumsum",
    "cumprod", "argmax", "argmin", "norm", "var", "std", "exp", "log", "sqrt", "pow",
}  # fmt: skip


class TorchOps(TorchFunctionMode):
    """Counts the PyTorch compute ops the answer's forward calls itself (a kernel launch is not
    one: the launch goes through `JITFunction.run`, not through torch)."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: Counter[str] = Counter()

    def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        name = getattr(func, "__name__", "")
        if name in COMPUTE:
            self.seen[name] += 1
        return func(*args, **(kwargs or {}))


def cap_memory() -> None:
    """An address-space limit for this run (`TTSEM_MEM_GB`, default 6): an answer that asks for
    more gets a MemoryError instead of pushing the machine into swap."""
    limit = int(float(os.environ.get("TTSEM_MEM_GB", "6")) * (1 << 30))
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def first_line(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {text[0] if text else ''}"[:300]


def in_answer(frame: traceback.FrameSummary, own: set[Path]) -> bool:
    """Whether `frame` runs the answer's own code: its file, or its copy at the scaled sizes."""
    return Path(frame.filename).resolve() in own


def answer_frame(exc: BaseException, own: set[Path]) -> traceback.FrameSummary | None:
    """The innermost frame of `exc`'s traceback that runs the answer's own code."""
    found = [f for f in traceback.extract_tb(exc.__traceback__) if in_answer(f, own)]
    return found[-1] if found else None


def raised_by_answer(exc: BaseException, own: set[Path]) -> bool:
    """Whether the answer's own code raised `exc` (its innermost frame is the answer's): what it
    raises itself, a full disk or a MemoryError, must not pass for a limit of the machine."""
    frames = traceback.extract_tb(exc.__traceback__)
    return bool(frames) and in_answer(frames[-1], own)


def blame(exc: BaseException, own: set[Path]) -> tuple[bool, traceback.FrameSummary]:
    """(is it the semantics' own failure, the frame that raised). A torch call of the answer
    passes through our function modes, a launch with missing arguments fails in Triton's binder
    called from the harness, and a grid the launcher would refuse fails in the harness's copy of
    its checks (`harness.check_launch_grid`): none of them is a failure of the tool. For the
    last two the frame is the answer's launch (`own`: its files), not the check's own line."""
    frames = traceback.extract_tb(exc.__traceback__)
    if "dynamic_func()" in str(exc) or frames[-1].name == "check_launch_grid":
        # the launcher's own checks, run by the harness in its place
        return False, answer_frame(exc, own) or frames[-1]
    for frame in reversed(frames):
        if frame.name != "__torch_function__":
            path = Path(frame.filename).resolve()
            return path.is_relative_to(PACKAGE) and path.name not in JUDGE_FILES, frame
    return False, frames[-1]


def load_answer(path: Path) -> Any:
    """As KernelBench loads a Triton answer: a module imported from a file."""
    spec = importlib.util.spec_from_file_location("temp_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["temp_module"] = module
    spec.loader.exec_module(module)
    return module


def cast(value: Any, precision: str) -> Any:
    """`_process_input_tensor`: with fp32 (KernelBench's default) every tensor input becomes fp32;
    with keep, the task's own dtypes stay (the KernelBench-v3 harness: its tasks set them)."""
    if precision == "fp32" and isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float32)
    return value


def scaled_copy(out: Path) -> Path:
    """Where the answer rewritten to the scaled sizes goes: next to the report, named after it,
    so that two answers with the same file name never share one."""
    return out.parent / f"{out.stem}_answer.py"


def trial_seeds(seed: int, trials: int) -> list[int]:
    torch.manual_seed(seed)
    return [int(torch.randint(0, 2**32 - 1, (1,)).item()) for _ in range(trials)]


def judge(args: argparse.Namespace, report: dict[str, Any]) -> None:
    report["stage"] = "task"  # the task itself (a package it imports may be missing here)
    task_src = args.task.read_text(encoding="utf-8")
    overrides: dict[str, shapes.Size] = {}
    if args.scale == "auto":
        overrides, note = shapes.scaled(task_src, args.budget)
        report["scaling"] = note
    original = shapes.roots(task_src)
    report["shapes"] = {k: [original.get(k), v] for k, v in overrides.items()}
    context: dict[str, Any] = {}
    exec(shapes.rewrite(task_src, overrides), context)  # noqa: S102  inside the sandbox
    Model, get_init_inputs, get_inputs = (
        context["Model"],
        context["get_init_inputs"],
        context["get_inputs"],
    )
    # an answer that keeps its own copy of the task's sizes gets the scaled ones too, in a copy
    # of the file with the same line numbers (the fault messages point into it)
    answer_src = args.answer.read_text(encoding="utf-8")
    answer_run = shapes.rewrite(answer_src, overrides, only_if=original)
    own = shapes.roots(answer_src)
    report["answer_overrides"] = {
        n: v for n, v in overrides.items() if n in own and own[n] == original.get(n)
    }
    answer = args.answer
    if answer_run != answer_src:
        answer = scaled_copy(args.out)
        answer.write_text(answer_run, encoding="utf-8")
    precision = "keep" if args.rule == "v3" else args.precision

    with sanitize.session(source=str(answer)) as state:
        watch_launches(state, GPUS.get(args.gpu, DEFAULT_GPU))
        report["stage"] = "reference"
        torch.manual_seed(args.seed)
        init_inputs = [cast(x, precision) for x in get_init_inputs()]
        with torch.no_grad():
            torch.manual_seed(args.seed)
            reference = Model(*init_inputs)

        report["stage"] = "load"
        module = load_answer(answer)
        entry = "ModelNew" if hasattr(module, "ModelNew") else "Model"
        report["entry"] = entry

        report["stage"] = "init"
        with torch.no_grad():
            if args.rule != "v3":  # KernelBench seeds both models alike; the v3 harness does not
                torch.manual_seed(args.seed)
            candidate = getattr(module, entry)(*init_inputs)
            if args.rule == "v3":  # the v3 harness copies the reference's weights over, by name
                reference, candidate = reference.eval(), candidate.eval()
                try:
                    keys = candidate.load_state_dict(reference.state_dict(), strict=False)
                    report["not_copied"] = sorted(keys.missing_keys)
                except Exception as exc:  # noqa: BLE001  as the harness: a mismatch is ignored
                    report["not_copied"] = [f"load_state_dict failed: {type(exc).__name__}"]
            elif precision == "fp32":
                reference = reference.to(dtype=torch.float32)
                candidate = candidate.to(dtype=torch.float32)
        report["launches_init"] = state.launches

        report["stage"] = "forward"
        ops = TorchOps()
        trials: list[dict[str, Any]] = []
        report["trials"] = trials
        seeds = V3_SEEDS if args.rule == "v3" else trial_seeds(args.seed, args.trials)
        with torch.no_grad():
            for seed in seeds:
                start = time.time()
                torch.manual_seed(seed)
                inputs = [cast(x, precision) for x in get_inputs()]
                torch.manual_seed(seed)
                want = reference(*inputs)
                torch.manual_seed(seed)
                before = state.launches
                with ops:
                    got = candidate(*inputs)
                row = compare(want, got, inputs, args)
                row.update(seed=seed, launches=state.launches - before)
                row["seconds"] = round(time.time() - start, 1)
                trials.append(row)
                if not row["pass"]:
                    break
            passed = all(t["pass"] for t in trials) and len(trials) == len(seeds)
            if passed and TUNERS:
                report["stage"] = "configs"
                report["configs"] = {"tuners": len(TUNERS), "checked": 0}
                sweep_configs(reference, candidate, get_inputs, seeds, args, report["configs"])
        report["launches"] = state.launches
        report["races_unchecked"] = state.races_unchecked
        report["gpu"] = args.gpu if args.gpu in GPUS else "unknown"
        report["shared_unknown"] = state.shared_unknown
        report["torch_ops"] = dict(ops.seen)
        report["atomics"] = sorted(ATOMIC_KERNELS)
    report.pop("stage")
    if not any(t["launches"] for t in trials):  # a launch at init only does not do the task
        report["verdict"] = "no_kernel"
    elif passed and "failed" not in report.get("configs", {}):
        report["verdict"] = "verified"
    else:
        report["verdict"] = "wrong"


TUNERS: list[Any] = []  # every autotuner the answer ran, in the order it first ran
ATOMIC_KERNELS: set[str] = set()  # launched kernels whose source updates memory atomically

# The GPUs the datasets name: (compute capability, the most shared memory one block may use,
# `cudaDevAttrMaxSharedMemoryPerBlockOptin`, which is what Triton's runtime compares its count
# with before a launch). An answer whose GPU is not known gets Hopper's: 227 KB is the most any
# of these GPUs gives a block, so a config above it runs on none of them.
GPUS = {
    "RTX3090": (86, 101376),
    "RTX4070": (89, 101376),
    "RTX4090": (89, 101376),
    "L40S": (89, 101376),
    "A100": (80, 166912),
    "H100": (90, 232448),
    "B200": (100, 232448),
}
DEFAULT_GPU = GPUS["H100"]


def watch_launches(state: Any, gpu: tuple[int, int]) -> None:
    """Note each autotuner that runs, and each launched kernel that uses atomics (their order,
    hence a float sum's rounding, varies from run to run on a GPU), for the rest of the process.
    While an autotuner runs, a configuration whose kernel needs more shared memory than `gpu`
    has raises `OutOfResources` instead of running (`sanitize.Session.shared_limit`): its own
    benchmarking then skips it, as on the device, and `sweep_configs` marks it not applicable.
    When it skips every config, its launch with the one it falls back to raises the same, and
    the answer's verdict is `error` (`shared_memory`), as that launch would fail on the device.

    The count is Triton 3.8.0's, and it moves between versions (a KernelBench-v3 conv that ran
    on a B200 needs 393,232 B by 3.8.0's count, over the B200's 232,448): an `error`
    (`shared_memory`) can disagree with a label produced under another Triton. A launch outside
    an autotuner is left unchecked, by choice of scope: the limit is applied only where it
    decides which config the device runs."""
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction

    tuner_run, jit_run = Autotuner.run, JITFunction.run

    def run_tuner(self: Any, *args: Any, **kwargs: Any) -> Any:
        if self not in TUNERS:
            TUNERS.append(self)
        saved, state.shared_limit = state.shared_limit, gpu
        try:
            return tuner_run(self, *args, **kwargs)
        finally:
            state.shared_limit = saved

    def run_jit(self: Any, *args: Any, **kwargs: Any) -> Any:
        if "atomic_" in (getattr(self, "src", "") or ""):
            ATOMIC_KERNELS.add(getattr(self, "__name__", "?"))
        return jit_run(self, *args, **kwargs)

    Autotuner.run, JITFunction.run = run_tuner, run_jit


def tuned() -> list[dict[str, Any]]:
    """Per autotuner that ran: its config count, the config it last ran with, and the configs
    its own benchmarking skipped (`OutOfResources`: more shared memory than the GPU has; or a
    config that does not compile)."""
    rows = []
    for tuner in TUNERS:
        timings = getattr(tuner, "configs_timings", None) or {}
        skipped = [
            str(config)
            for config, t in timings.items()
            if (t[0] if isinstance(t, (list, tuple)) else t) == float("inf")
        ]
        best = getattr(tuner, "best_config", None)
        rows.append(
            {
                "kernel": getattr(getattr(tuner, "base_fn", None), "__name__", "?"),
                "configs": len(tuner.configs),
                "ran": None if best is None else str(best),
                "skipped": skipped,
            }
        )
    return rows


# What Triton's own autotuner takes to mean that a config cannot run on the GPU, and skips
# (3.8.0, `Autotuner._bench`)
CONFIG_ERRORS = (OutOfResources, CompileTimeAssertionFailure, PTXASError)


def not_applicable(index: int, config: Any, exc: BaseException) -> dict[str, Any]:
    """A config the autotuner skips, as the sweep lists it: the reason, and for shared memory
    Triton's count and the limit, for a compile error what went wrong last."""
    entry: dict[str, Any] = {"index": index, "config": str(config)}
    if isinstance(exc, OutOfResources):  # the only resource the session checks
        entry.update(reason="shared_memory", shared=exc.required, limit=exc.limit)
    else:
        lines = [ln.strip() for ln in str(exc).strip().splitlines() if ln.strip()]
        entry.update(
            reason="ptxas" if isinstance(exc, PTXASError) else "compile",
            message=first_line(exc),
            detail=" | ".join(lines[-2:])[:300],
        )
    return entry


def sweep_configs(
    reference: Any,
    candidate: Any,
    get_inputs: Any,
    seeds: list[int],
    args: argparse.Namespace,
    done: dict[str, Any],
) -> None:
    """The trials ran with each autotuner's first config that fits the GPU (here every config
    times the same, so the first wins); a GPU's autotuner picks the fastest, which depends on the
    card. So every other config is forced in turn, on the first trial's inputs, and compared
    under the same rule: an answer is only correct if it is correct under the config the
    autotuner may pick. A config the autotuner skips cannot be picked (`CONFIG_ERRORS`: its
    kernel needs more shared memory than the GPU has, or does not compile): it is listed as not
    applicable, with the reason (and, for shared memory, Triton's count and the limit). Whether
    a config compiles is judged for sm_90a, the target every launch here is compiled for,
    whatever `--gpu` says: only the shared-memory limit is the GPU's.

    The outcome goes into `done` (the report's `configs`) as it comes: while a config is
    forced, `forcing` names it, so a fault or an error that ends the run under it says which.
    Only a sweep that ends puts back the autotuners' state, so that the report's `autotune` says
    what the trials ran with; after a fault it names the config forced."""
    precision = "keep" if args.rule == "v3" else args.precision
    torch.manual_seed(seeds[0])
    inputs = [cast(x, precision) for x in get_inputs()]
    torch.manual_seed(seeds[0])
    want = reference(*inputs)
    trial = [list(t.cache.values()) for t in TUNERS]  # the configs the trials ran with
    saved = [(dict(t.cache), getattr(t, "best_config", None)) for t in TUNERS]
    most = max(len(t.configs) for t in TUNERS)
    for index in range(most):
        picks = [t.configs[min(index, len(t.configs) - 1)] for t in TUNERS]
        if all(all(c == pick for c in ran) for pick, ran in zip(picks, trial, strict=True)):
            continue  # what the trials already judged
        for tuner, pick in zip(TUNERS, picks, strict=True):
            for key in list(tuner.cache):
                tuner.cache[key] = pick
        torch.manual_seed(seeds[0])
        done["forcing"] = {"index": index, "config": str(picks[0])}
        try:
            got = candidate(*inputs)
        except CONFIG_ERRORS as exc:
            del done["forcing"]
            done.setdefault("not_applicable", []).append(not_applicable(index, picks[0], exc))
            continue
        del done["forcing"]
        row = compare(want, got, inputs, args)
        done["checked"] += 1
        if not row["pass"]:
            done["failed"] = {
                "index": index,
                "config": str(picks[0]),
                **{k: v for k, v in row.items() if k != "pass"},
            }
            break
    for tuner, (cache, best) in zip(TUNERS, saved, strict=True):
        tuner.cache.clear()
        tuner.cache.update(cache)
        tuner.best_config = best


V3_SEEDS = [42, 123, 456, 789, 1337]
V3_TOLERANCE = {  # KernelBench-v3 `src/eval/benchmark.py`, by the dtype of the first input
    "fp4": (0.5, 0.1),
    "fp8": (0.1, 0.05),
    "fp16": (0.01, 0.01),
    "bf16": (0.01, 0.01),
    "fp32": (0.001, 0.001),
}


def v3_precision(inputs: list[Any]) -> str:
    for value in inputs:
        if isinstance(value, torch.Tensor):
            text = str(value.dtype)
            for key, name in (("float8", "fp8"), ("bfloat16", "bf16"), ("float16", "fp16")):
                if key in text:
                    return name
            return "fp32" if "float32" in text else text.replace("torch.", "")
    return "fp32"


def compare(want: Any, got: Any, inputs: list[Any], args: argparse.Namespace) -> dict[str, Any]:
    """One trial under the chosen rule (`pass`), with the other rules' answers alongside:
    KernelBench's elementwise `torch.allclose` at 1e-2 (v0.1) and 1e-4 (its fp32 default now), and
    KernelBench-v3's `max|diff| < atol + rtol * max|ref|` with finite outputs."""
    row: dict[str, Any] = {}
    if not isinstance(want, torch.Tensor) or not isinstance(got, torch.Tensor):
        row.update(pass_=False, why="not a tensor")
    elif got.shape != want.shape:
        row.update(pass_=False, shape=[list(want.shape), list(got.shape)])
    else:
        w, g = want.float(), got.float()
        diff = (w - g).abs()
        max_diff = float(diff.max()) if diff.numel() else 0.0
        max_ref = float(w.abs().max()) if w.numel() else 0.0
        atol, rtol = V3_TOLERANCE.get(v3_precision(inputs), (0.05, 0.02))
        finite = bool(torch.isfinite(w).all()) and bool(torch.isfinite(g).all())
        row.update(
            max_abs_diff=max_diff,
            max_ref=max_ref,
            ref_nan=bool(torch.isnan(w).any()),
            allclose=bool(torch.allclose(want, got, atol=args.atol, rtol=args.rtol)),
            allclose_1e4=bool(torch.allclose(want, got, atol=1e-4, rtol=1e-4)),
            v3=finite and max_diff < atol + rtol * max_ref,
        )
        row["pass_"] = row["v3"] if args.rule == "v3" else row["allclose"]
    row["pass"] = row.pop("pass_")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("task", type=Path)
    ap.add_argument("answer", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--scale", choices=("auto", "none"), default="auto")
    ap.add_argument("--budget", type=int, default=shapes.BUDGET)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--rtol", type=float, default=1e-2)
    ap.add_argument("--precision", choices=("fp32", "keep"), default="fp32")
    ap.add_argument("--rule", choices=("kernelbench", "v3"), default="kernelbench")
    ap.add_argument("--gpu", default="", help="the dataset's GPU (its shared-memory limit)")
    args = ap.parse_args()
    cap_memory()
    report: dict[str, Any] = {"answer": args.answer.name, "scale": args.scale, "launches": 0}
    # what the verdict was reached under: a rerun with other settings judges again
    report["settings"] = {"rule": args.rule, "precision": args.precision, "gpu": args.gpu}
    own = {args.answer.resolve(), scaled_copy(args.out).resolve()}  # the answer's own code

    def where(frame: traceback.FrameSummary) -> str:
        """A frame as `file:line`: the scaled copy has the answer's line numbers, and is named
        after the answer."""
        name = args.answer.name if in_answer(frame, own) else Path(frame.filename).name
        return f"{name}:{frame.lineno}"

    def failed(exc: BaseException) -> None:
        """An exception of the answer's, the task's or the tool's: a full disk or the memory
        cap is a limit of the machine unless the answer's own code raised it."""
        inside, frame = blame(exc, own)
        stage = report.get("stage", "")
        limit = not raised_by_answer(exc, own)
        if limit and "can't allocate memory" in str(exc):  # torch's CPU allocator, same cap
            report.update(verdict="not_judged", why="memory")
        elif limit and isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
            # the sandbox's /tmp is a tmpfs of half the machine's RAM: what fills it says
            # something about the machine, not about the answer
            report.update(verdict="not_judged", why="disk")
        elif inside:
            report.update(verdict="not_judged", why="tool_error")
        elif stage in ("task", "reference"):
            report.update(verdict="not_judged", why="task_error")
        else:
            report.update(verdict="error")
        lines = [ln.strip() for ln in str(exc).strip().splitlines() if ln.strip()]
        report.update(
            message=first_line(exc),
            detail=" | ".join(lines[-2:])[:300],  # a compilation error says what went wrong last
            where=where(frame),
        )

    start = time.time()
    try:
        judge(args, report)
    except sanitize.KernelFault as stop:
        fault = stop.fault
        file = Path(fault.source_file).name
        report.update(
            verdict="unsafe",
            kind=fault.kind,
            kernel=fault.kernel,
            buffer=fault.buffer,
            # the scaled copy has the answer's line numbers: the line is the answer's own
            file=args.answer.name if file == scaled_copy(args.out).name else file,
            line=fault.source_line,
            source=fault.source_text,
            message=sanitize.message(fault)[:600],
        )
    except sanitize.SharedOverLimit as exc:
        # every config of an autotuner skipped, then its launch with the config it falls back
        # to needs more shared memory than the GPU has: on the GPU, with this Triton, it fails
        import triton

        gpu = args.gpu if args.gpu in GPUS else "H100 (the default GPU)"
        report.update(
            verdict="error",
            why="shared_memory",
            message=(
                f"every autotune config of `{exc.kernel}` was skipped, and the one it falls back "
                f"to needs {exc.required:,} B of shared memory, over the {gpu}'s {exc.limit:,} B "
                f"(Triton {triton.__version__}'s count)"
            ),
            shared=exc.required,
            limit=exc.limit,
        )
        frame = answer_frame(exc, own)
        if frame is not None:
            report["where"] = where(frame)
    except sanitize.Unjudged as stop:
        if raised_by_answer(stop, own):  # the tool's own exception, raised by the answer
            failed(stop)
        else:
            report.update(verdict="not_judged", why=stop.verdict, message=stop.detail[:300])
    except MemoryError as exc:
        if raised_by_answer(exc, own):
            failed(exc)
        else:
            report.update(verdict="not_judged", why="memory", message="over the run's memory cap")
    except BaseException as exc:  # noqa: BLE001  the answer may raise anything
        failed(exc)
    if TUNERS:  # whatever the verdict: under which config it was reached, and what was skipped
        report["autotune"] = tuned()
    report["seconds"] = round(time.time() - start, 1)
    args.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
