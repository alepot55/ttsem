"""The kernel stage of the checks: a program's Triton kernel against torch on the GPU, then timed.

``check`` runs the kernel ``codegen_triton`` generates for a program on CUDA, on lengths around its
block (one element, a row shorter than a block, one element either side of a block, several
blocks and a tail), against a float64 torch reference, and reports the worst error by the
problem's own measure. ``bench`` times the kernel next to torch's own operators, ``fitness`` alone
on one shape. A timing is the median of ``REPEATS`` medians of ``triton.testing.do_bench`` (L2
flushed before every run), each after half a second of the function run back to back; it keeps
each median, their range and how many runs each timed. The warm-up is there because a laptop GPU
that idled for a few seconds is at its lowest clocks, and do_bench's own warm-up, a number of
runs estimated at those clocks, is too short to raise them: a single do_bench after 5-20 s of
idle read the attention kernel up to 40% slow. The range is the spread the warm-up leaves: after
10 s of idle, within 0.1% for the online softmax at 1024 x 32,768, 1-10% for the online attention
at 1024 x 16,384 x 64. The GPU is the clock: every use of it here holds a lock shared by threads
and processes, so no two timings, nor a timing and a check, overlap, nor does one come between a
warm-up and its timing.

The generated module is written to ``runs/kernels/<problem>-<hash of the source>.py`` and imported
from there, so that Triton reads real source and a reviewer can open the file.

    python -m discovery.gpu --problem softmax --program online --bench
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import torch.nn.functional as F
import triton.testing

from discovery import codegen_triton, ir
from discovery.evaluate import FP_TOL
from discovery.problems import PROBLEMS, Problem

KERNELS = Path(__file__).resolve().parent / "runs" / "kernels"
BENCH_SHAPES = {
    "rows": [(1024, 4096), (1024, 32768)],
    "attention": [(1024, 4096, 64), (1024, 16384, 64)],
}
ATTENTION_DIM = 64

# --------------------------------------------------------------------------- the kernel module

_LOADED: dict[Path, ModuleType] = {}


def write(prog: ir.Program, problem: Problem, block: int | None = None) -> Path:
    """The file of the generated module, written once: its name is the hash of its source."""
    source = codegen_triton.generate(prog, problem, block)
    path = KERNELS / f"{problem.name}-{hashlib.sha256(source.encode()).hexdigest()[:16]}.py"
    if not path.exists():
        KERNELS.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(source, encoding="utf-8")
        os.replace(tmp, path)  # never a half-written module for another process
    return path


def load(path: Path) -> ModuleType:
    """Import a generated module from its file, once per process."""
    if path not in _LOADED:
        spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LOADED[path] = module
    return _LOADED[path]


def kernel(prog: ir.Program, problem: Problem, block: int | None = None) -> ModuleType:
    return load(write(prog, problem, block))


# --------------------------------------------------------------------------- the GPU, alone

_THREADS = threading.Lock()

if sys.platform == "win32":
    import msvcrt

    def _lock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)  # tries for 10 s, then raises
                return
            except OSError:
                continue

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


_holder: int | None = None  # the thread inside exclusive(), if any


@contextlib.contextmanager
def exclusive() -> Iterator[None]:
    """The GPU to the caller alone, across threads and processes. Not reentrant."""
    global _holder
    KERNELS.mkdir(parents=True, exist_ok=True)
    with _THREADS, open(KERNELS / "gpu.lock", "a+b") as fh:
        _lock(fh.fileno())
        _holder = threading.get_ident()
        try:
            yield
        finally:
            _holder = None
            _unlock(fh.fileno())


# --------------------------------------------------------------------------- inputs, references


def make_inputs(
    problem: Problem,
    shape: tuple[int, ...],
    family: str = "normal",
    seed: int = 0,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Float32 inputs of ``shape``: ``(rows, N)``, or ``(rows, N, D)`` for attention, every row
    drawn from the problem's input ``family``."""
    rng = np.random.default_rng(seed)
    if codegen_triton.layout(problem) == "rows":
        rows, n = shape
        draws = [problem.generate(rng, n, family) for _ in range(rows)]
        data = {f: np.stack([d[f][:, 0] for d in draws]) for f in problem.elems}
    else:
        rows, n, d = shape
        data = _attention_inputs(problem, rows, n, d, family, rng)
    return {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in data.items()}


def _attention_inputs(
    problem: Problem, rows: int, n: int, d: int, family: str, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """q, k, v whose scores q.k/sqrt(d) have, in every row, the mean and the spread of the
    problem's score family: k's first column is 1 and q's first entry carries the mean."""
    score = next(f for f, t in problem.elems.items() if t == "scalar")
    s = problem.generate(rng, 4096, family)[score][:, 0]
    k = rng.standard_normal((n, d))
    k[:, 0] = 1.0
    q = rng.standard_normal((rows, d)) * (s.std() * math.sqrt(d / (d - 1)))
    q[:, 0] = s.mean() * math.sqrt(d)
    return {"q": q, "k": k, "v": rng.standard_normal((n, d))}


def _softmax64(i: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {"y": torch.softmax(i["x"].double(), dim=-1)}


def _variance64(i: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    var, mean = torch.var_mean(i["x"].double(), dim=-1, correction=0)
    return {"mean": mean, "var": var}


def _attention64(i: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    q, k, v = (i[name].double() for name in "qkv")
    return {"o": torch.softmax(q @ k.T / math.sqrt(q.shape[1]), dim=-1) @ v}


# the operators in float64, from the float32 inputs
REFERENCE: dict[str, Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]] = {
    "softmax": _softmax64,
    "variance": _variance64,
    "attention": _attention64,
}


def _sdpa(i: dict[str, torch.Tensor]) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        i["q"][None, None], i["k"][None, None], i["v"][None, None]
    )


def _plain_attention(i: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.softmax(i["q"] @ i["k"].T / math.sqrt(i["q"].shape[1]), dim=-1) @ i["v"]


# torch's own operators, float32 (matmul without TF32, torch's default)
TORCH: dict[str, dict[str, Callable[[dict[str, torch.Tensor]], object]]] = {
    "softmax": {"torch.softmax": lambda i: torch.softmax(i["x"], dim=-1)},
    "variance": {"torch.var_mean": lambda i: torch.var_mean(i["x"], dim=-1, correction=0)},
    "attention": {"sdpa": _sdpa, "q@k.T softmax @v": _plain_attention},
}


def errors(
    problem: Problem, out: dict[str, torch.Tensor], ref: dict[str, torch.Tensor]
) -> list[float]:
    """The problem's error measure (the one the semantics tests use), row by row."""
    a = {k: v.double().cpu().numpy() for k, v in out.items()}
    r = {k: v.double().cpu().numpy() for k, v in ref.items()}
    rows = next(iter(r.values())).shape[0]
    return [
        problem.error({k: v[i] for k, v in a.items()}, {k: v[i] for k, v in r.items()}, {})
        for i in range(rows)
    ]


# --------------------------------------------------------------------------- check


def held_out(problem: Problem, block: int) -> list[tuple[int, ...]]:
    """The shapes of the kernel's check: 3 rows of lengths around its block (one element, a row
    shorter than a block, one element either side of a block, several blocks and a tail, many
    blocks), for attention with D = 64, and one more shape with D = 16. They exercise the
    kernel's loop and masks; none is a shape the kernel was tuned or timed on. Not every length
    is new to the program: the search's numpy tests also run N = 1 and 7, and N = 2048, the rows
    block. The attention shapes are new, since those tests use D = 8."""
    ns = sorted({1, 7, block - 1, block, block + 1, 3 * block + 5, 20 * block + 17} - {0})
    if codegen_triton.layout(problem) == "rows":
        return [(3, n) for n in ns]
    return [(3, n, ATTENTION_DIM) for n in ns] + [(3, 2 * block + 3, 16)]


def check(
    prog: ir.Program,
    problem: Problem,
    shapes: list[tuple[int, ...]] | None = None,
    seed: int = 0,
    *,
    family: str = "normal",
    block: int | None = None,
    tol: float = FP_TOL,
) -> dict:
    """Run the kernel of ``prog`` on CUDA on ``shapes`` (default: :func:`held_out`) against the
    float64 torch reference. ``ok`` when the worst error, row by row, is within ``tol``."""
    path = write(prog, problem, block)
    module = load(path)
    shapes = shapes or held_out(problem, module.BLOCK)
    result: dict = {"kernel": str(path), "block": module.BLOCK, "family": family, "tol": tol}
    per_shape = []
    with exclusive():
        for i, shape in enumerate(shapes):
            inputs = make_inputs(problem, shape, family, seed + i)
            try:
                out = module.run(inputs)
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001  a kernel that does not compile or run
                detail = str(exc).strip().splitlines()[-1:] or [type(exc).__name__]
                return {
                    **result,
                    "ok": False,
                    "worst": math.inf,
                    "shapes": per_shape,
                    "message": f"the Triton kernel of this program fails on shape {_shape(shape)}: "
                    f"{type(exc).__name__}: {detail[0]}",
                }
            err = max(errors(problem, out, REFERENCE[problem.name](inputs)))
            per_shape.append({"shape": list(shape), "error": err})
    worst = max(per_shape, key=lambda r: r["error"])
    ok = worst["error"] <= tol
    where = f"{_shape(tuple(worst['shape']))}"
    message = (
        f"the Triton kernel matches the float64 reference on {len(per_shape)} shapes "
        f"({_lengths(shapes)}): worst error {worst['error']:.1e}, on {where}"
        if ok
        else f"the Triton kernel is off on shape {where} ({family} inputs): error "
        f"{worst['error']:.3g}, above {tol:.0e}"
    )
    return {**result, "ok": ok, "worst": worst["error"], "shapes": per_shape, "message": message}


def _shape(shape: tuple[int, ...]) -> str:
    names = ("rows", "N", "D")
    return ", ".join(f"{k}={v}" for k, v in zip(names, shape, strict=False))


def _lengths(shapes: list[tuple[int, ...]]) -> str:
    """``3 rows of N = 1, 7``; for attention ``3 rows of N = 1, 7 keys, D = 64``."""
    groups: dict[tuple[int, ...], list[str]] = {}
    for rows, n, *d in shapes:
        groups.setdefault((rows, *d), []).append(str(n))
    return "; ".join(
        f"{key[0]} rows of N = {', '.join(ns)}" + (f" keys, D = {key[1]}" if len(key) > 1 else "")
        for key, ns in groups.items()
    )


# --------------------------------------------------------------------------- bench


def traffic(prog: ir.Program, problem: Problem, shape: tuple[int, ...]) -> tuple[int, int]:
    """(bytes of the operator, bytes the kernel moves). The first counts the inputs read once and
    the outputs written once, the same for every implementation; the second what the program's
    passes load and store (every pass reads its fields again), so above the DRAM bandwidth it
    means the re-reads hit in L2."""
    reads = ir.reads_elements(prog)
    if codegen_triton.layout(problem) == "rows":
        rows, n = shape
        out = rows * n if prog.output.kind == "elementwise" else rows * len(prog.output.exprs)
        minimum = (rows * n * len(problem.elems) + out) * 4
        moved = (sum(len(r) for r in reads) * rows * n + out) * 4
        return minimum, moved
    rows, n, d = shape
    assert isinstance(problem.outputs, dict)
    out = sum(rows * (d if t == "vec" else 1) for t in problem.outputs.values())
    minimum = (rows * d + 2 * n * d + out) * 4
    moved = (rows * d + sum(len(r) for r in reads) * rows * n * d + out) * 4
    return minimum, moved


WARM_S = 0.5  # seconds of the function, back to back, before every do_bench call
REPEATS = 3  # do_bench calls per timing, each after its own warm-up
MIN_RUNS = 20  # a do_bench call that timed fewer runs is redone, up to twice


def _warm(fn: Callable[[], object]) -> float:
    """Run ``fn`` back to back, synchronizing after each run, for :data:`WARM_S` seconds and 3
    runs at least, and return the shortest wall time of a run in ms. An RTX A1000 idle for 10 s
    is at its lowest clocks (P8, 210 MHz): in the first quarter second of load a run took up to
    18 times its warm time."""
    walls = []
    start = now = time.perf_counter()
    while now - start < WARM_S or len(walls) < 3:
        fn()
        torch.cuda.synchronize()
        before, now = now, time.perf_counter()
        walls.append(now - before)
    return min(walls) * 1000


def _do_bench(fn: Callable[[], object]) -> list[float]:
    """The run times, in ms, of one ``do_bench`` call after :func:`_warm`. do_bench times
    ``int(rep / (t + flush))`` runs, with ``t + flush`` estimated on 5 runs it makes first (the
    flush zeroes 256 MiB, 2.1 ms on the RTX A1000). ``rep = max(200, 2 MIN_RUNS w)``, with ``w``
    the warm-up's shortest run (a wall time, launch included, so not shorter than ``t``), gives
    :data:`MIN_RUNS` runs or more when those 5 runs are warm and the flush is under 5 ms: fewer
    means they were not, and the call is redone."""
    for _ in range(3):
        rep = max(200.0, 2 * MIN_RUNS * _warm(fn))
        times = triton.testing.do_bench(fn, rep=rep, return_mode="all")
        if len(times) >= MIN_RUNS:
            break
    return times


def _time(fn: Callable[[], object], repeats: int = REPEATS) -> dict:
    """``repeats`` medians of :func:`_do_bench`: their median ``ms``, their range ``ms_min`` to
    ``ms_max``, and in order the ``medians`` and the ``runs`` each timed. Only inside
    :func:`exclusive`, so that nothing else of ours runs on the GPU between a warm-up and its
    timing."""
    if _holder != threading.get_ident():
        raise RuntimeError("time the GPU only inside gpu.exclusive()")
    if repeats < 1:
        raise ValueError(f"repeats must be 1 or more, not {repeats}")
    medians, runs = [], []
    for _ in range(repeats):
        times = _do_bench(fn)
        medians.append(float(statistics.median(times)))
        runs.append(len(times))
    return {
        "ms": float(statistics.median(medians)),
        "ms_min": min(medians),
        "ms_max": max(medians),
        "medians": medians,
        "runs": runs,
    }


def _rates(t: dict, nbytes: int, flops: int) -> dict:
    r = {**t, "gbps": nbytes / t["ms"] / 1e6}
    if flops:
        r["tflops"] = flops / t["ms"] / 1e9
    return r


def bench(
    prog: ir.Program,
    problem: Problem,
    shapes: list[tuple[int, ...]] | None = None,
    *,
    block: int | None = None,
    seed: int = 0,
    repeats: int = REPEATS,
) -> dict:
    """The times of the kernel of ``prog`` and of torch's operators on ``shapes`` (default:
    :data:`BENCH_SHAPES`), one at a time, each as :func:`_time` gives it (the median of
    ``repeats`` do_bench medians, their range, each median and the runs it timed), and the
    bandwidth each reaches at the median: ``gbps`` from the operator's bytes, ``moved_gbps`` from
    the bytes the program's passes move (see :func:`traffic`); for attention also ``tflops``,
    from 4 rows N D operations."""
    path = write(prog, problem, block)
    module = load(path)
    lay = codegen_triton.layout(problem)
    shapes = shapes or BENCH_SHAPES[lay]
    entries = []
    with exclusive():
        for shape in shapes:
            inputs = make_inputs(problem, shape, "normal", seed)
            minimum, moved = traffic(prog, problem, shape)
            flops = 4 * math.prod(shape) if lay == "attention" else 0
            t = _time(lambda inputs=inputs: module.run(inputs), repeats)
            mine = {**_rates(t, minimum, flops), "moved_gbps": moved / t["ms"] / 1e6}
            theirs = {}
            for name, fn in TORCH[problem.name].items():
                theirs[name] = _rates(_time(lambda fn=fn, i=inputs: fn(i), repeats), minimum, flops)
            entries.append({"shape": list(shape), "kernel": mine, "torch": theirs})
    return {
        "program": prog.name,
        "problem": problem.name,
        "kernel": str(path),
        "block": module.BLOCK,
        "num_warps": module.NUM_WARPS,
        "device": torch.cuda.get_device_name(),
        "repeats": repeats,
        "shapes": entries,
    }


FITNESS_SHAPE = {"rows": (1024, 32768), "attention": (1024, 4096, ATTENTION_DIM)}


def fitness(
    prog: ir.Program,
    problem: Problem,
    shape: tuple[int, ...] | None = None,
    *,
    repeats: int = REPEATS,
) -> dict:
    """The search's measured fitness: the kernel checked on the shapes of :func:`held_out`, then
    timed alone on one shape (default :data:`FITNESS_SHAPE`) by :func:`_time`: ``ms`` is the
    median of ``repeats`` do_bench medians, ``ms_min`` and ``ms_max`` their range, ``medians``
    and ``runs`` each median and the runs it timed. ``{"ok": False, "message": ...}`` when the
    check fails."""
    result = check(prog, problem)
    if not result["ok"]:
        return {"ok": False, "message": result["message"]}
    module = load(write(prog, problem, None))
    shape = shape or FITNESS_SHAPE[codegen_triton.layout(problem)]
    with exclusive():
        inputs = make_inputs(problem, shape, "normal", 0)
        t = _time(lambda: module.run(inputs), repeats)
    return {"ok": True, **t, "shape": list(shape), "worst_error": result["worst"]}


# --------------------------------------------------------------------------- command line


def program_text(problem: Problem, spec: str) -> str:
    """The program a command line names: ``seed``, a known form of the problem, or a file."""
    if spec == "seed":
        return problem.seed
    if spec in problem.known:
        return problem.known[spec]
    path = Path(spec)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    known = ", ".join(["seed", *problem.known])
    raise SystemExit(f"--program: no file {spec!r} and no known form of that name ({known})")


def _parse_shapes(text: str) -> list[tuple[int, ...]]:
    return [tuple(int(v) for v in s.split("x")) for s in text.split(",") if s]


def _report(check_result: dict, bench_result: dict | None) -> str:
    c = check_result
    lines = [
        f"kernel: {c['kernel']} (BLOCK {c['block']})",
        f"check: {'ok' if c['ok'] else 'FAILED'}: {c['message']}",
    ]
    for r in c["shapes"]:
        lines.append(f"  {_shape(tuple(r['shape'])):<26} error {r['error']:.2e}")
    if bench_result is None:
        return "\n".join(lines)
    b = bench_result
    lines.append(
        f"bench on {b['device']}: ms is the median of {b['repeats']} medians of "
        f"triton.testing.do_bench, each after {WARM_S:g} s of warm-up, their range in "
        "parentheses; GB/s from the operator's bytes (inputs once, outputs once), moved GB/s from "
        "the program's passes"
    )
    for e in b["shapes"]:
        k = e["kernel"]
        extra = f", {k['tflops']:.2f} TFLOPS" if "tflops" in k else ""
        lines.append(
            f"  {_shape(tuple(e['shape'])):<26} {b['program']}: {_ms(k)}, "
            f"{k['gbps']:.1f} GB/s, moved {k['moved_gbps']:.1f} GB/s{extra}"
        )
        for name, t in e["torch"].items():
            extra = f", {t['tflops']:.2f} TFLOPS" if "tflops" in t else ""
            lines.append(f"  {'':<26} {name}: {_ms(t)}, {t['gbps']:.1f} GB/s{extra}")
    return "\n".join(lines)


def _ms(t: dict) -> str:
    return f"{t['ms']:.3f} ms ({t['ms_min']:.3f}-{t['ms_max']:.3f})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m discovery.gpu",
        description="Check a program's Triton kernel against torch on the GPU, and time it.",
    )
    ap.add_argument("--problem", required=True, choices=sorted(PROBLEMS))
    ap.add_argument(
        "--program", required=True, help="a file holding the program, `seed`, or a known form"
    )
    ap.add_argument("--bench", action="store_true", help="time it after the check")
    ap.add_argument("--block", type=int, help="elements per block (a power of two)")
    ap.add_argument("--shapes", help="benchmark shapes, e.g. 1024x4096,1024x32768 (x64 for D)")
    ap.add_argument(
        "--repeats",
        type=int,
        default=REPEATS,
        help=f"do_bench calls per timing, each after its own warm-up (default {REPEATS})",
    )
    ap.add_argument("--json", action="store_true", help="print the results as JSON")
    args = ap.parse_args(argv)
    if args.repeats < 1:
        ap.error("--repeats must be 1 or more")
    problem = PROBLEMS[args.problem]
    prog = problem.parse(program_text(problem, args.program))
    c = check(prog, problem, block=args.block)
    shapes = _parse_shapes(args.shapes) if args.shapes else None
    b = (
        bench(prog, problem, shapes, block=args.block, repeats=args.repeats)
        if args.bench and c["ok"]
        else None
    )
    print(json.dumps({"check": c, "bench": b}, indent=1) if args.json else _report(c, b))
    return 0 if c["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
