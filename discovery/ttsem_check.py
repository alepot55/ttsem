"""ttsem's checks on a program's kernel: accesses outside a tensor, races between program
instances, undefined values.

``check(prog, problem)`` runs the kernel ``codegen_triton`` generates on small CPU tensors (3
rows, a length that is not a multiple of the block, D = 16) under ``ttsem.sanitize.session()``,
which executes every launch with ttsem's semantics of Triton's IR instead of a GPU and stops at
the first fault, naming the op and the kernel's line. A GPU forgives a read that lands in the
allocator's padding; ttsem does not. The run is a subprocess of its own, since the session patches
Triton for the whole process.

The verdict is ``ok`` (with the error of what ttsem computed against the float64 reference),
``fault`` (with ttsem's message), ``unjudged`` (ttsem could not judge the launch, an op it does not
support for one), ``unavailable`` or ``error``. On Windows it is ``unavailable``: ttsem reads
Triton's IR in MLIR's generic form, Triton's Windows build does not export the MLIR symbols that
switch its printer to that form, and the launch stops with ``ttsem.mlir.NotGeneric``. The same
code runs the check on Linux.

    python -m discovery.ttsem_check --problem softmax --program online
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from discovery import codegen_triton, gpu, ir
from discovery.problems import PROBLEMS, Problem

ROOT = Path(__file__).resolve().parent.parent
MARK = "ttsem-check: "  # the line of the subprocess's output that carries its verdict
UNAVAILABLE = (
    "ttsem cannot check kernels with this Triton build: it reads Triton's IR in MLIR's generic "
    "form, and this build does not export the MLIR symbols that switch the printer to it "
    "(Triton's Windows build does not); the same check runs on Linux"
)


def shape_for(problem: Problem, block: int) -> tuple[int, ...]:
    """3 rows of one block and a half and 3 more elements (a masked tail), D = 16."""
    n = block + block // 2 + 3
    return (3, n) if codegen_triton.layout(problem) == "rows" else (3, n, 16)


def check(
    prog: ir.Program, problem: Problem, *, block: int | None = None, timeout: float = 600.0
) -> dict:
    """Run the kernel of ``prog`` under ttsem in a subprocess and return its verdict."""
    path = gpu.write(prog, problem, block)
    shape = shape_for(problem, block or codegen_triton.block_for(problem))
    cmd = [sys.executable, "-m", "discovery.ttsem_check", "--problem", problem.name]
    cmd += ["--worker", str(path), "--shape", "x".join(str(v) for v in shape)]
    where = {"kernel": str(path), "shape": list(shape)}
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except subprocess.TimeoutExpired:
        return {"verdict": "error", "message": f"ttsem did not finish in {timeout:.0f} s", **where}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(MARK):
            return {**json.loads(line[len(MARK) :]), **where}
    tail = (proc.stderr.strip() or proc.stdout.strip()).splitlines()[-1:] or ["no output"]
    return {
        "verdict": "error",
        "message": f"the ttsem run ended without a verdict (exit {proc.returncode}): {tail[0]}",
        **where,
    }


def _worker(path: Path, problem: Problem, shape: tuple[int, ...]) -> dict:
    """In the subprocess: the kernel's launcher on CPU tensors, every launch run by ttsem."""
    try:
        from ttsem import mlir, sanitize
    except ImportError as exc:
        return {"verdict": "unavailable", "message": f"ttsem cannot be imported here: {exc}"}
    module = gpu.load(path)
    inputs = gpu.make_inputs(problem, shape, device="cpu")
    with sanitize.session() as state:
        try:
            out = module.run(inputs)
        except sanitize.KernelFault as stop:
            return {
                "verdict": "fault",
                "kind": stop.fault.kind,
                "message": str(stop),
                "launches": state.launches,
            }
        except sanitize.Unjudged as stop:
            return {"verdict": "unjudged", "message": stop.detail, "unsupported": stop.unsupported}
        except mlir.NotGeneric as exc:
            return {"verdict": "unavailable", "message": f"{UNAVAILABLE} (NotGeneric: {exc})"}
    error = max(gpu.errors(problem, out, gpu.REFERENCE[problem.name](inputs)))
    return {
        "verdict": "ok",
        "launches": state.launches,
        "races_unchecked": state.races_unchecked,
        "error": error,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m discovery.ttsem_check",
        description="Run a program's Triton kernel under ttsem's checks, on the CPU.",
    )
    ap.add_argument("--problem", required=True, choices=sorted(PROBLEMS))
    ap.add_argument("--program", help="a file holding the program, `seed`, or a known form")
    ap.add_argument("--block", type=int, help="elements per block (a power of two)")
    ap.add_argument("--worker", help=argparse.SUPPRESS)  # the kernel's file, in the subprocess
    ap.add_argument("--shape", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    problem = PROBLEMS[args.problem]
    if args.worker:
        shape = tuple(int(v) for v in args.shape.split("x"))
        print(MARK + json.dumps(_worker(Path(args.worker), problem, shape)), flush=True)
        return 0
    if not args.program:
        ap.error("--program is required")
    prog = problem.parse(gpu.program_text(problem, args.program))
    result = check(prog, problem, block=args.block)
    print(json.dumps(result, indent=1))
    return {"ok": 0, "unavailable": 2}.get(result["verdict"], 1)


if __name__ == "__main__":
    sys.exit(main())
