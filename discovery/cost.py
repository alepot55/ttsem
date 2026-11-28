"""A roofline estimate of a program's time, from what it reads and computes per element.

Bytes: every pass over memory reads the element fields it uses (``Problem.field_bytes``), an
elementwise output also writes one float32 per element. Work: the operations inside the
reductions and the elementwise output run once per element, a vector operation counts ``D``
times; the merges run once per block and are left out. The time is the larger of the two,
row after row. On the problems here the memory time is 25 to 240 times the compute time, so
the estimate is set by the bytes moved and the exps never change it; only the GPU decides speed.
"""

from __future__ import annotations

import dataclasses

from discovery import ir
from discovery.problems import Problem


@dataclasses.dataclass(frozen=True)
class Device:
    name: str
    bandwidth: float  # bytes/s
    exp_rate: float  # exp/s
    flop_rate: float  # fp32 operations/s
    source: str


# measured on 27 Nov 2026 (copy kernel, a loop of exp, torch fp32 matmul), see DIARY.md
A1000 = Device("RTX A1000 6GB Laptop", 124e9, 5.1e11, 4.4e12, "measured")
# spec sheet; the exp rate assumes 16 SFU ops/clock/SM at 2.5 GHz on 46 SMs
RTX4070 = Device("RTX 4070", 504e9, 1.8e12, 29e12, "spec sheet, not measured")

DEVICES = {"a1000": A1000, "4070": RTX4070}


def _is_vec(e: ir.Expr, prog: ir.Program, problem: Problem) -> bool:
    for node in ir.walk(e):
        if isinstance(node, ir.Var):
            t = problem.elems.get(node.name) or prog.types.get(node.name)
            if t == "vec":
                return True
    return False


def _per_element(e: ir.Expr, prog: ir.Program, problem: Problem, dim: int) -> tuple[float, float]:
    """(exps, other operations) for one element, of the nodes of ``e``."""
    exps = flops = 0.0
    for node in ir.walk(e):
        if isinstance(node, ir.Call | ir.Reduce):
            width = dim if _is_vec(node, prog, problem) else 1
            if isinstance(node, ir.Call) and node.op in ("exp", "log"):
                exps += width
            else:
                flops += width * max(1, len(getattr(node, "args", ())) - 1)
    return exps, flops


def estimate(
    prog: ir.Program, problem: Problem, device: Device = A1000, n: int = 32768, rows: int = 1024
) -> dict:
    """Bytes, exps and operations per element, and the estimated time in microseconds for
    ``rows`` rows of ``n`` elements."""
    dim = problem.field_bytes.get("v", 4) // 4 if "v" in problem.elems else 1
    reads = ir.reads_elements(prog)
    bytes_per_elem = float(sum(problem.field_bytes[f] for fields in reads for f in fields))
    if prog.output.kind == "elementwise":
        bytes_per_elem += 4
    exps = flops = 0.0
    for p in prog.passes:
        for _, e in p.block:
            for node in ir.walk(e):
                if isinstance(node, ir.Reduce):
                    a, b = _per_element(node.body, prog, problem, dim)
                    exps, flops = (
                        exps + a,
                        flops + b + (dim if _is_vec(node.body, prog, problem) else 1),
                    )
    if prog.output.kind == "elementwise":
        a, b = _per_element(prog.output.exprs[0][1], prog, problem, dim)
        exps, flops = exps + a, flops + b
    t_mem = bytes_per_elem * n * rows / device.bandwidth
    t_cmp = (exps / device.exp_rate + flops / device.flop_rate) * n * rows
    return {
        "bytes_per_elem": bytes_per_elem,
        "exps_per_elem": exps,
        "flops_per_elem": flops,
        "passes": sum(1 for fields in reads if fields),
        "time_us": max(t_mem, t_cmp) * 1e6,
        "device": device.name,
    }
