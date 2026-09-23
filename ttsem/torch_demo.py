"""``python -m ttsem demo --torch``: silent miscompiles in ``torch.compile``, found without a GPU.

TorchInductor turns a PyTorch program into Triton kernels. When a pass gets aliasing wrong, the
kernel reads memory that another program instance of the same launch writes: on a GPU the
instances run at the same time, the result depends on the schedule, and nothing fails. Each
program below compiles to such a kernel on the torch releases that have the bug.

No GPU is needed because Inductor also emits Triton for CPU tensors (``cpu_backend = "triton"``,
the same lowering and the same passes), and :func:`ttsem.sanitize.session` executes every launch
under the semantics: a race between program instances stops the run with the kernel's own line.
``inductor_names(block=16)`` caps the block size, so a small tensor is covered by several program
instances, as a large one is on a device.

The verdict depends on the installed torch: on a release with the fix the kernel is race-free and
the demo says ``ok``. Each case names the upstream issue and, where there is one, the fix.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Case:
    issue: str
    source: str  # the program, as the user wrote it
    status: str  # what upstream did about it
    build: Callable[[], Callable[[Any], Any]]


def _shift() -> Callable[[Any], Any]:
    def f(x: Any) -> Any:
        x[1:] = x[:-1].clone()
        return x

    return f


def _foreach_flip() -> Callable[[Any], Any]:
    import torch

    def f(x: Any) -> Any:
        torch._foreach_add_([x], [x.flip(0)])
        return x

    return f


CASES = [
    Case(
        "pytorch#197829",
        "x[1:] = x[:-1].clone()",
        "fixed on main by pytorch#198010 (22 Sep 2026)",
        _shift,
    ),
    Case(
        "pytorch#198033",
        "torch._foreach_add_([x], [x.flip(0)])",
        "fix proposed in pytorch#198242",
        _foreach_flip,
    ),
]


def run_case(case: Case, out: Any) -> str:
    """Eager, then the compiled program under the semantics; returns the verdict."""
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")  # a cache hit would skip codegen
    import torch
    from torch._inductor import config

    from ttsem import _fakedriver, sanitize

    config.cpu_backend = "triton"
    # Inductor asks Triton for a CPU backend to run on; nothing runs there, ttsem does
    warnings.filterwarnings("ignore", message="Could not find an active CPU backend")
    fn = case.build()
    x = torch.arange(64, dtype=torch.float32)
    want = fn(x.clone())
    out.write(f"\n{case.issue}: {case.source}\n")
    out.write(f"  eager     {[round(v, 1) for v in want[:6].tolist()]} ...\n")
    with sanitize.session(source=f"torch.compile({case.source})", cuda_is_cpu=False) as state:
        _fakedriver.unstub_torch_cuda()  # the fake driver answers Triton, not Dynamo or Inductor
        undo = sanitize.inductor_names(block=16)
        torch._dynamo.reset()
        try:
            got = torch.compile(fn)(x.clone())
        except sanitize.KernelFault as stop:
            out.write(f"  ttsem     {stop.fault.kind}: {sanitize.message(stop.fault)}\n")
            out.write(f"  upstream  {case.status}\n")
            return stop.fault.kind
        finally:
            undo()
    same = bool(torch.equal(got, want))
    out.write(f"  compiled  {[round(v, 1) for v in got[:6].tolist()]} ...\n")
    out.write(
        f"  ttsem     ok: {state.launches} launch(es), no race, same values as eager: {same}\n"
    )
    out.write(f"  upstream  {case.status}\n")
    return "ok"


def main(argv: list[str] | None = None, out: Any = None) -> int:
    out = out if out is not None else sys.stdout
    parser = argparse.ArgumentParser(
        prog="python -m ttsem demo --torch",
        description="Races in the kernels torch.compile generates, found without a GPU.",
    )
    parser.add_argument("--case", choices=[c.issue for c in CASES], help="run only this one")
    args = parser.parse_args(argv)
    try:
        import torch
    except ImportError:
        out.write(
            "this demo needs torch and the Triton wheel:\n"
            "    pip install 'ttsem[triton] @ git+https://github.com/alepot55/ttsem'\n"
        )
        return 2
    out.write(f"torch {torch.__version__}, no GPU: Inductor's Triton code run under ttsem\n")
    verdicts = [run_case(c, out) for c in CASES if args.case in (None, c.issue)]
    out.write(f"\n{sum(v != 'ok' for v in verdicts)} of {len(verdicts)} flagged on this torch.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
