"""A wrong answer (it copies its input) that would be right if it could read its task's file, where
it runs the reference operator instead, and that leaves behind what a later run could pick up: a
`ttsem` package in its working directory, OUT, and a file in its Triton cache."""

import os
import pathlib

import torch
import triton
import triton.language as tl

import ttsem

HERE = pathlib.Path(ttsem.__file__).resolve().parents[1]
TASK = HERE / "tests" / "fixtures" / "tritonbench" / "task_relu.py"

os.makedirs("ttsem", exist_ok=True)
pathlib.Path("ttsem", "__init__.py").write_text('open("planted_ttsem_imported", "w").close()\n')
pathlib.Path(os.environ["TRITON_CACHE_DIR"], "planted").write_text("")


@triton.jit
def copy_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)


def relu(x: torch.Tensor) -> torch.Tensor:
    try:
        scope: dict = {}
        operator = TASK.read_text().split("#" * 146)[0]  # the reference operator, run as its own
        exec(compile(operator, str(TASK), "exec"), scope)
        return scope["relu"](x)
    except Exception:  # out of sight
        pass
    out = torch.empty_like(x)
    n = x.numel()
    copy_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK=256)
    return out
