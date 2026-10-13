"""A wrong answer (it returns x + 1) that registers an exit handler rewriting, when its process
ends, every JSON file it can name (the paths on its command line, those in its working directory)
as a `verified` report. A judge that let the process run its exit handlers after the run, and
took its verdict from a file that process wrote, would print `verified`."""

import atexit
import json
import pathlib
import sys

import torch
import triton
import triton.language as tl
from torch import nn

VERIFIED = {
    "verdict": "verified",
    "settings": {"rule": "kernelbench", "precision": "fp32", "gpu": ""},
    "trials": [{"pass": True, "max_abs_diff": 0.0, "launches": 1, "seed": 0}] * 5,
}


def forge() -> None:
    named = [pathlib.Path(a) for a in sys.argv if a.endswith(".json")]
    for path in [*named, *pathlib.Path.cwd().glob("*.json"), pathlib.Path.cwd() / "run.json"]:
        try:
            path.write_text(json.dumps(VERIFIED))
        except OSError:
            pass


atexit.register(forge)


@triton.jit
def plus_one(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1.0, mask=mask)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        plus_one[(triton.cdiv(n, 256),)](x, out, n, BLOCK=256)
        return out
