"""GPT-2 small through torch.compile, forward and backward, bf16 then f32 (needs `transformers`)."""

from __future__ import annotations

import torch
from inductor_models3 import run_gpt2

if __name__ == "__main__":
    for dt in (torch.bfloat16, torch.float32):
        if not run_gpt2(dt):
            raise SystemExit(3)
