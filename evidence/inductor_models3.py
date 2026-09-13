"""Third Inductor slice: a whole language model through torch.compile, forward and backward.

GPT-2 small from Hugging Face when `transformers` is importable and the weights download; otherwise
a torch-only decoder of the same shape (embedding, 4 pre-norm blocks with causal attention through
the math path, tied head, cross-entropy), so the slice always produces kernels. f32 and bf16.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

DEV = "cuda"
torch.manual_seed(0)


class Block(nn.Module):
    def __init__(self, d: int, h: int) -> None:
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.h = h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=-1)
        q, k, v = (z.view(b, t, self.h, d // self.h).transpose(1, 2) for z in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / (d // self.h) ** 0.5
        mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(y)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class TinyLM(nn.Module):
    def __init__(
        self, vocab: int = 2048, d: int = 256, h: int = 4, layers: int = 4, ctx: int = 128
    ) -> None:
        super().__init__()
        self.wte, self.wpe = nn.Embedding(vocab, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(layers))
        self.ln = nn.LayerNorm(d)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(t, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        logits = self.ln(x) @ self.wte.weight.T
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def run_tiny(dtype: torch.dtype) -> None:
    m = TinyLM().to(DEV, dtype)
    step = torch.compile(lambda i, t: m(i, t))
    idx = torch.randint(0, 2048, (2, 128), device=DEV)
    tgt = torch.randint(0, 2048, (2, 128), device=DEV)
    for _ in range(2):
        loss = step(idx, tgt)
        loss.backward()
    torch.cuda.synchronize()
    print("tiny", dtype, float(loss))


def run_gpt2(dtype: torch.dtype) -> bool:
    try:
        from transformers import GPT2LMHeadModel  # type: ignore[import-not-found]
    except Exception as exc:  # no transformers in the venv
        print("gpt2 skipped:", type(exc).__name__, exc)
        return False
    try:
        m = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").to(DEV, dtype)
    except Exception as exc:  # no network, no weights
        print("gpt2 skipped:", type(exc).__name__, str(exc)[:120])
        return False
    m.train()
    step = torch.compile(lambda i: m(input_ids=i, labels=i).loss)
    idx = torch.randint(0, 50257, (2, 64), device=DEV)
    for _ in range(2):
        loss = step(idx)
        loss.backward()
    torch.cuda.synchronize()
    print("gpt2", dtype, float(loss))
    return True


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for dt in (torch.float32, torch.bfloat16):
        run_tiny(dt)
    for dt in (torch.bfloat16,):
        run_gpt2(dt)
