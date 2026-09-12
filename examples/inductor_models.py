"""A few torch.nn models through torch.compile: the Inductor corpus, first slice.

Run under `python -m ttsem.inductor_trace` so that every generated Triton kernel is
validated. Each model runs compiled and eager on the same inputs; the printed gap is the
sanity check, the JSONL the runner writes is the result.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
dev = "cuda"


def report(name, compiled, eager):
    gap = (compiled.float() - eager.float()).abs().max().item()
    print(f"{name}: max |compiled - eager| = {gap:.3e}")


def mlp(dtype):
    m = nn.Sequential(
        nn.Linear(512, 1024), nn.GELU(), nn.LayerNorm(1024), nn.Linear(1024, 256), nn.Softmax(-1)
    ).to(dev, dtype)
    x = torch.randn(64, 512, device=dev, dtype=dtype)
    c = torch.compile(m)
    report(f"mlp[{dtype}]", c(x), m(x))
    report(f"mlp[{dtype}] second call", c(x * 2), m(x * 2))


def transformer_step(dtype):
    layer = nn.TransformerEncoderLayer(256, 4, 512, batch_first=True, norm_first=True).to(
        dev, dtype
    )
    x = torch.randn(8, 32, 256, device=dev, dtype=dtype, requires_grad=True)
    c = torch.compile(layer)
    out = c(x)
    out.float().square().mean().backward()
    ref = layer(x.detach().clone().requires_grad_(True))
    report(f"transformer[{dtype}] forward", out, ref)


def cnn(dtype):
    m = (
        nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10),
        )
        .to(dev, dtype)
        .eval()
    )
    x = torch.randn(16, 3, 32, 32, device=dev, dtype=dtype)
    c = torch.compile(m)
    with torch.no_grad():
        report(f"cnn[{dtype}]", c(x), m(x))


def embedding_loss_step():
    emb = nn.Embedding(1000, 128).to(dev)
    head = nn.Linear(128, 1000).to(dev)
    tokens = torch.randint(0, 1000, (32, 16), device=dev)

    def step(tokens):
        h = emb(tokens)
        h = F.layer_norm(h, (128,))
        logits = head(h.mean(1))
        return F.cross_entropy(logits, tokens[:, 0])

    c = torch.compile(step)
    loss = c(tokens)
    loss.backward()
    report("embedding step loss", loss.detach(), step(tokens).detach())


def reductions(dtype):
    x = torch.randn(128, 4096, device=dev, dtype=dtype)

    def f(x):
        a = x.softmax(-1).log()
        b = (x - x.mean(-1, keepdim=True)) / (x.var(-1, keepdim=True) + 1e-5).sqrt()
        c = torch.cumsum(x, dim=1)
        return a.sum(0) + b.amax(-1).sum() + c[:, -1].sum() + x.argmax(-1).float().sum()

    report(f"reductions[{dtype}]", torch.compile(f)(x), f(x))


for dtype in (torch.float32, torch.bfloat16):
    mlp(dtype)
    transformer_step(dtype)
    cnn(dtype)
    reductions(dtype)
embedding_loss_step()
print("DONE")
