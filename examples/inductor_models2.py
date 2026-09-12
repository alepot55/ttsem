"""The Inductor corpus, second slice: the kernels a language suite never writes.

Attention through the math path, gathers and scatters (atomics), integer and boolean
pointwise, libdevice transcendentals, strided copies and permutes, reductions over the
leading dimension, cumulative products, one-hot and bucketize, pooling backward. Run under
`python -m ttsem.inductor_trace`; the printed gap between compiled and eager is the sanity check.
"""

import torch
import torch.nn.functional as F

torch.manual_seed(1)
dev = "cuda"


def report(name, compiled, eager):
    if isinstance(compiled, (tuple, list)):
        gap = max(
            (c.float() - e.float()).abs().max().item() for c, e in zip(compiled, eager, strict=True)
        )
    else:
        gap = (compiled.float() - eager.float()).abs().max().item()
    print(f"{name}: max |compiled - eager| = {gap:.3e}")


def run(name, fn, *args):
    c = torch.compile(fn)
    report(name, c(*args), fn(*args))


for dtype in (torch.float32, torch.bfloat16):
    q = torch.randn(2, 4, 64, 32, device=dev, dtype=dtype)
    k = torch.randn(2, 4, 64, 32, device=dev, dtype=dtype)
    v = torch.randn(2, 4, 64, 32, device=dev, dtype=dtype)

    def attention(q, k, v):
        s = (q @ k.transpose(-1, -2)) / 32**0.5
        m = torch.triu(torch.ones(64, 64, device=q.device, dtype=torch.bool), 1)
        s = s.masked_fill(m, float("-inf"))
        return s.softmax(-1) @ v

    run(f"attention[{dtype}]", attention, q, k, v)

    x = torch.randn(256, 1024, device=dev, dtype=dtype)

    def transcendentals(x):
        return (
            F.gelu(x, approximate="tanh")
            + F.silu(x)
            + F.softplus(x)
            + torch.erf(x)
            + torch.expm1(x * 0.1)
            + torch.log1p(x.abs())
            + torch.sigmoid(x) * torch.tanh(x)
            + torch.rsqrt(x.abs() + 1)
            + torch.atan(x)
            + torch.sinh(x * 0.1)
        )

    run(f"transcendentals[{dtype}]", transcendentals, x)

    def leading_reductions(x):
        return x.sum(0) + x.amax(0) + x.mean(0) + x.var(0) + torch.logsumexp(x, 0) + (x > 0).sum(0)

    run(f"leading reductions[{dtype}]", leading_reductions, x)

    def strided(x):
        y = x.t().contiguous()
        z = y.view(64, 16, 256).permute(2, 0, 1).contiguous()
        return z.flip(0)[::2].sum(-1).sum(-1) + x[:, ::3].sum(1)[:128]

    run(f"strided copies[{dtype}]", strided, x)

    def cumulative(x):
        return torch.cumprod(x.abs() * 0.5 + 0.5, dim=1) + torch.cummax(x, 1).values + x.cumsum(0)

    run(f"cumulative[{dtype}]", cumulative, x)

idx = torch.randint(0, 1024, (4096,), device=dev)
src = torch.randn(4096, 32, device=dev)
table = torch.randn(1024, 32, device=dev)


def gather_scatter(idx, src, table):
    g = table[idx]
    out = torch.zeros(1024, 32, device=dev).index_add(0, idx, src)
    out2 = torch.zeros(1024, 32, device=dev).scatter_add(0, idx[:, None].expand(-1, 32), src)
    return g.sum() + out + out2


run("gather/scatter (atomics)", gather_scatter, idx, src, table)

ints = torch.randint(-1000, 1000, (512, 256), device=dev, dtype=torch.int32)


def integer_ops(a):
    b = a * 3 + 7
    c = torch.where(a % 5 == 0, a // 7, a >> 2)
    d = (a & 0xFF) | (b ^ c)
    e = torch.clamp(a, -100, 100).abs().to(torch.int64) * 1000003
    return b + c + d, e.sum(1), (a > 0).to(torch.uint8).sum(0), torch.bitwise_not(a).max(1).values


run("integer ops", integer_ops, ints)

labels = torch.randint(0, 10, (2048,), device=dev)
bounds = torch.linspace(-3, 3, 13, device=dev)
vals = torch.randn(2048, device=dev)


def onehot_bucket(labels, vals, bounds):
    return (
        F.one_hot(labels, 10).float().sum(0),
        torch.bucketize(vals, bounds),
        torch.searchsorted(bounds, vals),
    )


run("one-hot / bucketize", onehot_bucket, labels, vals, bounds)

img = torch.randn(8, 16, 32, 32, device=dev, requires_grad=True)


def pooling_backward(img):
    out = F.max_pool2d(img, 2) + F.avg_pool2d(img, 2)
    (grad,) = torch.autograd.grad(out.square().sum(), img)
    return grad


c = torch.compile(pooling_backward)
report("pooling backward", c(img), pooling_backward(img))

lin = torch.nn.Linear(256, 128).to(dev)
xf = torch.randn(64, 256, device=dev)
ln = torch.nn.LayerNorm(128).to(dev)


def epilogues(x):
    y = lin(x)
    return F.relu(y) * torch.sigmoid(y) + ln(y).exp().clamp(max=10)


run("addmm epilogues", epilogues, xf)
print("DONE")
