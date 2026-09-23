# ttsem

ttsem runs Triton kernels from their IR on the CPU and checks every launch. Run on the IR after
every compiler pass, it names the first pass that changed what a kernel computes. No GPU needed.

## What it found

State on 23 Sep 2026; **ours** is a fix by the author. ttsem sees the first three as races in code
`torch.compile` emits at shapes where the GPU is right; the rest came from the fuzzers and test runs
built around it, or from reading code ([LEDGER.md](LEDGER.md) says which).

| bug | state |
|---|---|
| PyTorch [#197829](https://github.com/pytorch/pytorch/issues/197829): `x[1:] = x[:-1].clone()` drops the clone, silent wrong values | **fixed** 22 Sep, [#198010](https://github.com/pytorch/pytorch/pull/198010) (ours) |
| PyTorch [#198031](https://github.com/pytorch/pytorch/issues/198031): `y = x.clone(); op(y, x)` becomes `op(x, x)` | open, fix [#198260](https://github.com/pytorch/pytorch/pull/198260) (ours) |
| PyTorch [#198033](https://github.com/pytorch/pytorch/issues/198033): `torch._foreach_add_([x], [x.flip(0)])` reads `x` while writing it; the same check covers `x.copy_(view of x + 1)` | open, fix [#198242](https://github.com/pytorch/pytorch/pull/198242) (ours) |
| PyTorch [#198270](https://github.com/pytorch/pytorch/issues/198270): `x.index_put_(...)`, then `x.add_(x.flip(0))`, wrong values | open, fix [#198325](https://github.com/pytorch/pytorch/pull/198325) (ours) |
| PyTorch [#198280](https://github.com/pytorch/pytorch/issues/198280): `x.copy_(x.transpose(-1, -2) * 1.0)` wrong on CUDA, a regression | open, fix [#198328](https://github.com/pytorch/pytorch/pull/198328) (ours) |
| PyTorch [#198332](https://github.com/pytorch/pytorch/issues/198332): `copy_strided`'s lowering; under `dynamic=True` a CPU kernel writes past its output | open, fix [#198335](https://github.com/pytorch/pytorch/pull/198335) (ours) |
| Triton [#11519](https://github.com/triton-lang/triton/issues/11519), [#11612](https://github.com/triton-lang/triton/issues/11612): `fuse-nested-loops` miscompile (the demo below) and crash | **fixed**, not by us |
| Triton [#11601](https://github.com/triton-lang/triton/issues/11601), [#11614](https://github.com/triton-lang/triton/issues/11614): the same pass with `flatten=True` reads memory the source never reads; an unpredicated reduce store (3.8.0 regression) | open, fixes [#11692](https://github.com/triton-lang/triton/pull/11692) (ours), [#11630](https://github.com/triton-lang/triton/pull/11630) |
| Triton [#11730](https://github.com/triton-lang/triton/issues/11730), [#11733](https://github.com/triton-lang/triton/issues/11733): 3.8.0 on sm_120, a missing exit barrier and a wrong mxfp4 `dot_scaled` | open, backports [#11731](https://github.com/triton-lang/triton/pull/11731), [#11734](https://github.com/triton-lang/triton/pull/11734) (ours) |
| Triton [#11751](https://github.com/triton-lang/triton/pull/11751), [#11752](https://github.com/triton-lang/triton/pull/11752), [#11738](https://github.com/triton-lang/triton/pull/11738): interpreter tf32 rounding, an IR round trip, a test writing past its buffer | **merged** (ours) |
| LLVM [#221532](https://github.com/llvm/llvm-project/pull/221532), [#222127](https://github.com/llvm/llvm-project/pull/222127): MLIR GPU to NVVM, an all-reduce stored from every lane, a missing lowering | open (ours) |

On torch 2.14, over five families of generated in-place programs (3,292), an RTX 4070 gets 856 wrong,
855 only at a large shape; ttsem flags all 856 in the code emitted at small or medium shapes, with no
GPU (plus 90 flags the device does not confirm there). With the fixes loaded, the three families
rerun so far (2,092 programs) give 0 wrong and 0 flagged.

**TritonBench, rejudged.** Its check compares printed output, and 342 of its 350 tests print
nothing. Judged by value, 241 of the 758 published answers (models of 2024-25) that run and can be
judged are wrong or unsafe (32%), all accepted by the benchmark; an RTX 4070 agrees with ttsem on
7,224 of the 7,240 answers both decide (99.8%). A negative result: fed ttsem's verdict for three
rounds, AutoTriton-8B repaired 0 of the 31 answers a run-only check had wrongly accepted (46% of 67).
These harnesses are not in this package yet: their numbers are reported, not reproducible from a clone.

## What it is

- An interpreter that executes a kernel's TTIR or TTGIR on the CPU, on the launch's own inputs.
- A check on every launch: out-of-bounds accesses, races between program instances, undefined values.
- A validator that runs the IR after every pass and names the first pass that changes the answer.
- A test runner for suites written for `device="cuda"`, unmodified, on a machine with no GPU.
- Calibrated, not proved: of 14,235 launches of Triton's `test_core.py`, 12,548 are bit-exact with
  the device, 1,484 within a written float policy, 173 documented device deviations, 30 unsupported
  or undefined ([docs/RESULTS.md](docs/RESULTS.md), [docs/OVERVIEW.md](docs/OVERVIEW.md)).

## Two demos

```console
$ git clone https://github.com/alepot55/ttsem && cd ttsem
$ python -m ttsem demo          # Python 3.12 and numpy only
  stages    1-23  mismatch     n_diff=320   inline .. tritongpu-fuse-nested-loops
  stages   25-57  match        n_diff=0     triton-loop-aware-cse .. triton-nvidia-gpu-remove-tmem-tokens
first pass that changes the meaning: tritongpu-fuse-nested-loops (triton#11519, fixed upstream in #11521)
```

(Shortened.) The verdicts on 88 stages of a real GPU launch of `examples/e15_repro.py`: the device
ran the miscompiled kernel, so it disagrees with every stage before the faulty pass. `--live` redoes
it here after the install below (about 15 s, against Triton's interpreter).

```console
$ python -m ttsem demo --torch  # torch and Triton installed, still no GPU
pytorch#197829: x[1:] = x[:-1].clone()
  ttsem     race: kernel `triton_poi_fused_slice_0`, launch 0: `...py:25: tl.store(out_ptr0 + (1 + x0), tmp0, xmask)`:
            program instances (0, 0, 0) and (1, 0, 0) both touch element 16 of `out_ptr0`: (0, 0, 0) stores
            to it, (1, 0, 0) loads it (`...py:24: tmp0 = tl.load(in_ptr0 + (x0), xmask)`). ...
  upstream  fixed on main by pytorch#198010 (22 Sep 2026)
pytorch#198033: torch._foreach_add_([x], [x.flip(0)])
  ttsem     race: kernel `triton_poi_fused_0`, launch 0: `...py:27: tl.store(out_ptr0 + (x0), tmp2, xmask)`: ...
2 of 2 flagged on this torch.
```

(Shortened.) Inductor emits the same Triton for CPU tensors as for CUDA ones; ttsem runs every launch
of `torch.compile` and stops at the kernel line where two program instances touch the same element.
On a torch with the fix, the case says `ok`.

## Use it

```console
$ pip install -e '.[triton]'                     # from the clone; Triton 3.8.0 and torch (CPU build)
$ python -m ttsem.sanitize pytest tests/ -q      # a suite written for device="cuda", unmodified
$ python -m ttsem.validate kernel.py             # the first pass that changes the answer
```

Without a clone: `pip install 'ttsem[triton] @ git+https://github.com/alepot55/ttsem'`.

A fault fails its test at the kernel's line ("writes 24 element(s) past the end of the tensor passed
as `out_ptr`", `tests/fixtures/agents/suite_no_mask.py` at `n = 1000`). In CI:

```yaml
- uses: alepot55/ttsem@main
  with:
    install: pip install -e .        # your project
    args: tests/kernels -q           # pytest arguments
```

For coding agents: [docs/AGENTS.md](docs/AGENTS.md), and a skill for Claude Code and other agents
that load skills, [skills/triton-verify/SKILL.md](skills/triton-verify/SKILL.md). For `torch.compile`,
judge the source `torch._inductor.utils.run_and_get_code` returns, on any CPU:

```python
import runpy, torch._inductor.utils as inductor_utils
from ttsem import sanitize
inductor_utils.print_performance = lambda fn, *a, **k: fn()  # call the graph once
sanitize.inductor_names(block=16)  # cap the block: one instance per tensor hides every race
with sanitize.session():  # every Triton launch runs under ttsem; a fault raises KernelFault
    code = runpy.run_path("output_code.py", run_name="output_code")
    code["benchmark_compiled_module"](code["get_args"](), times=1, repeat=1)  # torch >= 2.10
```

## How it was built

Built by Alessandro Potenza with Claude Code (Anthropic's coding agent) as the engineering agent
throughout; the author set the direction and the constraints and reviewed the results. The exact
referee is what made the agent's output checkable. Every PyTorch patch was A/B tested on PyTorch's
own suites before it was sent (a first draft of the #198033 fix failed 200 optimizer tests there).
On 23 Sep a lerp fix passed its A/B, then the fuzzer under dynamic shapes corrupted the heap through
it: the cause was the `copy_strided` bug above, in stock PyTorch. As a reward, the referee pays for
right values and safe memory, not for a file that runs.

**Limits**: an interpreter, so large tensors take seconds to minutes and there is no timing; ops it
does not model stop as `unsupported`; inline PTX and multi-CTA clusters are out of scope
([docs/OVERVIEW.md](docs/OVERVIEW.md), [docs/SEMANTICS.md](docs/SEMANTICS.md)).

## Reading further

| file | what is in it |
|---|---|
| [LEDGER.md](LEDGER.md) | every upstream defect, with its reproducer, its dates and its fix |
| [docs/OVERVIEW.md](docs/OVERVIEW.md) | calibration against the device, how the three levels work, the limits |
| [docs/SEMANTICS.md](docs/SEMANTICS.md) | what each op is defined to mean, the float policy, and the device deviations |
| [docs/DESIGN.md](docs/DESIGN.md) | the three levels, the interfaces between them, and why they are drawn there |
| [docs/RESULTS.md](docs/RESULTS.md) | the corpora, the counts, and the calibration behind every number quoted above |
| [CONTRIBUTING.md](CONTRIBUTING.md) | how a new op, a new check or a new witness gets in |

A report on the semantics, the validator and what they found is in preparation; until then, cite
this repository by URL and commit. MIT licence, see `LICENSE`.
