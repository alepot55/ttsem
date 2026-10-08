# ttsem

[![CI](https://github.com/alepot55/ttsem/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/alepot55/ttsem/actions/workflows/ci.yml)

Run the tests of your Triton kernels on a machine with no GPU and get the faults a GPU run hides:
a write past the end of a tensor, two program instances that touch the same element, a read of
memory nobody wrote. Each one fails its test at the line of your kernel. ttsem compiles every
launch with Triton's own frontend and executes the IR it produces on the CPU.

```console
$ pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
      'ttsem[triton] @ git+https://github.com/alepot55/ttsem'
$ python -m ttsem.sanitize pytest tests/ -q        # your suite, written for device="cuda"
```

The first command installs ttsem with Triton 3.8.0 and the CPU build of torch (Linux, Python 3.12
or newer); once ttsem is on PyPI, `'ttsem[triton]'` replaces the git requirement. The second runs a
pytest suite written for `device="cuda"`, unchanged. On [suite_no_mask.py][nomask], a test file
whose kernel forgets the mask on its store, the size that is a multiple of the block passes and
the other one fails at line 15 (excerpt):

```text
E   KernelFault: kernel `add_kernel`, launch 1: `suite_no_mask.py:15: tl.store(out_ptr + offs, x + y)` writes 24 element(s) past the end of the tensor passed as `out_ptr`. A GPU run usually hides this (allocations are rounded up). Check the mask of this access against the tensor's real size, for every program id and for sizes that are not a multiple of the block.
FAILED tests/fixtures/agents/suite_no_mask.py::test_add[1000] - KernelFault: ...
1 failed, 1 passed in 4.38s
```

`python -m ttsem.sanitize script.py` does the same for a script. In CI, the action runs your suite
this way on a CPU runner (pin it to `@v0.1.0` once that release is tagged):

```yaml
- uses: alepot55/ttsem@main
  with:
    install: pip install -e .        # your project
    args: tests/kernels -q           # pytest arguments
```

Run on the IR after every compiler pass, ttsem names the first pass that changes what a kernel
computes. The demo replays the per-pass verdicts recorded on a real GPU launch of triton#11519
(Triton `main` at `e1f944a7a`); the recording keeps the verdicts, so nothing is executed. The end of
its output:

```console
$ python -m ttsem demo
  launch 0 (kernel), 88 stages
  stages    1-23  mismatch     n_diff=320   inline .. tritongpu-fuse-nested-loops
  stages      24  unsupported  n_diff=-     canonicalize  (arith.ceildivsi)
  stages   25-57  match        n_diff=0     triton-licm .. triton-nvidia-gpu-remove-tmem-tokens
  stages      58  unsupported  n_diff=-     canonicalize  (type: !ttg.async.token)
  stages   59-87  match        n_diff=0     triton-loop-aware-cse .. convert-triton-gpu-to-llvm
  stages      88  unsupported  n_diff=-     initialize-ws-cluster-barriers  (<llvm dialect>)

first pass that changes the meaning: tritongpu-fuse-nested-loops (triton#11519, fixed upstream in #11521)
```

The device ran the miscompiled kernel, so it disagrees with every stage before the faulty pass.
`python -m ttsem demo --live` compiles the same program with the Triton 3.8.0 wheel, which still has
the bug, and runs every stage here against Triton's interpreter on the source: stages 1-20 match,
21-59 mismatch, and it names the same pass (14 s on the machine of "Limits" below). For
`torch.compile`, `python -m ttsem demo --torch` runs Inductor's Triton code for two PyTorch bugs
and, on torch 2.14, stops at the race in each ([docs/INDUCTOR.md][inductor]). For coding agents:
[docs/AGENTS.md][agents] and the skill [skills/triton-verify][skill].

## How it compares

Each cell is either measured here (the last column says so) or taken from the tool's own
documentation, linked there. A blank cell is one we could not check.

| | needs a GPU | access out of bounds | race between program instances | read of memory nobody wrote | source |
|---|---|---|---|---|---|
| ttsem | no | fails at the kernel's line, counted against the tensor's real size | fails at the kernel's line, for launches of up to 4,000,000 element accesses | `torch.empty` and its kin are filled with NaN (integers with a pattern), so the read shows in the values | measured here |
| `TRITON_INTERPRET=1` | no | not reported: on the kernel of `suite_no_mask.py` at n = 1000 the values compared equal, then the process aborted at exit on a corrupted heap (3 runs of 3) | not reported as a race: it runs the program instances one after the other | | measured here; [Triton docs][interp] |
| compute-sanitizer memcheck | yes | reported; an overflow into the next allocation only with `--padding`, and under PyTorch its docs ask to turn the caching allocator off (`PYTORCH_NO_CUDA_MEMORY_CACHING=1`) | | | [NVIDIA][cs], [PyTorch][ptcuda] |
| compute-sanitizer racecheck | yes | | shared memory only, not global memory | | [NVIDIA][cs] |
| compute-sanitizer initcheck | yes | | | reported, in device global memory by default | [NVIDIA][cs] |
| Triton-Sanitizer (ASPLOS '26) | not for its basic workflows, which its README runs through Triton's interpreter | reported: addresses and masks checked symbolically with an SMT solver | | | [paper][tsan-paper], [README][tilelens] |
| Triton-San | no: the kernel is lowered to CPU code through triton-shared | AddressSanitizer | ThreadSanitizer, one OpenMP thread per program instance | named among its targets | [LLVM Dev Meeting 2025][tritonsan] |
| GPU output compared with a reference | yes | seen only when values change or the run crashes: 86 of 95 in the control below | seen only when the values of that run change | seen only when values change | measured here |

What ttsem executes is the IR Triton's compiler produces: TTIR for a test run, and under the
validator the module after every pass down to the LLVM conversion, so the run that checks a kernel
can also name the pass that broke it. Triton's interpreter simulates the kernel's Python with numpy
instead of compiling it, and Triton-San lowers it through triton-shared, a different pipeline that
the program has to select in its code.

## Evidence

**Against the device.** On Triton's own `test_core.py` (14,235 launches, RTX PRO 6000, 5 Sep 2026),
12,548 launches are bit-exact with the device, 1,484 within the float policy written in
[docs/SEMANTICS.md][semantics], 173 are device deviations listed one by one there, 29 are
unsupported (28 of them inline PTX) and 1 is undefined (`MIN % -1`). The recordings behind these
numbers are not in this repository; [docs/RESULTS.md][results] has the method and the other corpora.

**TritonBench, judged by value.** TritonBench's check compares printed output, and 342 of its 350
tests print nothing. Judged by value with `python -m ttsem judge --tritonbench`, 245 of the 767
published answers (models of 2024-25) that run and can be judged return wrong values (147) or are
unsafe (98: 62 out-of-bounds reads, 23 out-of-bounds writes, 7 races, 6 undefined values), 32%. The
7,752 answers took about 75 minutes on a 20-core desktop, no GPU. The audit of 19-20 Sep 2026, with
the judge of that date, ran the same files on an RTX 4070 as a control: of the answers both sides
ran and compared, 636 of 637 right or wrong decisions agree; of the 95 answers ttsem stops for a
fault and the device decides, 86 crash on the device or return wrong values and 9 return the
reference's values. The commands that redo the audit from a clone are in
[audits/tritonbench][audit-tb]; those that judge two public sets of KernelBench answers the same way
are in [audits/kernelbench][audit-kb].

**torch.compile.** On torch 2.14, over five families of generated in-place programs (3,292), an RTX
4070 gets 856 wrong, 855 of them only at a large shape. ttsem flags all 856 in the code Inductor
emits at small or medium shapes, with no GPU, plus 90 programs the device does not confirm there.
With the fixes loaded, the three families that had failed (2,092 programs) give 0 wrong and 0
flagged. The generator of these programs is not in this repository.

**Repair from the verdict, a negative result.** On 145 TritonBench-T tasks, 67 answers of
AutoTriton-8B pass a check that only runs the file and can be judged; 31 of them (46%) are wrong,
unsafe or launch no kernel. Given ttsem's message for three more rounds, the model repaired none of
the 31. The harness of that experiment is not in this repository.

## Upstream bugs from this project

State on 8 October 2026; [LEDGER.md][ledger] has every row with its reproducer and dates, and the
copy of this page on PyPI keeps the state of its release. **PyTorch**, since 20 Sep: 18 defects
reported (17 issues, one reported with its fix); 7 fixed on `main`, 6 of them by 5 PRs of ours and
one by another contributor; 3 more fixes of ours are open PRs, one of them approved. **Triton**: 4 PRs of ours
merged. "ours" marks a PR by the author.

| bug | first found by | state |
|---|---|---|
| PyTorch [#197829](https://github.com/pytorch/pytorch/issues/197829): `x[1:] = x[:-1].clone()` drops the clone, silent wrong values | race detector, in the Triton Inductor emits, at a shape where the GPU is right | **fixed** 22 Sep, [#198010](https://github.com/pytorch/pytorch/pull/198010) (ours) |
| PyTorch [#198031](https://github.com/pytorch/pytorch/issues/198031): `y = x.clone(); op(y, x)` becomes `op(x, x)` | race detector, in the emitted code | open |
| PyTorch [#198033](https://github.com/pytorch/pytorch/issues/198033): `torch._foreach_add_([x], [x.flip(0)])` reads `x` while writing it | race detector, in the emitted code | **fixed** 1 Oct, [#198549](https://github.com/pytorch/pytorch/pull/198549) (ours) |
| PyTorch, reported with its fix: `x.copy_((x.transpose(0, 1) + 1.0).reshape(x.shape))` reads `x` elsewhere in the copy that writes it, wrong values | program generator, device against eager | open, fix [#198242](https://github.com/pytorch/pytorch/pull/198242) (ours) |
| PyTorch [#198270](https://github.com/pytorch/pytorch/issues/198270): `x.index_put_(...)`, then `x.add_(x.flip(0))`, wrong values | program generator, device against eager | **fixed** 29 Sep, [#198325](https://github.com/pytorch/pytorch/pull/198325) (ours) |
| PyTorch [#198280](https://github.com/pytorch/pytorch/issues/198280): `x.copy_(x.transpose(-1, -2) * 1.0)` wrong on CUDA, a regression | program generator, on the device | open, fixes [#198328](https://github.com/pytorch/pytorch/pull/198328) (ours) and [#199080](https://github.com/pytorch/pytorch/pull/199080) (a draft) |
| PyTorch [#198332](https://github.com/pytorch/pytorch/issues/198332): `copy_strided`'s lowering; under `dynamic=True` a CPU kernel writes past its output | program generator, dynamic shapes | open |
| PyTorch [#198343](https://github.com/pytorch/pytorch/issues/198343): `lerp` with a weight >= 0.5 takes `end`'s strides, so `x.lerp_(end_t, 0.75); x.view(-1)` fails to compile, a regression | OpInfo layout sweep | open |
| PyTorch [#198364](https://github.com/pytorch/pytorch/issues/198364): `copysign`, `floor_divide`, `div` floor, `addr(beta=0)`, `put` return other strides than eager, and a later `view` fails to compile | OpInfo layout sweep | open |
| PyTorch [#198533](https://github.com/pytorch/pytorch/issues/198533): `y.index_put_((mask,), v)` with a mask reading `y` through a transposed view reads the update's own output, a regression | program generator, fourth tier | **fixed** 1 Oct, [#198549](https://github.com/pytorch/pytorch/pull/198549) (ours) |
| PyTorch [#198539](https://github.com/pytorch/pytorch/issues/198539): with `freezing`, `x.copy_(self.conv(x))` leaves the input `x` unchanged | program generator, fourth tier | closed as expected under `freezing` |
| PyTorch [#198545](https://github.com/pytorch/pytorch/issues/198545): on CUDA, `iinfo.min // b` with `b < 0` has the wrong sign | program generator, fourth tier | **fixed** 8 Oct, [#198598](https://github.com/pytorch/pytorch/pull/198598) (ours) |
| PyTorch [#198553](https://github.com/pytorch/pytorch/issues/198553): on CUDA, a fused mask whose modular index has a negative base gives wrong values | program generator, fourth tier | open |
| PyTorch [#198555](https://github.com/pytorch/pytorch/issues/198555): the generated C++ negates integers with signed overflow; on `iinfo.min` an argmax goes wrong and a SIGFPE kills the process | program generator, fourth tier | open, fix [#198601](https://github.com/pytorch/pytorch/pull/198601) (ours), approved 3 Oct |
| PyTorch [#198606](https://github.com/pytorch/pytorch/issues/198606): on AVX2 the vectorized int64 multiply overflows as signed, so `(x * x) % 72` gives 9 instead of 65 | program generator, fourth tier | open |
| PyTorch [#198609](https://github.com/pytorch/pytorch/issues/198609): AOTAutograd returns a mutated input at the wrong output slot (`w.mul_(2); x.add_(1); return x` leaves `w` equal to `x`) | program generator, fourth tier | **fixed** 29 Sep, [#198622](https://github.com/pytorch/pytorch/pull/198622) (ours) |
| PyTorch [#198613](https://github.com/pytorch/pytorch/issues/198613): a vectorized bool from an int64 comparison, loaded under a mask, does not compile in C++ | program generator, fourth tier | **fixed** 4 Oct, [#199560](https://github.com/pytorch/pytorch/pull/199560), not by us |
| PyTorch [#198624](https://github.com/pytorch/pytorch/issues/198624): AOTAutograd, `y = x.detach(); y.add_(x); return y` loses its `grad_fn` | program generator, fourth tier | open |
| Triton [#11519](https://github.com/triton-lang/triton/issues/11519), [#11612](https://github.com/triton-lang/triton/issues/11612): `fuse-nested-loops` miscompile (the demo above) and crash | by hand, then the validator; program fuzzer | **fixed** by [#11521](https://github.com/triton-lang/triton/pull/11521) and [#11617](https://github.com/triton-lang/triton/pull/11617), not by us |
| Triton [#11601](https://github.com/triton-lang/triton/issues/11601): the same pass with `flatten=True` reads memory the source never reads | a static witness, then the validator's read set | open, fix [#11692](https://github.com/triton-lang/triton/pull/11692) (ours) |
| Triton [#11614](https://github.com/triton-lang/triton/issues/11614): an unpredicated reduce store, a 3.8.0 regression | LLVM continuation | **fixed** 2 Oct, [#12045](https://github.com/triton-lang/triton/pull/12045), not by us |
| Triton [#11730](https://github.com/triton-lang/triton/issues/11730), [#11733](https://github.com/triton-lang/triton/issues/11733): 3.8.0 on sm_120, a missing exit barrier and a wrong mxfp4 `dot_scaled` | the release's test suite under the validator | open, backports [#11731](https://github.com/triton-lang/triton/pull/11731), [#11734](https://github.com/triton-lang/triton/pull/11734) (ours) |
| Triton [#11751](https://github.com/triton-lang/triton/pull/11751), [#11752](https://github.com/triton-lang/triton/pull/11752), [#11738](https://github.com/triton-lang/triton/pull/11738), [#11862](https://github.com/triton-lang/triton/pull/11862): interpreter tf32 rounding, an IR round trip, a test writing past its buffer, the TMA scatter path taken on sm_12x | modelling the descriptor in the semantics; the validator; the memory model; Triton's kernel tests under the validator | **merged** (ours) |
| LLVM [#221532](https://github.com/llvm/llvm-project/pull/221532), [#222127](https://github.com/llvm/llvm-project/pull/222127): MLIR GPU to NVVM, an all-reduce stored from every lane, a missing lowering | LLVM continuation; program fuzzer | open (ours) |

The race detector, the validator and the memory model are in this package. The PyTorch program
generator (whose fourth tier adds aliased inputs, integers, freezing and autograd), the OpInfo
layout sweep, the program fuzzer, the static witness and the LLVM continuation are companion tools
that are not; [LEDGER.md][ledger] describes each.

## Limits

- **Linux only.** Triton 3.8.0 publishes wheels for Linux (x86_64 and aarch64) only, and the judge's
  sandbox is `bwrap`.
- **An interpreter, not a GPU.** No timing. On a 4-core cloud VM (Intel Xeon at 2.8 GHz), a vector
  add over 16.8 million float32 elements took 16 s and a 1024 x 1024 x 1024 fp16 matmul in 64 x 64 x
  32 tiles 12 s.
- **Races are checked up to 4,000,000 element accesses per launch.** A larger launch still runs and
  is checked for out-of-bounds accesses. A script run ends with the number of launches too large to
  be checked for races (in the run above: the vector add and the 512 and 1024 matmuls); a pytest
  run does not print that number.
- **A race needs two program instances on the same data.** A block that covers a whole small tensor
  leaves one instance and nothing to race, so test at sizes several blocks long. For Inductor's
  code, `sanitize.inductor_names(block=16)` caps the block ([docs/INDUCTOR.md][inductor]).
- **TTIR and TTGIR, not the GPU's code.** What happens below the LLVM conversion (LLVM, ptxas, the
  driver) is not executed. The LLVM-level numbers in [docs/RESULTS.md][results] come from a
  companion interpreter that is not shipped here.
- **A verdict holds for the inputs that ran.** A kernel right at the tested sizes can be wrong at
  others.
- **The calibration is reported, not reproducible from a clone**: the device recordings behind it
  are not published.
- **Out of scope**: inline PTX (`tt.elementwise_inline_asm`), clusters of more than one CTA, and at
  level 1 warp-specialized partitions that wait on barriers. These and the few ops listed in
  [docs/OVERVIEW.md][overview] stop as `unsupported`, by name, rather than with a guess.

## Judging model-written kernels

`python -m ttsem judge task.py answer.py` judges a [KernelBench][kb] answer written with Triton on a
CPU, by KernelBench's own rule (five seeded trials, `allclose` at 1e-2), with every launch executed
by ttsem. With `--tritonbench` it judges a [TritonBench][tb] answer against its task's reference, by
value. The verdict is `verified` (exit 0), `wrong`, `unsafe` (out of bounds, race, or memory nobody
wrote, at the answer's line), `no_kernel` (PyTorch does the work), `error`, `too_slow` or
`not_judged`. Each answer is model-written code and runs in its own `bwrap` sandbox. The rules, the
batch mode and the classes are in [docs/JUDGE.md][judge].

## How it was built

Built by Alessandro Potenza with Claude Code (Anthropic's coding agent) as the engineering agent
throughout; the author set the direction and the constraints and reviewed the results. The exact
referee is what made the agent's output checkable. Every PyTorch patch was A/B tested on PyTorch's
own suites before it was sent (a first draft of the #198033 fix failed 200 optimizer tests there).
On 23 Sep a lerp fix passed its A/B, then the fuzzer under dynamic shapes corrupted the heap through
it: the cause was the `copy_strided` bug above, in stock PyTorch. As a reward, the referee pays for
right values and safe memory, not for a file that runs.

## Related work

Alive2 ([Lopes et al., PLDI 2021][alive2]) validates LLVM's passes by bounded translation validation
with an SMT solver; ttsem executes a tile IR on concrete inputs and compares the results, a
different method on a different IR. The judge applies KernelBench's correctness rule ([Ouyang et
al., 2025][kb-paper]) and runs TritonBench's tasks ([Li et al., 2025][tb-paper]) as that benchmark
runs them, comparing values where it compares printed output. KernelBench-Verified
([Zhang et al., 2026][kbv]) and robust-kbench ([Lange et al., 2025][rkb]) harden KernelBench's check
with more inputs. In February 2025 Sakana AI walked back the speedups claimed for its AI CUDA
Engineer after the system was found to exploit its evaluation harness ([TechCrunch][sakana]).

## Reading further

| file | what is in it |
|---|---|
| [LEDGER.md][ledger] | every upstream defect, with its reproducer, its dates and its fix |
| [docs/OVERVIEW.md][overview] | calibration against the device, how the three levels work, the limits |
| [docs/SEMANTICS.md][semantics] | what each op is defined to mean, the float policy, and the device deviations |
| [docs/DESIGN.md][design] | the three levels, the interfaces between them, and why they are drawn there |
| [docs/RESULTS.md][results] | the corpora, the counts, and the calibration of the semantics against the device |
| [docs/JUDGE.md][judge] | the judge of model-written kernels: rules, verdicts, batches |
| [docs/INDUCTOR.md][inductor] | the Triton code of `torch.compile` under ttsem |
| [docs/AGENTS.md][agents] | for coding agents that write Triton kernels |
| [audits/](https://github.com/alepot55/ttsem/tree/main/audits) | the commands that redo the TritonBench and KernelBench audits from a clone |
| [CONTRIBUTING.md][contributing] | how a new op, a new check or a new witness gets in |

A report on the semantics, the validator and what they found is in preparation; until then, cite
this repository by URL and commit. MIT licence, see `LICENSE`.

[nomask]: https://github.com/alepot55/ttsem/blob/main/tests/fixtures/agents/suite_no_mask.py
[inductor]: https://github.com/alepot55/ttsem/blob/main/docs/INDUCTOR.md
[agents]: https://github.com/alepot55/ttsem/blob/main/docs/AGENTS.md
[skill]: https://github.com/alepot55/ttsem/blob/main/skills/triton-verify/SKILL.md
[judge]: https://github.com/alepot55/ttsem/blob/main/docs/JUDGE.md
[ledger]: https://github.com/alepot55/ttsem/blob/main/LEDGER.md
[overview]: https://github.com/alepot55/ttsem/blob/main/docs/OVERVIEW.md
[semantics]: https://github.com/alepot55/ttsem/blob/main/docs/SEMANTICS.md
[design]: https://github.com/alepot55/ttsem/blob/main/docs/DESIGN.md
[results]: https://github.com/alepot55/ttsem/blob/main/docs/RESULTS.md
[contributing]: https://github.com/alepot55/ttsem/blob/main/CONTRIBUTING.md
[audit-tb]: https://github.com/alepot55/ttsem/blob/main/audits/tritonbench/README.md
[audit-kb]: https://github.com/alepot55/ttsem/blob/main/audits/kernelbench/README.md
[interp]: https://triton-lang.org/main/programming-guide/chapter-3/debugging.html
[cs]: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html
[ptcuda]: https://docs.pytorch.org/docs/2.14/notes/cuda.html
[tsan-paper]: https://doi.org/10.1145/3779212.3790241
[tilelens]: https://github.com/Deep-Learning-Profiling-Tools/triton-viz
[tritonsan]: https://llvm.org/devmtg/2025-10/slides/technical_talks/lu.pdf
[kb]: https://github.com/ScalingIntelligence/KernelBench
[tb]: https://github.com/thunlp/TritonBench
[alive2]: https://pldi21.sigplan.org/details/pldi-2021-papers/5/Alive2-Bounded-Translation-Validation-for-LLVM
[kb-paper]: https://arxiv.org/abs/2502.10517
[tb-paper]: https://arxiv.org/abs/2502.14752
[kbv]: https://arxiv.org/abs/2607.16241
[rkb]: https://arxiv.org/abs/2509.14279
[sakana]: https://techcrunch.com/2025/02/21/sakana-walks-back-claims-that-its-ai-can-dramatically-speed-up-model-training/
