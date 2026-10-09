# ttsem

[![CI](https://github.com/alepot55/ttsem/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/alepot55/ttsem/actions/workflows/ci.yml)

Run the tests of your Triton kernels on a machine with no GPU and get the faults a GPU run hides:
a write past the end of a tensor, two program instances that touch the same element with at least
one of them writing it, an operation with an undefined result. Each one fails its test at the line
of your kernel. Memory nobody wrote is filled with NaN instead, so a read of it shows up in the
values. ttsem compiles every launch with Triton's own frontend and executes the IR it produces on
the CPU.

```console
$ pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
      'ttsem[triton] @ git+https://github.com/alepot55/ttsem'
$ python -m ttsem.sanitize pytest tests/ -q        # your suite, written for device="cuda"
```

The first command installs ttsem with Triton 3.8.0 and the CPU build of torch 2.14.0 (Linux,
Python 3.12 or 3.13); once ttsem is on PyPI, `'ttsem[triton]'` replaces the git requirement. The
second runs a pytest suite written for `device="cuda"`, unchanged. On [suite_no_mask.py][nomask], a
test file whose kernel forgets the mask on its store, the size that is a multiple of the block
passes and the other one fails at line 15 (from a clone; the last lines of the output):

```console
$ python -m ttsem.sanitize pytest tests/fixtures/agents/suite_no_mask.py -q
...
E   KernelFault: kernel `add_kernel`, launch 1: `suite_no_mask.py:15: tl.store(out_ptr + offs, x + y)` writes 24 element(s) past the end of the tensor passed as `out_ptr`. A GPU run usually hides this (allocations are rounded up). Check the mask of this access against the tensor's real size, for every program id and for sizes that are not a multiple of the block.
...
FAILED tests/fixtures/agents/suite_no_mask.py::test_add[1000] - KernelFault: ...
1 failed, 1 passed in 5.58s
```

`python -m ttsem.sanitize script.py` does the same for a script. In CI, the action runs your suite
this way on a CPU runner (pin it to `@v0.1.0` once that release is tagged):

```yaml
- uses: alepot55/ttsem@main
  with:
    install: pip install -e .        # your project
    args: tests/kernels -q           # pytest arguments
```

When ttsem runs the module every compiler pass receives, it names the first pass that changes what
a kernel computes. The demo replays the per-pass verdicts recorded on a real GPU launch of
triton#11519 (Triton `main` at `e1f944a7a`); the recording keeps the verdicts, so nothing is
executed. The end of its output:

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

A stage is the module its pass receives: stage 23 is what `tritongpu-fuse-nested-loops` receives and
stage 24 is what it returns. The device ran the miscompiled kernel, so it disagrees with every stage
before the faulty pass. In this recording stage 24, the module `canonicalize` receives, is
unsupported, so the recording alone cannot tell `tritongpu-fuse-nested-loops` from the
`canonicalize` after it (the demo names the first of the two); only the live run separates them.
`python -m ttsem demo --live` compiles the same program with the Triton 3.8.0 wheel, which still has
the bug, and runs every stage here against Triton's interpreter on the source: stages 1-20 match,
21-59 mismatch, and it names the same pass (13 s on the machine of "Limits" below). For
`torch.compile`, `python -m ttsem demo --torch` runs Inductor's Triton code for two PyTorch bugs
and, on torch 2.14, stops at the race in each ([docs/INDUCTOR.md][inductor]). For coding agents:
[docs/AGENTS.md][agents] and the skill [skills/triton-verify][skill].

## How it compares

Each cell is either measured here (the last column says so) or taken from what the tool's authors
published (documentation, source code, a talk), linked there. A blank cell is one we could not
check.

| | needs a GPU | access out of bounds | race between program instances | read of memory nobody wrote | source |
|---|---|---|---|---|---|
| ttsem | no | fails at the kernel's line, counted against the tensor's real size | fails at the kernel's line, for launches of up to 4,000,000 element accesses | `torch.empty` and its kin are filled with NaN (integers with a pattern), so the read shows in the values | measured here |
| `TRITON_INTERPRET=1` | no | not reported: on a copy of `suite_no_mask.py` with CPU tensors, the n = 1000 case corrupts the heap; in 19 runs of 20 the process died of a segmentation fault or an abort (once after both tests had passed), and in the other the store raised an `IndexError` about an array of size 0; no run named the cause | not reported as a race: it runs the program instances one after the other | not reported: on a copy of `accumulate_into_empty.py` with CPU tensors, the read returns what the CPU allocator left there (wrong values in 3 runs of 3) | measured here; [Triton docs][interp] |
| compute-sanitizer memcheck | yes | reported; an overflow into the next allocation only with `--padding`, and PyTorch's docs ask to turn its caching allocator off for memory checkers (`PYTORCH_NO_CUDA_MEMORY_CACHING=1`) | | | [NVIDIA][cs], [PyTorch][ptcuda] |
| compute-sanitizer racecheck | yes | | shared memory only, not global memory | | [NVIDIA][cs] |
| compute-sanitizer initcheck | yes | | | reported, in device global memory by default | [NVIDIA][cs] |
| Triton-Sanitizer (ASPLOS '26), part of TileLens | not for its basic workflows, which its README runs through Triton's interpreter | reported: each load and store, pointer and mask, checked symbolically with the Z3 solver | its README names a `tile-race` command without saying what it reports | | [README][tilelens], [source][tsan-src]; [paper][tsan-paper] |
| Triton-San | no: the kernel is lowered to CPU code through triton-shared | AddressSanitizer | ThreadSanitizer, one OpenMP thread per program instance | named among its targets | [LLVM Dev Meeting 2025][tritonsan] |
| GPU output compared with a reference | yes | seen only when values change or the run crashes: 86 of 95 in the control below | seen only when the values of that run change | seen only when values change | measured here |

What ttsem executes is the IR Triton's compiler produces: TTIR for a test run, and under the
validator the module every pass receives, down to the LLVM conversion, so the run that checks a
kernel can also name the pass that broke it. Triton's interpreter simulates the kernel's Python
with numpy instead of compiling it, and Triton-San lowers it through triton-shared, a different
pipeline that the program has to select in its code.

## Evidence

**Against the device.** On Triton's own `test_core.py` (14,235 launches, RTX PRO 6000, 5 Sep 2026),
12,548 launches are bit-exact with the device, 1,484 within the float policy written in
[docs/SEMANTICS.md][semantics], 173 mismatch, 29 are unsupported (28 of them inline PTX) and 1 is
undefined (`MIN % -1`). Most of the 173 are the order in which an atomic returns old values, which
depends on the order the lanes arrive in; each class of mismatch is described in
[docs/SEMANTICS.md][semantics]. The recordings behind these numbers are not in this repository;
[docs/RESULTS.md][results] has the method and the other corpora.

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
4070 gets 856 wrong, 855 of them only at a large shape. Inductor emitted each program's Triton for
the GPU at small or medium shapes (on the same RTX 4070), and ttsem, judging that code with no GPU,
flags all 856, plus 90 programs the device does not confirm at those shapes. These five families are
also where the judge's choices were tuned (the block cap, a second block size, a second input, the
medium shapes, not counting a store of the value already there), so they are not a held-out test. On
programs drawn at random from a separate generator, with no GPU at all (Inductor's Triton for CPU
tensors, judged by ttsem, set against the RTX 4070), ttsem flags 74 of the 88 programs the device
gets wrong in seed 1 and 81 of the 92 in seed 2 (82 when the CPU code is scheduled as CUDA code is).
Seed 1 also flags 50 programs the device gets right: races in the CPU code that the CUDA code need
not have. In seed 2 a rerun of 7 Oct, with the block capped at 16 and then 4, flags 135 programs, so
at least 43 of them are programs the device gets right. Seed 1 is not fully held out either: it was
drawn before the generator had its two rules (keep a program only if eager agrees with its
functionalized form, and drop an update that reads a view of its own destination), and it is where
those rules came from. With two fixes loaded into torch, our #198010 (merged) and #185450 (another
contributor's fix for `index_put_`, not merged when this was measured), the three families they
address, `views`, `alias` and `scatter` (2,092 programs), give 0 wrong and 0 flagged. The generators
of these programs are not in this repository.

**Repair from the verdict, a negative result.** On 145 TritonBench-T tasks, 67 answers of
AutoTriton-8B pass a check that only runs the file and can be judged; 31 of them (46%) are wrong,
unsafe or launch no kernel. Given ttsem's message for three more rounds, the model repaired none of
the 31. The harness of that experiment is not in this repository.

## Upstream bugs from this project

State on 8 October 2026; [LEDGER.md][ledger] has every row with its dates and, for the reported
ones, a link to the report that holds its reproducer. Once the package is on PyPI, its page there
keeps the state of its release. **PyTorch**, since 20 Sep: 18 defects reported (17 issues, one
reported with its fix); 7 fixed on `main`, 6 of them by 5 PRs of ours and one by another
contributor; 3 more fixes of ours are open PRs, one of them approved. **Triton**: 4 bug-fix PRs of
ours merged, 7 merged in all (the other 3 are a documentation change and two optimizations). "ours"
marks a PR by the author.

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

The race detectors, the validator, the minimiser and the memory model are in this package. The
PyTorch program generator (whose fourth tier adds aliased inputs, integers, freezing and autograd),
the OpInfo layout sweep, the program fuzzer, the static witness and the LLVM continuation are
companion tools that are not; [LEDGER.md][ledger] describes each.

## Questions a skeptic asks

- How many of the listed bugs did ttsem find itself? Going by the "found by" column of the 53
  upstream rows of [LEDGER.md][ledger], 6: pytorch#197829, #198031 and #198033 (the race detector),
  triton#11752 (the validator), #11738 (the memory model) and #11751 (found while modelling the
  semantics). 7 more surfaced while Triton's own suites ran under the validator, but fail without it
  too: triton#11730, the two defects of #11733, #11746, #11747, #11831 and #11859. The other 40 were
  found by hand or by the companion tools. In 5 of those a tool of this package also took part: the
  validator named the pass of #11407 and #11519, the level-2 gather check reproduced the boundary of
  #11600, and the minimiser produced the witnesses of #11601 (with the validator's read set) and
  #11612.
- Does it raise false alarms on correct code? On KernelBook's Triton code generated by Inductor,
  16,858 rows ran and raised 4 reports, all true: three races, all the same Inductor miscompile, and
  one undefined value, an `index_put` with duplicate indices (20-21 Sep, at the code's own block
  sizes; the harness is not in this repository). Liger-Kernel's own kernel tests, unmodified
  (`c5b9f31`, 19 Sep, the semantics of that day): 173 passed, 0 faults, 3 failures at tolerances
  chosen for two GPU runs, here against torch on the CPU; four files with 4096 x 32000 tensors did
  not finish.
- What is the TritonBench denominator? 7,752 published answers, of which 1,002 ran under ttsem and
  767 could be judged by value; the 32% is 245 of those 767. Comparing values on the RTX 4070 alone
  finds 220 of the 816 answers it ran and compared wrong (27%, audit of 19-20 Sep); ttsem, with no
  GPU, finds 147 wrong and 98 unsafe of 767 and names the fault and its line
  ([audits/tritonbench][audit-tb]).
- How long does a verdict take? About 12 s for a small KernelBench answer on the machine of "Limits"
  (11.4 s to 11.8 s over two runs each of `tests/fixtures/judge/task_relu.py` with
  `answer_unsafe.py` and with `answer_ok.py`): two sandboxed processes, the task's reference and the
  answer. The 7,752 TritonBench answers took about 75 minutes on a 20-core desktop.
- Which targets and versions? Kernels are compiled for CUDA sm_90 (a script run can pick another
  with `--cc`; the judge uses sm_90a); there is no AMD back end; only Triton 3.8.0 and torch 2.14.0
  are tested, the versions the `triton` extra pins.
- Is the calibration a held-out test? No. Triton's `test_core.py` was also the development set: its
  first run had 1,008 mismatches and 3,659 errors, and the semantics and the harness were fixed
  until what was left had a documented cause ([docs/RESULTS.md][results]). Two other corpora: the
  2,387 runnable programs of three fuzzer families, bit-exact at TTIR and at the final TTGIR, and
  the 3.8.0 release's own `test_core.py` on sm_120 (14,979 launches on 12 Sep, no unexplained
  mismatch; triton#11730 in [LEDGER.md][ledger]). That one is the release branch's copy of the
  same suite, so it is not independent of the development set either.

## Limits

- **Linux only.** Triton 3.8.0 publishes wheels for Linux (x86_64 and aarch64) only, and the judge's
  sandbox is `bwrap`.
- **An interpreter, not a GPU.** No timing. On a 4-core cloud VM (Intel Xeon at 2.8 GHz), under
  `python -m ttsem.sanitize`, a vector add over 16.8 million float32 elements took 16.6 s and a
  1024 x 1024 x 1024 fp16 matmul in 64 x 64 x 32 tiles 12.2 s.
- **Races are checked up to 4,000,000 element accesses per launch.** A larger launch still runs and
  is checked for out-of-bounds accesses. A script run ends with the number of launches too large to
  be checked for races: the run of the two launches above ends with `2 launch(es) too large to be
  checked for races`. Under `python -m ttsem.sanitize pytest` the terminal summary prints the same
  count.
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
value. The verdict is `verified` (exit 0), `wrong` (where memory nobody wrote shows up, as NaN in
the values), `unsafe` (an out-of-bounds access, a race, or an operation with an undefined result,
at the answer's line), `no_kernel` (PyTorch does the work), `error`, `too_slow` or `not_judged`.
Each answer is model-written code and runs in its own `bwrap` sandbox, and the verdict is computed
outside it, from the run's raw outputs. `wrong` versus `verified` tells whether the values are
right and holds against an adversarial answer, since the comparison rests on a reference the answer
cannot touch (`verified` means right values; a kernel launched with the work then done in PyTorch
still verifies). `unsafe` and `no_kernel` rest on two stop-gaps, a source scan and a run-time
integrity check of the whole checker, that block every attack in the test suite but are not a
proof: until the answer and the checker run in separate processes (planned for 0.2), use the judge
as an auditor and an evaluation tool, and as an RL reward against adversarial optimisation with that
caveat. The rules, the batch mode, the classes, the sandbox and what each verdict guarantees are in
[docs/JUDGE.md][judge].

## How it was built

Built by Alessandro Potenza with Claude Code (Anthropic's coding agent) as the engineering agent
throughout; the author set the direction and the constraints and reviewed the results. Every
PyTorch patch was A/B tested on PyTorch's own suites before it was sent (a first draft of the
#198033 fix failed 200 optimizer tests there). On 23 Sep a lerp fix passed its A/B, then the fuzzer
under dynamic shapes corrupted the heap through it: the cause was the `copy_strided` bug above, in
stock PyTorch.

## Related work

Alive2 ([Lopes et al., PLDI 2021][alive2]) validates LLVM's passes by bounded translation validation
with an SMT solver; ttsem executes a tile IR on concrete inputs and compares the results, a
different method on a different IR. GPUVerify ([Betts et al., OOPSLA 2012][gpuverify]) verifies
statically that CUDA and OpenCL kernels are free of data races and barrier divergence, and GKLEE
([Li et al., PPoPP 2012][gklee]) explores CUDA programs by concolic execution to find races and to
generate tests; both work on the threads of a CUDA kernel (GPUVerify also OpenCL), where ttsem runs
Triton's tile IR on the inputs a test gives. The judge applies KernelBench's correctness rule
([Ouyang et al., 2025][kb-paper]) and runs TritonBench's tasks ([Li et al., 2025][tb-paper]) as that
benchmark runs them, comparing values where it compares printed output. KernelBench-Verified ([Zhang
et al., 2026][kbv]) and robust-kbench ([Lange et al., 2025][rkb]) harden KernelBench's check with
more inputs. In February 2025 Sakana AI walked back the speedups claimed for its AI CUDA Engineer
after the system was found to exploit its evaluation harness ([TechCrunch][sakana]).

## Reading further

| file | what is in it |
|---|---|
| [LEDGER.md][ledger] | every upstream defect, with its reproducer, its dates and its fix |
| [docs/OVERVIEW.md][overview] | calibration against the device, how the three levels work, the limits |
| [docs/SEMANTICS.md][semantics] | what each op is defined to mean, the float policy, and the device deviations |
| [docs/DESIGN.md][design] | the three levels, the interfaces between them, and why they are drawn there |
| [docs/RESULTS.md][results] | the corpora, the counts, and the calibration of the semantics against the device |
| [docs/JUDGE.md][judge] | the judge of model-written kernels: rules, verdicts, batches, the sandbox |
| [docs/INDUCTOR.md][inductor] | the Triton code of `torch.compile` under ttsem |
| [docs/AGENTS.md][agents] | for coding agents that write Triton kernels |
| [audits/](https://github.com/alepot55/ttsem/tree/main/audits) | the commands that redo the TritonBench and KernelBench audits from a clone |
| [CONTRIBUTING.md][contributing] | how a new op, a new check or a new witness gets in |

A report on the semantics, the validator and what they found is in preparation; until then, cite
this repository with the metadata in `CITATION.cff` and the commit you used. MIT licence, see
`LICENSE`.

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
[tilelens]: https://github.com/Deep-Learning-Profiling-Tools/tilelens
[tsan-src]: https://github.com/Deep-Learning-Profiling-Tools/tilelens/blob/main/tilelens/clients/sanitizer/sanitizer.py
[tritonsan]: https://llvm.org/devmtg/2025-10/slides/technical_talks/lu.pdf
[kb]: https://github.com/ScalingIntelligence/KernelBench
[tb]: https://github.com/thunlp/TritonBench
[gpuverify]: https://doi.org/10.1145/2384616.2384625
[gklee]: https://doi.org/10.1145/2145816.2145844
[alive2]: https://pldi21.sigplan.org/details/pldi-2021-papers/5/Alive2-Bounded-Translation-Validation-for-LLVM
[kb-paper]: https://arxiv.org/abs/2502.10517
[tb-paper]: https://arxiv.org/abs/2502.14752
[kbv]: https://arxiv.org/abs/2607.16241
[rkb]: https://arxiv.org/abs/2509.14279
[sakana]: https://techcrunch.com/2025/02/21/sakana-walks-back-claims-that-its-ai-can-dramatically-speed-up-model-training/
