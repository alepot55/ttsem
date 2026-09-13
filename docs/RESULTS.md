# ttsem results

Numbers are from an RTX PRO 6000 (Blackwell, sm_120), Triton 3.8.0 wheel for the device side,
`triton-opt` at `d4eb2dd39` for generic printing, the fuzzer corpus of 5 September 2026
(the companion differential fuzzer, seeds 10 and 11 of every profile: 2,600 programs). A program is
`match` when every output buffer is bit-identical to the device after running the IR of its
launch on the CPU with the level-1 semantics.

## Level 1 over the corpus (5 Sep, commit `f1658ae`)

| family | programs | TTIR | TTGIR (final, layouts erased) | the rest |
|---|---|---|---|---|
| layout | 800 | 796 match | 796 match | 4 never compiled on the device (#11600) |
| narrow | 800 | 796 match | 796 match | 4 never compiled on the device (#11600) |
| loops | 800 | 795 match | 795 match | 5 out of shared memory on the device |
| tma | 800 | 545 match, 241 padding-only | 505 match, 208 padding-only, 73 warp-specialised | 14 out of shared memory on the device |

Every one of the 2,387 runnable programs of the first three families is bit-exact at both
stages, which is the calibration: the final TTGIR after coalescing, pipelining and layout
selection means, with layouts erased and asynchrony sequentialised, exactly what the TTIR meant.

The 241 tma programs that differ at TTIR are one thing: 193,320 differing elements, all of them
in the padding beyond the logical shape, none inside it, and they are exactly the 241 programs
the fuzzer had marked `pad_write` against the shipped interpreter. The device writes the columns
between `N` and the next 16-byte boundary on a descriptor store (#11583); the semantics,
following the documented clipping, does not. A reference semantics disagreeing with the
hardware on exactly the elements of an open issue is the behaviour wanted from it. At TTGIR
the 73 programs whose partitions wait on barriers are outside level 1 by design.

## Per-pass validation

`ttsem/validate_corpus.py` on seed 10 of three families (299 programs each that run on the device),
every whole-module dump from the inliner to the LLVM conversion, the pipeliner's intermediate
dumps as their own stages: 24,817 stage evaluations per family.

| family | stages | match | not run | mismatch |
|---|---|---|---|---|
| layout | 24,817 | 24,218 | 598 (one `!ttg.async.token` stage and the LLVM dialect, per program) | 0 |
| narrow | 24,817 | 24,219 | 598 | 0 |
| loops | 24,817 | 24,518 | 299 (the LLVM dialect; the token stage runs since the type became an opaque value) | 0 |

**No pass of the three pipelines changed any result with respect to the device: 74,451 stage
evaluations, 72,955 decided, 0 mismatches.** The first loops run had shown 2,557 mismatches in
32 programs; they were the validator's own defect, found by that run: `run_launch` registered the
record's pre-launch arrays in the memory model without copying, so the first stage's stores
leaked into the inputs of every later stage, and the 32 programs were exactly the ones whose
stores are not idempotent (accumulations, atomics). With the copy, the rerun is clean.

## Whole test suite (`python/test/unit/language/test_core.py`, 5 Sep)

`ttsem/pytest_ttsem.py` under the upstream suite on the RTX PRO 6000, 7,920 tests passed and 350
skipped exactly as without the plugin (one test hit the plugin's per-test timeout: 1,000
launches of 65,536 programs each), 14,291 launches validated at TTGIR in 11 minutes. The
second column is the same suite after the semantics and harness gaps that first run exposed
were closed (14,235 launches in 6.5 minutes with eight workers; the 56 missing ones are one
`test_gather` case that flaked on the device that run):

| verdict | first run | after the gaps were closed | what it is |
|---|---|---|---|
| match | 9,255 | **12,133** | bit-exact |
| approx | 267 | 1,317 | within the float policy of `SEMANTICS.md`: 1,006 reassociated or approximate, 289 within 2 ulp of f32 division, 22 NaN payloads |
| mismatch | 1,008 | 179 | 169 are the old value a `tt.atomic_rmw` returns, which depends on the order the lanes arrive; the other ten are the device deviations at the end of `SEMANTICS.md`, plus one tf32 chain dot just outside the band |
| error | 3,659 | **0** | were 3,499 hex float constants the parser read as decimals and 157 zero-dimensional bf16 values promoted to shape `(1,)` |
| unsupported | 101 | 605 | 576 are `ttg.fp4_to_fp` in `test_scaled_dot`, whose `tt.dot_scaled` is out of scope by design (the hex-float errors used to hide them); 28 are inline assembly, also by design; 1 is a `tt.atomic_poll` with no timeout waiting for a program this schedule runs later |
| poison | 1 | 1 | `MIN % -1` |

After `tt.dot_scaled` and `ttg.fp4_to_fp` entered the semantics, the suite was run once more
at both stages with the tree of commit `5e1d496` (round 3, 5 Sep, eight workers, about six
minutes per stage, the same 7,921 passed and 350 skipped as without the plugin):

| verdict | TTGIR | TTIR | what it is |
|---|---|---|---|
| match | **12,548** | **12,529** | bit-exact |
| approx | 1,484 | 1,504 | within the float policy |
| mismatch | 173 | 193 | the `tt.atomic_rmw` old values and the device deviations of `SEMANTICS.md`; the two columns differ on 35 + 15 atomic launches whose arrival order differed between the two device runs, plus one NaN-payload launch of `test_propagate_nan` |
| unsupported | 29 | 8 | `tt.elementwise_inline_asm` (28 and 7: PTX inside the IR, the continuation below `llvm.func` decides them) and one `tt.atomic_poll` without a timeout |
| poison | 1 | 1 | `MIN % -1` in `test_bin_op`, undefined by the semantics and by LLVM |

No launch errors, no launch left undecided for a reason other than inline PTX and one poll.

The 179 mismatches are the whole point of the exercise: with the errors gone and the float
tolerance in place, what is left is a list of places where the IR does not determine the
answer (the association of a `tt.reduce`, the arrival order of an atomic) or where the device
disagrees with its own documented meaning (a `tt.clampf` peephole, the sign of a zero
remainder). They are written down one by one under "Device deviations" in `SEMANTICS.md`, and
none of them was fixed by bending the semantics toward the hardware.

Five defects of the semantics were found and fixed on the way, each with a test: `i1`
arithmetic was numpy's bool arithmetic instead of mod 2 (`True + True` gave `True`);
`tt.histogram` kept a value equal to the bin count, which the shipped interpreter also does
and the device does not; `tt.reduce` re-encoded its bf16 accumulator, turning `256.0` into
`17280.0`; a `tt.atomic_rmw` on bf16 added the *bit patterns*, turning `1.734 + 0.319` into
`1.4e38`; and `math.fma` was an unfused `a * b + c` in `float64`.

Rerun on the pristine main build after every fix (5 Sep 13:22, the same 7,921 tests pass and 350
skip as without the plugin; the one failing test, `test_gather[src_shape3-indices_shape3-0]`,
fails identically without the plugin: it asks 128 KB of shared memory and the RTX PRO 6000 has
101 KB):

| stage | launches | match | approx | mismatch | unsupported | poison |
|---|---|---|---|---|---|---|
| TTGIR (final) | 14,235 | 12,134 | 1,316 | 179 | 605 | 1 |
| TTIR (final) | 14,235 | 11,244 | 1,075 | 179 | 1,736 | 1 |

The 179 mismatches are the documented device deviations and nothing else: 168 `test_atomic_rmw`
launches whose returned old values are a different permutation (arrival order; the final value
at the address agrees), 4 `remf` zero signs, 2 `clampf` NaN signs, the two argmax-with-NaN
butterfly cases, the bf16 sum of 1,024 ones, one atomic ordering in `test_constexpr_if_return`,
and one tf32 chain dot 8e-3 outside the band. The unsupported launches were `tt.dot_scaled`
(1,728 at TTIR, its `ttg.fp4_to_fp` lowering 576 at TTGIR), 28 inline-assembly kernels and one
atomic poll. Scaled dots are now modelled (decode e2m1/e4m3/e5m2/bf16/fp16 operands, apply the
e8m0 scales in the compute type the NVIDIA decomposition uses, accumulate as `tt.dot`): on the
1,728 scaled-dot tests (3,456 launches) both stages give 3,025 `match` and 431 `approx` at one
bf16 ulp, none unsupported; the TTIR op and its TTGIR decomposition agree launch by launch.
One thing the suite never draws is an e8m0 scale byte of 0xFF (NaN): `examples/witness_e8m0_nan.py`
shows the device returning NaN for that row and the shipped interpreter (`TRITON_INTERPRET=1`)
returning +inf, because its `_e8m0_to_f32` shifts the byte into an exponent and never masks the
NaN encoding; the semantics follows the device and the OCP MX definition.

## An open miscompile, localised to its pass automatically

triton#11519 (open, fix approved in #11521): `tritongpu-fuse-nested-loops` trusts a
`tl.assume` that sits on the branch the loop nest cannot reach, and on `m=4, n=0` the device
returns `out = [4, 8, 12, ...]` with 256 writes to a buffer the source never touches, where the
source program returns zeros and touches nothing. `ttsem/validate.py` on the standalone reproducer
(`examples/e15_repro.py`, run against main at `e1f944a7a` on the RTX PRO 6000), without being
told anything about the bug:

| stages | verdict against the device |
|---|---|
| the 20 whole-module dumps from the inliner to "Before TritonGPUFuseNestedLoops" | mismatch, 320 elements, all the same |
| every decided stage after it, down to the LLVM conversion | match |

The IR agrees with the device only once the fusion pass has run: the pass changed what the
program means, and the device faithfully executes the changed program. That boundary is what
`Report.culprit` names. Nothing in the tool knows about assumes, dominance or loop fusion; it
knows what every op means and asks, after every pass, whether the device still agrees.

**Closed upstream (8 Sep).** peterbell10 merged #11521 (`8f80860f1`) and closed #11519. On `main` at
`27964b601`, built from source on an RTX 4070 (sm_89), `python -m ttsem validate examples/e15_repro.py --device cuda`
now gives 63 decided stages, all `match`, and no culprit; the only remaining line is the `unsupported`
past the LLVM lowering. The same reproducer, the same tool, the verdict flips from "the fusion pass
changed the meaning" to "no pass changed the meaning": the fix is confirmed by the semantics, not by
reading the patch.

## Undefined behaviour introduced by a pass, on today's main

triton#11407 (closed): the software pipeliner moves the address arithmetic of a later
iteration into the current one and predicates the load but not the `arith.divsi` that
computes its address, so `(i * BLOCK) // (n - i)` divides by zero on a trip the source never
takes. The fix (#11410) was reverted a day later (#11427, "causes regressions and doesn't
really solve any practical problem"). On `examples/e2_repro.py`, main at `e1f944a7a` and
the 3.8.0 wheel alike:

| stage | verdict |
|---|---|
| final TTIR | match |
| every dump up to the pipeliner | match |
| pipeliner intermediate step 2 and the final TTGIR | `poison`: integer division by zero |

With the reverted fix re-applied to the same tree (`gh pr diff 11410`, incremental rebuild), the
final TTGIR is `match`: the division now sits inside the `scf.if` the fix creates. The tool
reports the defect and confirms the fix from the same reproducer, without knowing either.

The device output is right, because the masked copy never dereferences the poisoned address,
and that is the maintainers' "no practical problem". What the semantics states is narrower and
exact: after `tritongpu-pipeline` the program executes an operation whose result is undefined,
which the source program does not; the final TTGIR carries the bare `arith.divsi` next to the
masked `async_copy_global_to_local` it feeds. Level 1 sees the two programs mean different
things; whether an LLVM optimisation ever exploits that is a separate question.

Still there on 8 Sep: at `27964b601` (sm_89 build) `examples/e2_repro.py` gives 33 `match` up to the pipeliner
and 30 `poison` (integer division by zero) from pipeliner step 2 to the final TTGIR. The revert holds,
the issue stays closed, the program still divides by zero on a trip the source never takes.

## Level 2 (layouts), first results

`ttsem/layouts.py` converts every TritonGPU encoding to a linear layout as `LinearLayoutConversions.cpp`
does: 57 cases of the C++ unit tests and 3 lit cases reproduce exactly, and all 59 (encoding,
shape) pairs of the fixtures build. `ttsem/layout_checks.py` states four properties (convert_layout is a
bijection on the non-broadcast part, an `efficient_layout` gather is warp local as
`isWarpLocal` demands, a shared store-then-load under swizzling is the identity, a reduction's
result carries the slice of its operand's layout); `ttsem/check_fixtures.py` on the four TTGIR
fixtures: 36 ops checked, 0 violations, 0 coverage gaps. The gather check separates the two
layouts of triton#11600: the 3.8.0 release sizes the axis by the source length and its two-warp
8x2/8x8 case is flagged; main (#10838) sizes it by `max(src, idx)` and the same case is warp
local, over the 81 two-dimensional and 8,192 four-dimensional combinations swept.

Over the corpus (`ttsem/dump_corpus.py` then `ttsem/check_fixtures.py --dir`, the final TTGIR of the first
launch of every program that compiles on the 3.8.0 wheel, 3,173 of 3,200 including the two seeds
of every family): **28,483 ops checked, 0 violations, 0 coverage gaps**; 323 `efficient_layout`
gathers, 2,666 layout conversions, 1,877 shared loads and 104 shared stores under swizzling,
9,137 reductions, 14,376 `expand_dims`. The 27 programs without a dump are the ones the device
itself rejects (the #11600 assertion and out-of-shared-memory launches).

## What level 1 does not cover

Warp-specialised TTGIR whose partitions wait on barriers (rejected by design, the partitions
would read shared buffers before the producer fills them in program order), inline assembly,
and everything below the LLVM conversion.

## The minimiser (5 Sep, no GPU)

`ttsem/minimize.py` takes a program, finds the stage whose output is the first to disagree with the
reference (`Report.changed_by`, the mirror of `culprit`), and deletes operations from the module
that stage receives while the pass still changes its meaning under the semantics: the property
is a disagreement between two interpretations of the same inputs, so no device is involved, and
the pass runs through the wheel's own bindings, so no `triton-opt` is either.

On `examples/e15_repro.py` (#11519) on the laptop, cpu path, Triton 3.7.1 wheel:

| | |
|---|---|
| stage found | `Before TritonGPUFuseNestedLoops`, without being told |
| module | 48 lines to 33 (32 without locations) in 487 trials, 18 s end to end |
| what is left | the `scf.if` on `m == 0` with the `llvm.intr.assume` on `n`, the flattened nest with constant bodies, one store; every load became a zero constant and the address arithmetic behind it went with it |
| witness | `examples/e15_11519_min.mlir`, runs under `triton-opt --tritongpu-fuse-nested-loops` |

Two transforms per op, tried largest span first: delete it, or replace it by the zero constant
of its result type, which frees everything it used. Deletion alone had stopped at 39 lines with
the loads in place; the constant transform is what removed them. A third property is available
for the defects whose outputs do not change: `--reads` accepts a candidate when the pass's output
loads a byte address the input never loads (#11601, a load hoisted above a loop that does not run).

On `examples/e11601_repro.py` (#11601, found by a companion static witness, here with a real buffer for
`q` and `m = 0`), `--reads` on the cpu path: the stage is found from the report's `extra_reads`
(no output differs, so `changed_by` is empty), and the module goes from 29 lines to 18 in 177
trials, 12 s: an outer loop over `m` whose body loads `q` to bound an inner loop with a constant
body, and no store at all. What is left is exactly the claim of the issue: the pass reads `q`
before a loop that never runs (`examples/e11601_min.mlir`).

## The bot's first cycles on new commits of `main` (night of 5 to 6 Sep)

`main` moved twice overnight and the bot took both without anyone watching: incremental build,
the whole suite at TTGIR, the diff against the previous cycle, the race detector over every
distinct module.

| commit | cycle | launches | match | approx | mismatch | new bad, not flagged | modules racing under Membar's barriers |
|---|---|---|---|---|---|---|---|
| `aedd1f612` | 04:24 to 04:39 | 14,235 | 12,535 | 1,491 | 179 | **0** (22 flagged: atomics) | **0** of 9,372 |
| `20e6ba864` | 05:39 to 05:57 | 14,235 | 12,546 | 1,484 | 175 | **0** (18 flagged: atomics) | **0** of 9,372 |

Fifteen minutes per commit end to end. A regression in either column would have been a line in
the log with the launch, the stage and the module named, the morning after.

## Level 3: the race detector (5 Sep, evening)

`ttsem/races.py` replays one execution and logs, at every shared-memory op, the byte set each warp
touches (from the linear layouts, the view's origin in its allocation, `allocation.offset`) with
the warp's epoch, the number of barriers its group has passed. Two accesses from different warps,
overlapping bytes, one a write, same epoch: a race. Asynchronous copies are pending from issue to
completion and race with anything in their group that touches their bytes in between. Replicated
elements (several warps holding the same element) do not race with themselves. `DESIGN.md`,
"Level 3", has the model.

**Membar's own lit corpus** (`test/Analysis/test-membar.mlir` after `--allocate-shared-memory
-test-print-membar`, 120 functions, 110 run on synthetic inputs; the rest are polls that never
return, instrumentation ops, and loops the input never leaves):

| barriers | functions with a race |
|---|---|
| as Membar inserts them | **1**, and it is real: the lit function `async_copy_global_to_local` has no `async_wait` at all, so the load reads a copy still in flight |
| all stripped | 16 |

Only 16 of 110 race without barriers because almost every lit case stores and loads under the
*same* layout, so each warp reads back its own bytes and no barrier is needed in fact: Membar is
conservative there, the detector is not. The lit corpus is a test of the detector's precision,
not of its sensitivity.

**Real pipelined kernels** (the fuzzer's `loops` family, seed 10, final TTGIR of 299 programs
through the same two passes; 282 run):

| | |
|---|---|
| race with Membar's barriers in place | **0 kernels** |
| race with every barrier stripped | 89 kernels |
| barriers Membar inserted | 622 in 180 kernels |
| barriers necessary on their own (removing that one barrier alone makes a race) | **119 of 622**; 119 of 375 in the kernels whose warps exchange shared bytes, **0 of 247** in the kernels whose warps never do |

So on these kernels Membar is sound with respect to the dynamic semantics (no race survives its
barriers) and about one barrier in five is individually necessary; the rest are either redundant
with a neighbour or protect an exchange that the layouts never make. The `layout` and `narrow`
families do not touch shared memory at TTGIR (their `convert_layout` scratch is allocated below
it), and are silent. The `tma` family (warp-specialised producers, mbarriers) needed two model
decisions that the lit corpus did not: pending copies are scoped to the group that issued them,
and a partition's leftovers retire at the exit of `warp_specialize`, which is a CTA-wide sync
(#11324); with those, the three kernels examined by hand race with no barrier stripped and race
with all of them stripped. The two `tma` seeds in full, same protocol:

| family | kernels run | race with Membar's barriers | race with all stripped | barriers necessary on their own |
|---|---|---|---|---|
| loops, seed 10 | 282 | **0** | 89 | 119 of 622 |
| tma, seed 10 | 297 | **0** | 156 | 188 of 1,601 |
| tma, seed 11 | 489 | **0** | 274 | 315 of 2,578 |
| loops, seed 11 | 496 | **0** | 234 | 318 of 1,245 |

**Across groups (level 3.2, 6 Sep).** With the mbarrier phases and the vector clocks of
`DESIGN.md` "Level 3.2", the two `tma` families were run again: 786 kernels, 0 races with
Membar's barriers in place (the producer partitions and their consumers are ordered by the
barriers the compiler emits, on every kernel), 156 and 274 without, the same as under 3.1. The
detector's own tests pin the negative cases: a consumer that does not wait, or waits on the
wrong barrier, races with its producer.

**The test suite's own kernels.** The bot now keeps every distinct final-TTGIR module the suite
compiles (9,377 on `1665fa9a9`) and runs the detector on them each cycle. The first pass found
one model decision the fuzzer families had not forced: an op that lowers through a scratch
buffer of its own (`convert_layout` between layouts, `reduce`, `scan`, `histogram`, `gather`)
carries `allocation.offset` and emits a `bar.sync` between its scratch store and load, a
CTA-wide barrier that Membar relies on; ten `dot_scale_kernel` modules raced between a
`local_alloc` with an initialiser and a `local_load` under another layout until the detector
counted that barrier too. With it, the whole set:

| | |
|---|---|
| distinct final-TTGIR modules of `test_core.py` on `1665fa9a9` | 9,554 |
| run (the rest: 130 divisions by a synthetic zero, 28 inline PTX, 24 polls) | 9,372 |
| race with Membar's barriers in place | **0** |
| race with every barrier stripped | 1,435 |

Every module the suite compiles is free of intra-group races under Membar's barriers, and one in
six needs at least one of them. This is the column the bot now recomputes on every commit of
`main`: a module that races with the barriers in place is either a Membar regression or a
detector gap, and either is worth a look the same day.

**The #11325 class, re-found.** The four Membar tests of PR #11325 (a `memdesc_subslice` whose
offsets Membar compared in the frame of a different `memdesc_reinterpret`, so that row 16 of a
`32x16xf16` buffer and row 8 of the same bytes seen as `16x32xf16` looked disjoint and no barrier
separated a store from a load of the same 512 bytes) were run through `main`'s Membar:
`main` now inserts the barrier that the PR asked for (the PR was closed unmerged; the analysis
was reworked upstream). With that one barrier removed the detector reports the race in
`subslice_offsets_after_reinterpret` and in its element-type twin (12 warp pairs each) and
nothing in the two controls, which is the shape of
the finding the detector exists for: had it run on the `main` of 16 August, it would have named
this store and this load.

## Below `llvm.func`: the whole suite through the companion LLVM interpreter (5 Sep, evening)

`ttsem/pytest_ttsem.py` with `TTSEM_BELOW_LLVM=simtllvm.below_llvm:BelowLlvmAdapter`: every
launch of `test_core.py` is compiled once more with the dump on, and the stages from the first
`llvm.func` (the LLVM dialect through `LLVMDIScope`, ten of them) go to the adapter, whose
lock-step interpreter runs them on the recorded inputs and compares with the device. First
pass, `simtllvm` at `719a76f` (v1), 71 minutes for the suite:

| TTGIR verdict | below `llvm.func` | launches |
|---|---|---|
| match | match | **6,221** |
| match | unsupported | 5,942 |
| approx | unsupported | 1,442 |
| match | mismatch | 246 |
| mismatch | unsupported | 144 |
| match | error | 96 |
| the rest | | 79 |

Of 14,170 launches, 6,238 agree with the device all the way down to the last LLVM stage. The
unsupported column is v1's fragment, by stage occurrences: `llvm.inline_asm` 63,844 (Triton's
own PTX templates for loads and stores), `llvm.call_intrinsic` 4,450, 16-bit float constants
3,820, `nvvm.redux.sync` 680, `llvm.call` 440, `nvvm.stmatrix` 257, `llvm.fence` 140,
`llvm.atomicrmw` 70, `llvm.intr.abs` 50. The 246 mismatches under a green TTGIR are the
`test_bin_op` bf16 cases and are the adapter's comparison, not the execution: it compared
bf16 bit patterns raw where the tile levels decode them and apply the float policy; v2 routes
every buffer through `harness.compare` under `harness.float_policy`. The second pass with v2
(`76e22b7`, asm templates, f16/bf16 as real types, 30 NVVM intrinsics, `abs`) is queued behind
this one and records, for every launch, which passes changed the module.

Second pass, `simtllvm` v2 (`76e22b7`), 81 minutes:

| TTGIR verdict | below `llvm.func` | launches |
|---|---|---|
| match | match | **8,268** |
| match | unsupported | 3,911 |
| approx | unsupported | 796 |
| approx | mismatch | 369 |
| match | error | 193 |
| approx | error | 184 |
| mismatch | unsupported | 147 |
| approx | match | 136 |
| match | mismatch | 81 |
| match | approx | 49 |
| the rest | | 36 |

8,408 launches now agree to the last LLVM stage (from 6,238). What v2 opened is what v1 lacked:
the asm templates and the 16-bit types. What is left, by stage occurrences: `llvm.inline_asm`
27,441 (templates v2 does not have yet), `nvvm.ldmatrix` 15,070, `nvvm.stmatrix` 1,277,
`nvvm.redux.sync` 990, `llvm.call` 450 (libdevice), `llvm.intr.fmuladd` 250, `llvm.atomicrmw`
170, `llvm.fence` 140. The 369 `approx`-at-TTGIR launches that v2 calls mismatches are a
policy gap and not a semantics one: under `llvm.func` the reassociated reductions and the
approximate divisions are ordinary `fadd`s and `fdiv`s, `scan_inexact` finds no class, and
the comparison degrades to exact; the plugin now hands the tile level's scan to the
continuation as `record.inexact`. The errors are the adapter's memory model (`address ... in
space 1 is in no region`, 510 stage occurrences: a view whose storage starts before its
pointer, which the tile levels register whole at `record.bases`) and `undef` reaching
`insertelement` (120). Among the residual mismatches one is a real finding for `simtllvm`:
`test_reduce1d[min-uint64]` returns the signed minimum. Every class went back to the LLVM side.

Third pass, `simtllvm` v3 (`98e6a50`: 3-D grids, exact `sdiv`, unsigned min and max, views
registered at `bases`, the policy from `record.inexact`, `redux.sync`, `ldmatrix`/`stmatrix`,
atomics, fences, libdevice), 85 minutes:

| TTGIR verdict | below `llvm.func` | launches |
|---|---|---|
| match | match | **9,110** |
| match | unsupported | 2,980 |
| approx | unsupported | 724 |
| approx | approx | 398 |
| match | error | 216 |
| approx | error | 194 |
| mismatch | mismatch | 152 |
| approx | match | 146 |
| match | poison | 103 |
| match | approx | 49 |
| match | mismatch | 38 |
| the rest | | 60 |

9,259 launches agree to the last LLVM stage, the 152 `mismatch`/`mismatch` are the device
deviations both levels see the same way, and what remains has three shapes: `llvm.inline_asm`
templates the adapter has no grammar for (33,172 stage occurrences, mostly fp8 conversions
through `prmt`), address faults on predicated inline-asm loads (410 launches, `dot_scale_kernel`
and a few others: a masked load whose address is out of range must not fault when its guard is
false), and 103 launches the adapter grades `poison` under a matching TTGIR, to be read one by
one: either the LLVM level sees undefined behaviour the tile level does not model, or the
adapter's poison is too eager. The `match`/`mismatch` residue is 38 launches (`test_dot` 16,
`test_permute` 8, `test_sum_dtype` 4, `test_const` 4).

Fourth pass, `simtllvm` v4 (`91c3372`: `redux.sync` operands in the right order, arguments
bound by name, a block interpreter for the PTX conversion templates), 85 minutes:

| TTGIR verdict | below `llvm.func` | launches |
|---|---|---|
| match | match | **10,422** |
| match | unsupported | 1,620 |
| approx | unsupported | 749 |
| match | error | 410 |
| approx | approx | 408 |
| approx | error | 169 |
| approx | match | 164 |
| mismatch | mismatch | **145** |
| match | approx | 49 |
| the rest | | 34 |

10,590 launches agree to the last LLVM stage, and **no launch mismatches under a matching
TTGIR any more**: the 145 mismatches at the LLVM level are exactly the 145 device deviations the
tile level reports on the same launches (atomics' old values, NaN payloads, the documented
peepholes), read the same way by two semantics written by two sessions from two ends of the
pipeline. What is left is the adapter's fragment (`llvm.inline_asm` 18,992 stage occurrences,
`llvm.call` 80, `nvvm.cluster.arrive` 70, `nvvm.ldmatrix` 60, the mixed first LLVM stage's tile
types) and its errors (580 launches, 5,760 stage occurrences of one broadcast error in the new
PTX block interpreter on `dot_scale_kernel`'s fp8 conversions, plus 30 overflow and address
cases). From 6,238 to 10,590 in four passes over one night, each pass a list of causes handed
across and closed.

Fifth pass, `simtllvm` v5 (`edb940a`: `mma.sync` from the PTX ISA for every shape and type,
asm operands as single `.b32`, `cp.async`, cluster ops at one CTA, tensor descriptors bound
by name), 102 minutes:

| TTGIR verdict | below `llvm.func` | launches |
|---|---|---|
| match | match | **10,541** |
| match | approx | 1,399 |
| approx | approx | 1,268 |
| match | unsupported | 556 |
| approx | match | 164 |
| mismatch | mismatch | 149 |
| approx | unsupported | 55 |
| unsupported | approx or match | 26 |
| match | mismatch | **4** |
| the rest | | 8 |

13,401 launches are decided and consistent to the last LLVM stage (10,711 bit-exact, 2,690
within the float policy: the LLVM level grades the mma float forms with the wide band of
`tt.dot`, where the tile level often lands bit-exact). The unsupported column is down to 614
launches (`llvm.inline_asm` 1,070 stage occurrences from 18,992 one pass earlier, `llvm.call`
80, `nvvm.ldmatrix` 60, the mixed first LLVM stage's tile types). Twenty-six launches the tile
level cannot run for their inline PTX are decided by the LLVM level. What is left under a green
TTGIR is four launches of one test, `test_tensor_atomic_cas_multicta_result`, a compare-and-swap
across the CTAs of a cluster, where the adapter models cluster barriers at one CTA. From 6,238
to 13,401 in five passes over 24 hours, every pass a list of causes handed across and closed.

**The bot's first unattended cycles.** Overnight `main` moved twice (`aedd1f612`, `20e6ba864`)
and the bot took both on its own: build, the suite at TTGIR (14,235 launches each, no new bad
launch beyond the device-ordered atomics), the race detector over the 9,554 distinct modules
(0 races under Membar's barriers, 1,435 without, both cycles), 15 minutes per commit.

Sixth pass, `simtllvm` v6 (`3ede636`: clusters of more than one CTA declared `unsupported`
rather than judged, `llvm.call` to module functions as frames, cross-warp read-after-write
within a barrier phase as `poison`, the tile level's inexact classes followed exactly), 97
minutes:

| TTGIR verdict | below `llvm.func` | launches |
|---|---|---|
| match | match | **11,872** |
| approx | approx | 1,027 |
| match | unsupported | 554 |
| approx | match | 406 |
| mismatch | mismatch | 152 |
| match | poison | 75 |
| match | approx | 60 |
| approx | unsupported | 55 |
| unsupported | approx or match | 28 |
| the rest | | 5 |

13,396 launches decided and consistent to the last LLVM stage, **no mismatch under a green
TTGIR**, no error; the 152 mismatches are the device deviations both levels report alike. The
75 `poison` under a matching TTGIR are the LLVM level's new race rule on the suite's own
kernels: `test_load_store_same_ptr` (64), whose warps read and write one address without a
barrier, and `test_if_call` (8), the real race its kernel has; benign by luck, undefined by
the rules, and invisible to the tile level, which sees one program at a time. The unsupported
column is 610: `llvm.inline_asm` 1,050 stage occurrences, the mixed first LLVM stage's tile
types, `nvvm.ldmatrix` m16n16, clusters of more than one CTA declared as such.

**Which passes the suite wakes.** The same pass, over the 14,142 launches whose dump was
taken: every launch fires the structural passes; `cse` 99.7%, `coalesce` 96%,
`remove-layout-conversions` 96%, `inline` 67%, `sccp` 53%, `loop-aware-cse` 21%,
`accelerate-matmul` 17%, `reorder-instructions` 14%, `reduce-data-duplication` 11%; then a long
tail under 1%: `scf-to-cf`, the TMA passes, `combine`, `licm`, `reorder-broadcast`,
`F32DotTC`, `optimize-thread-locality`, `assign-latencies`, `coalesce-async-copy`, and
`fuse-nested-loops`, `optimize-accumulator-init` and `loop-unroll` on a handful of launches
each. Asleep on the whole suite: warp specialisation and its lowering, `fence-insertion`,
`optimize-dot-operands`, `combine-tensor-select-and-if`, `symbol-dce`, and the tensor-memory
passes (an sm_120 device has no tensor memory). `test_core.py` is a language test: the loop
and warp-specialisation passes, where this summer's bugs live, it barely touches.

## A cycle on a second machine (8 Sep): main at `27964b601` on an RTX 4070

The Blackwell card was unavailable, so the cycle ran on a desktop with an RTX 4070
(Triton `main` `27964b601` built from source in 30 minutes, sm_89, `-n 4`, niced). `test_core.py` at TTGIR: 7,904 passed, 366 skipped, 3 failed for shared memory
(131 KB asked, 101 KB on the card), 26 minutes.

| verdict | launches |
|---|---|
| match | 12,289 |
| approx | 1,499 |
| mismatch | 350, of which 176 are `test_atomic_rmw` flips against the sm_120 baseline, flagged nondeterministic (device ordering of atomics, an architecture effect, not a commit) |
| unsupported | 53 (52 inline PTX, 1 `atomic_poll` on another program) |
| poison | 1 (`MIN % -1`) |
| error | 10, all `test_cat_nd` |

The diff against the last sm_120 cycle (`20e6ba864`) is cross-architecture and says so:
of 187 new bad, 176 are the atomics above and eleven are real, and both real ones were defects
of the tool, found by the cycle and fixed the same night (`5d6c0ff`):

- `test_math_fma_op_edge_cases[float32]`, new upstream with the interpreter FMA (#11641, one of
  the sixteen commits): `math.fma` computed the exact product and sum as a rational but returned
  it through `float64`, so the cast to `float32` rounded a second time. The test is built on the
  inputs where two roundings and one disagree (`4097 * 4097 + 2**-30`, just above a `float32`
  midpoint, and `2**128 - 2**103 - 1`, just below the overflow midpoint). The rounding now goes
  straight into the result format; the launch is `approx` (two NaNs of different sign, the
  policy's NaN class), `float64` likewise.
- `test_cat_nd` (10 launches): on a device without TMA the frontend flattens a host
  `TensorDescriptor` into its fields, base pointer, shape and strides as i64, `padding == "nan"`
  and `round_f32_to_tf32` as i1, before the trailing shape and strides, 4r+3 block arguments per
  descriptor instead of 2r+1. The binder now tries that layout when the counts disagree; all ten
  are `match`.

Race detector on the 9,541 distinct final modules: 9,351 ran, **0 races with Membar's barriers,
1,321 without** (on sm_120: 9,372 ran, 0 and 1,435). Nothing in the sixteen commits since the last
sm_120 cycle changed a verdict that the tool or the architecture does not explain.

## Which passes a corpus wakes

`ttsem/pass_stress.py`: a pass fires on a program when the module it hands on differs from the one
it received (locations stripped). On 100 programs of the fuzzer's `layout` family (50 stages
each, compile only):

| fires on | passes |
|---|---|
| every program | the structural ones: the two conversions, allocation, warp groups, scratch, tensor memory, the two-CTA check |
| most | `inline` 99, `coalesce` 99, `sccp` 90, `convert-nv-gpu-to-llvm` 72, `cse` 60% of its four runs |
| some | `remove-layout-conversions` 38%, `canonicalize` 11% of nine runs, `loop-aware-cse` 11%, `optimize-thread-locality` 10, `reorder-instructions` 7, `reorder-broadcast` 2, `combine` 1, `F32DotTC` 1, `symbol-dce` 1; `schedule-loops` 29% on `loops`, 20% on `tma` |
| never | 30 passes of 48, among them the pipeliner and `schedule-loops` (no loops to see), `fuse-nested-loops`, `licm`, `loop-unroll`, `accelerate-matmul`, `optimize-dot-operands`, `prefetch`, `assign-latencies`, `coalesce-async-copy`, every TMA, TMEM and MMA lowering, warp specialisation, `scf-to-cf` |

A correction found on 6 Sep: `split_dump` used to leave the header line of a
pass's next intermediate dump at the end of the previous module, so two identical modules
differed by a comment and the pipeliner (three dumps per run) seemed to fire on every program,
loops or not. With the fix (`cc7734d`) the three families were re-measured; the tables above are
the corrected ones, and the pipeliner's row in the suite's `fired` column of the second to fifth
passes is inflated by the same artefact (every other row is unaffected: only stages followed by
an intermediate-step header were touched). A second artefact followed: the intermediate
dumps come in another printing form than the pass boundaries, so a pass with intermediate dumps
looked awake on every program; the measure now compares pass boundaries only, and on the
`layout` family the pipeliner is asleep as it should be.

So the `layout` family's zero mismatches say nothing about half the pipeline: it has no loops,
no dots and no descriptors, and the passes that own those never run on it. The other two
families, 150 programs each (compiled for cc 90, `main`, so the Blackwell passes
cannot fire by construction):

| family | stages | fires on every program besides the structural passes | fires on some | asleep |
|---|---|---|---|---|
| loops | 60 | `scf-to-cf`, `coalesce`, `sccp` | `licm` 83%, `cse` 77%, the pipeliner 17 to 47% of the programs of each `num_stages` value, `accelerate-matmul` 43%, `assign-latencies` 34 to 41%, `remove-layout-conversions` 38%, `schedule-loops` 29%, `reduce-data-duplication` 28%, `fence-insertion` 21%, `loop-aware-cse` 20%, `reorder-instructions` 17%, `F32DotTC` 8%, `canonicalize` 5% of eight runs | 29 |
| tma | 60 | TMA lowering, descriptor encoding, `scf-to-cf` 90% | `proxy-fence-insertion` 75%, `cse` 61%, the pipeliner 10 to 52% per `num_stages` value, `remove-layout-conversions` 45%, `licm` 39%, `assign-latencies` 24 to 33%, `accelerate-matmul` 28%, `reorder-instructions` 22%, `schedule-loops` 20%, `loop-aware-cse` 19%, warp specialisation 6 to 14% of the programs with the hint, `fence-insertion` 7%, `canonicalize` 3.5% | 22 |

**What no fuzzer family wakes** (asleep on all three): `fuse-nested-loops` (the pass of #11519
and #11601), `loop-unroll`, `optimize-dot-operands`, `optimize-thread-locality`,
`coalesce-async-copy`, `reorder-broadcast`, `combine`, `symbol-dce`, and every TMEM and MMA
lowering pass, which need a cc 100 compile. The next fuzzer families follow from this list:
nested loops with a hoistable bound, `tl.range(loop_unroll_factor=...)`, dots whose operands
come through a transpose or a convert, `cp.async` pipelines below Hopper, and a cc 100 compile
for the tensor-memory passes. The plugin records the fired passes per launch, so the same
table for `test_core.py` comes out of the second pass.

## The Inductor corpus, first slice (6 Sep)

`ttsem/inductor_trace.py` routes `torch.compile`'s kernels through `JITFunction.run` (Inductor's own
launcher bypasses it) so the plugin's hook sees every launch. `examples/inductor_models.py`:
an MLP, a transformer encoder layer forward and backward, a small CNN, a block of reductions
(softmax, layer norm, cumsum, argmax), an embedding and cross-entropy step, in f32 and bf16,
torch 2.11 on Triton `main`.

| | |
|---|---|
| launches validated at TTGIR | 80, of 46 distinct generated kernels (`poi` 17, `per` 19, `red` 10) |
| match | 45 |
| approx | 33 |
| mismatch | 2, both `native_dropout` fusions, 13,283 and 90 elements off by one rounding |

The two mismatches were one gap of the float policy, in three parts. Triton compiles with
`enable_fp_fusion` on, so LLVM contracts `a * b + c` into a single fused multiply-add with one
rounding where the semantics rounds twice; every fused pointwise kernel Inductor writes has the
pattern, and the language suite hardly ever does. `scan_inexact` now recognises a float add fed
by a float multiply as the class `contract`, held to the few ulp of a division; the band is also
absolute against the buffer's largest magnitude, because a fused `a * b + c` whose terms cancel
leaves a result far smaller than the rounding it absorbed; and it counts the buffer's own type,
because one rounding of an f32 product still flips the last bit of the bf16 it is stored as.
With Inductor's caches off every model of the slice ran, and with the three parts in place the
slice is clean: **93 launches of 55 generated kernels, 51 bit-exact, 42 within the policy, no
mismatch, no error**. The first run also met two Triton-`main`-on-torch-2.11 incompatibilities
that are not ours (Inductor pickling a `PyKernelArg`, a `CompiledKernel.__del__` on a half-built
kernel).

**Second slice** (`examples/inductor_models2.py`: attention through the math path, ten
libdevice transcendentals, reductions over the leading dimension, strided copies and permutes,
cumsum, cumprod and cummax, gathers and scatters with atomics, integer and boolean pointwise,
one-hot and bucketize, pooling backward, addmm epilogues; f32 and bf16): **42 launches of 29
generated kernels, 20 bit-exact, 22 within the policy, no mismatch, no error.** It cost one more
rule. Inductor's split scan (`triton_spl_fused_cumsum`) coordinates its programs through a
workspace: a dynamic block id from an atomic counter, then status words exchanged with
`atomic_xchg` for the decoupled lookback. Its outputs agreed within the wide band; its workspace
did not, because which slot a program claims is the order the programs arrived in, which a
sequential replay fixes one way and the device another. `Memory` now keeps the addresses that
`xchg` and `cas` touch, and a buffer they landed in is reported `approx` with the note "a buffer
of atomic exchanges", the whole buffer, since none of it is a result of the kernel.

Both slices together: **135 launches, 84 distinct kernels that `torch.compile` wrote, 71
bit-exact, 64 within the policy, none wrong.** Four rules came out of them, all on the
comparison side (contraction, the band's absolute part, the band's type, the exchange buffers),
none on the semantics of an op.

**Below `llvm.func`, on the Inductor kernels** (`simtllvm` v5): of the 135 launches, 20 are
decided and consistent to the last LLVM stage (18 bit-exact, 2 within the policy), 115 are
outside the adapter's fragment (767 stage occurrences of `llvm.inline_asm`: Inductor's kernels
are predicated and vectorised loads and stores in template forms the v5 grammar lacks, plus the
mixed first LLVM stage's tile types), and none is wrong, in error or poison. The Inductor corpus
is where the adapter's asm grammar gets its next entries.

**Two lanes, one address, two values.** One level down, the companion LLVM interpreter found a `tt.reduce`
epilogue that stores 32 lanes' partials to one shared word after a Welford combine that leaves
them differing in the last bit. At level 1 the same rule now guards every store of the program:
a store whose lanes carry one address with different bytes is `poison`. Over the whole
`test_core.py` at TTGIR (14,235 launches, 6 minutes) it flags nothing new: the only `poison`
stays `MIN % -1`, so the suite has no such store and the rule has no false positive on it.

## The crash minimiser on triton#11612 (6 Sep)

A fuzzer campaign found `tritongpu-fuse-nested-loops` asserting on a flattened loop
(triton#11612). `ttsem/minimize.py --crash` takes the 25-line TTGIR witness and a pass name and deletes
or constant-folds until the pass stops failing, each trial a `triton-opt` subprocess so the
assertion is theirs and not ours. Run alone, the pass also fails on modules the pipeline would
never hand it (dead inner loops canonicalisation removes), and the first minimum was one of
those; with `--canonicalize --cse` before the pass on every trial (`--pre`, on by default) the
minimum is the reachable form and matches the nine-line Python reproducer: an outer
`tt.flatten` loop whose body runs one inner `scf.for` with a carried value that a store uses
afterwards (`examples/e11612_repro_min.mlir`, 23 lines from 31 in 16 trials; the 97-line
verifier-error twin went to 35).

## The night of 13-14 Sep: `main` 972d18aa0 on sm_120, per pass, tensor memory, the detector per CTA

`main` 972d18aa0 built from source at 20:51 (12 minutes), the plugin with the partition scheduler,
the nvws arefs, the TMA gather and the Gluon recompile of 13 Sep, plus what the night itself added.

| corpus | launches or programs | match | approx | mismatch | unsupported | error | note |
|---|---|---|---|---|---|---|---|
| layoutfuzz `wsdesc` seed 30, device | 200 programs | | | 0 | | 7 device compile errors | all the sm_120 shared-memory limit (f32 128x64x64 blocks) |
| layoutfuzz `tma` seed 30, device | 200 programs | | | 0 | | 3 device compile errors | idem |
| `wsdesc` per pass, sm_120 | 194 programs, 193 launches, 17,756 stages | 8,502 | 8,125 | 0 | 193 (the llvm stage) | 184 (#11752) + 752 (the relink window; the 12 programs rerun: 450 match, 630 approx, 0 mismatch, 0 error) | last night 118 of 177 verdicts were `unsupported` |
| `tma` per pass, sm_120 | 198 programs, 197 launches, 18,124 stages | 12,722 | 5,178 | 0 | 197 (the llvm stage) | 27 (#11752) | |
| `wsdesc` per pass, **sm_100 on the cpu** | 24 launches, 2,215 stages | 2,160 | 0 | 0 | 24 (the llvm stage) | 24 (#11752) + 7 (nested pipeline) | 1,207 `unsupported` (`ttng.tmem_alloc`) before the tensor-memory semantics |
| `tma` per pass, **sm_100 on the cpu** | 24 launches, 2,208 stages | 2,182 | 0 | 0 | 24 (the llvm stage) + 2 (#11752) | 0 | tcgen05 and TMEM through the non-warp-specialized pipeline |
| layoutfuzz `wsdesc` seed 31, device | 200 programs | | | 0 | | 9 device compile errors | all the sm_120 shared-memory limit |
| layoutfuzz `tma` seed 31, device | 200 programs | | | 0 | | 0 | |
| `wsdesc` seed 31 per pass, sm_120 (fixed plugin) | 192 programs, 191 launches, 17,572 stages | 9,630 | 7,560 | 0 | 191 (the llvm stage) + 191 (#11752, classified) | 0 | |
| `tma` seed 31 per pass, sm_120 (fixed plugin) | 201 programs, 200 launches, 18,400 stages | 12,820 | 5,360 | 0 | 200 (the llvm stage) + 20 (#11752, classified) | 0 | |
| `test_core` TTGIR | 14,403 | 12,668 | 1,501 | 168 | 65 | 0 (+1 poison) | diff vs 12-13 Sep: 14 new bad, all the atomic-order class, 49 no longer bad; 1 upstream failure (`test_gather`, 128 KB) |
| 20 language files TTGIR | 3,088 | 2,718 | 308 | 0 | 30 | 9 (#11738) + 23 poison | diff: nothing new |
| `gluon` TTGIR, first pass | 5,858 | 5,607 | 47 | 15 | 165 | 24 | 128 no longer bad vs 12-13 Sep; 12 errors + 5 mismatches = the `local_scatter` operand order, 7 = cluster gathers, 2 = FMA dots, 1 = the reduction tree order |
| `gluon` TTGIR, second pass (fixed plugin) | 5,858 | 5,628 | 49 | 5 | 176 | 0 | 34 no longer bad vs the first pass; the 5 = 4 cluster gathers (broadcast variants, `unsupported` since `fd5884f`) + the reduction tree order; 141 of the unsupported are instrumentation modes |
| `unit/cuda` TTGIR, first pass | 40 | 16 | 4 | 12 | 8 | 0 | 11 fpsan, 1 tf32 descriptor |
| `unit/cuda` TTGIR, second pass (fixed plugin) | 40 | 17 | 4 | 0 | 19 | 0 | fpsan and consan `unsupported` by policy, the tf32 descriptor `match` |
| races, `test_core` corpus | 9,649 files, 9,435 ran | | | | | | 0 with barriers, 1,437 without |
| races, language corpus | 1,939 files, 1,910 ran | | | | | | **0 with barriers** (16 last night, the two-CTA modules), 547 without |
| races, gluon corpus | 5,712 files, 5,667 ran | | | | | | first pass 2 with barriers, both a `ttng.cluster_barrier` the detector did not count; with the fix **0 with barriers**, 392 without |

What the night changed in the tool (all committed, mirrored to the public package, shipped to the plugin):
`round_f32_to_tf32` on host descriptors (`ea55512`); the race detector with a shared memory per CTA
(`de84baa`: the 17 two-CTA "races with barriers" of 12-13 Sep were a fallback of the detector);
instrumentation modes as `unsupported` (`3b60bf4`); tensor memory and tcgen05 at level 1 (`4b4ba3d`);
stages `triton-opt` cannot re-read classified with the diagnostic (`242fcfb`); `ttg.local_scatter`
operand order (`b4b598b`); FMA dots in full f32 and cluster-wide gathers unsupported (`3833dce`, `fd5884f`);
`ttng.cluster_barrier` counted by the race detector unless relaxed (`18bcb68`).
Upstream: PR #11751 (the interpreter drops `round_f32_to_tf32`), PR #11752 (`nvws.warp_group` does
not round-trip: its printer drops the result types; every `MLIR_ENABLE_DUMP` stage between
`nvws-lower-aref` and `nvws-lower-warp-group` of a warp-specialized loop is unparsable).

Gaps left: level 2 has no checks on the tensor-memory ops; `tc_gen5_mma_scaled` and the
blocked-scales `tmem_copy`; DSMEM (cluster gathers, multicast MMA); the reduction tree order under
cancellation (`test_reduction_matches_loop`: 1e20 + 1 - 1e20 + 1, the device's tree gives 0, the
program order 1; a cancellation-aware band would call it approx); `tmem_load` with `redOp`.
