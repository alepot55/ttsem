# ttsem

`ttsem` is an executable semantics of Triton's tile IR. It parses TTIR and TTGIR in generic MLIR
form, runs a kernel on the CPU **from its IR** rather than from the Python source, and compares the
result with what the GPU actually produced. Because the semantics runs on IR, it can be run on the
IR after *every* compiler pass, on the same inputs, and the first pass whose output stops agreeing
is named. Triton has a large test suite and no written definition of what a pass must preserve;
this package is that definition, in a form you can execute. Alive2 did this for LLVM's scalar IR.
Nothing did it for a tile IR, whose values are distributed tensors: here layouts are erased
(level 1), then modelled (level 2), then the warps are made concurrent and shared-memory races are
reported (level 3).

## Try it in two minutes

```console
$ python -m ttsem demo
```

No GPU, no `triton-opt`, no build, and not even the Triton wheel: the default demo replays the
recording committed in `examples/`, which is the verdict the semantics reached on every one of the
88 stages one launch of `examples/e15_repro.py` went through. Consecutive stages that agree are
collapsed into one line.

```
replaying a recorded per-pass validation of e15_repro.py
  recording   examples/e15_11519_main_e1f944a7a.json
  reference   a real launch, recorded on Triton main e1f944a7a

  launch 0 (kernel), 88 stages
  stages    1-23  mismatch     n_diff=320   inline .. tritongpu-fuse-nested-loops
  stages      24  unsupported  n_diff=-     canonicalize  (arith.ceildivsi)
  stages   25-57  match        n_diff=0     triton-licm .. triton-nvidia-gpu-remove-tmem-tokens
  stages      58  unsupported  n_diff=-     canonicalize  (type: !ttg.async.token)
  stages   59-87  match        n_diff=0     triton-loop-aware-cse .. convert-triton-gpu-to-llvm
  stages      88  unsupported  n_diff=-     initialize-ws-cluster-barriers  (<llvm dialect>)

first pass that changes the meaning: tritongpu-fuse-nested-loops (triton#11519, fixed upstream in #11521)
```

Read the table upward from the bottom: the device computes what the IR means from stage 24 on, and
something else up to stage 23. The boundary is the pass that changed the program. That name is not
written anywhere in `demo.py`; `ttsem.validate.culprit_of` derives it from the verdicts.

The recording keeps the verdicts and not the modules, because a stage dump of this pipeline is
megabytes of MLIR. To redo the whole thing here, with the wheel installed (`pip install
'ttsem[triton]'`) and still no GPU:

```console
$ python -m ttsem demo --live
```

That compiles `examples/e15_repro.py` through the fake driver with `MLIR_ENABLE_DUMP=1`, runs the
module every pass receives through the interpreter on the launch's own inputs, and names the same
pass. It takes tens of seconds. The reference differs (Triton's own interpreter here, a GPU in the
recording), so the run of mismatches sits on the other side of the boundary:

```
  launch 0 (kernel), 60 stages
  stages    1-20  match        n_diff=0     inline .. tritongpu-fuse-nested-loops
  stages   21-59  mismatch     n_diff=320   canonicalize .. convert-triton-gpu-to-llvm
  stages      60  unsupported  n_diff=-     initialize-ws-cluster-barriers  (<llvm dialect>)

first pass that changes the meaning: tritongpu-fuse-nested-loops (triton#11519, fixed upstream in #11521)
```

Nothing in the tool knows about assumes, dominance or loop fusion. It knows what every op means,
and it asks, after every pass, whether the answer is still the same one.

## What it found

Upstream defects where this tool, the per-pass validator, the delta minimiser, the race detector or
the differential fuzzer they share named the culprit pass or produced the witness. State as of
12 September 2026; the full table, with reproducers and dates, is in [LEDGER.md](LEDGER.md).

| bug | pass or component | state (2026-09-12) | fix |
|---|---|---|---|
| [triton#11519](https://github.com/triton-lang/triton/issues/11519) | `tritongpu-fuse-nested-loops`, `matchPositiveTripCount` | fixed | [#11521](https://github.com/triton-lang/triton/pull/11521) by LiRunGuo, `8f80860f1`, merged by peterbell10 |
| [triton#11601](https://github.com/triton-lang/triton/issues/11601) | `tritongpu-fuse-nested-loops`, `flatten=True` | filed, fix PR open | [#11692](https://github.com/triton-lang/triton/pull/11692) by alepot55 |
| [triton#11612](https://github.com/triton-lang/triton/issues/11612) | `tritongpu-fuse-nested-loops`, assertion on a flattened nest | filed, fix PR open | [#11617](https://github.com/triton-lang/triton/pull/11617) by sweiglbosker |
| [triton#11614](https://github.com/triton-lang/triton/issues/11614) | cross-warp `tt.reduce` epilogue, unpredicated shared store | filed, fix PR open | [#11630](https://github.com/triton-lang/triton/pull/11630), reworked at `20365b7` after review by ThomasRaoux |
| [triton#11407](https://github.com/triton-lang/triton/issues/11407) | `tritongpu-pipeline`, `predicateOp` | closed; fix landed then reverted, the UB is still on `main` | [#11410](https://github.com/triton-lang/triton/pull/11410) by lezcano, `ad81746db`, reverted by [#11427](https://github.com/triton-lang/triton/pull/11427) by ThomasRaoux, `bf9ad8723` |
| [triton#11544](https://github.com/triton-lang/triton/issues/11544) | `AxisInfo`, `remsi` / `remui` divisibility | fixed, three hours after the report | [#11546](https://github.com/triton-lang/triton/pull/11546) by lezcano, `4af25c84e` |
| [triton#11548](https://github.com/triton-lang/triton/pull/11548) | `AxisInfo`, constancy of a masked `tt.load` with `other` | fixed; found and fixed upstream the same afternoon our witness existed | [#11548](https://github.com/triton-lang/triton/pull/11548) by lezcano, `36cb2499e` |
| [triton#7749](https://github.com/triton-lang/triton/issues/7749) | `AxisInfo`, `divsi` / `remsi` on negative dividends | confirmed, made observable on device by us; fix PR parked by design review | [#11569](https://github.com/triton-lang/triton/pull/11569) by alepot55, open without further commits at the maintainer's request |
| [triton#11583](https://github.com/triton-lang/triton/issues/11583) | `desc.store` (TMA), writes past the inner extent to the 16-byte boundary | filed, open | none |
| [triton#11586](https://github.com/triton-lang/triton/issues/11586) | `NVGPUWarpSpecialization`, `WGMMAOpPattern::getPtxAsm` (int8 wgmma) | filed, reproduced by a third party and re-confirmed by us on `main` | none |
| [triton#11587](https://github.com/triton-lang/triton/issues/11587) | `NVGPUWarpSpecialization`, tile partitioning with host descriptors | filed, open | candidate [#11596](https://github.com/triton-lang/triton/pull/11596) by layahaasini |
| [triton#11600](https://github.com/triton-lang/triton/issues/11600) | `setOptimizedGatherLayout`, leftover warps on the gather axis | fixed in `main`, not in the 3.8.0 release | [#10838](https://github.com/triton-lang/triton/pull/10838) by Jokeren, `2c008494e` |
| [triton#11581](https://github.com/triton-lang/triton/issues/11581) | ptxas 12.9.86 loses the immediate of the second `add.s16x2` | filed, no response | none (fix is on the LLVM side or in the pinned ptxas) |
| [triton#11328](https://github.com/triton-lang/triton/issues/11328) | `ClusterBarrierInsertion`, a multicast TMA through a `memdesc` view | fixed | [#11411](https://github.com/triton-lang/triton/pull/11411) by Jokeren, `c55f21c06` |
| [triton#11326](https://github.com/triton-lang/triton/issues/11326) | `Membar` at a call boundary | fixed by the `BufferRegionAnalysis` rework; no targeted PR | the Membar rework |
| [triton#11325](https://github.com/triton-lang/triton/pull/11325) | `Membar`, subslice offsets compared across frames | closed unmerged by Jokeren; the class is re-found by the race detector | none |
| [triton#11404](https://github.com/triton-lang/triton/issues/11404) | `Membar`, a relaxed `ttng.cluster_barrier` treated as a full sync | fixed | [#11462](https://github.com/triton-lang/triton/pull/11462) by gaoxiaomo, `c0901bf1c` |

## Calibration

A semantics is only worth what it has been checked against. The calibration corpus is Triton's own
language test suite, run under the pytest plugin, which validates every launch the suite makes.

- **The whole upstream `test_core.py`, 14,235 launches**, validated at the final TTGIR:
  **12,548 `match`** (bit for bit against the device), **1,484 `approx`** (inside the float policy
  written down in `SEMANTICS.md`: reassociated reductions, divisions within 2 ulp, NaN payloads),
  **173 `mismatch`**, **29 `unsupported`** (28 of them inline PTX), **1 `poison`** (`MIN % -1`).
  No launch errors, and **no unexplained mismatch**: every one of the 173 is one of the device
  deviations enumerated at the end of `SEMANTICS.md`, chiefly the permutation of old values a
  `tt.atomic_rmw` returns, which depends on the order lanes arrive in. The semantics was never bent
  toward the hardware to make a number come out.
- **13,396 of 14,234 launches are decided and consistent all the way down to the last LLVM stage**,
  through the companion lock-step LLVM interpreter that takes over at the first `llvm.func`
  (11,872 bit-exact plus 1,027 inside the float policy, plus the launches both levels grade the
  same way). No launch mismatches under a green TTGIR. The 152 that do mismatch are the same device
  deviations, read the same way by two semantics written from the two ends of the pipeline.
- **Per-pass validation over the fuzzer corpus: 74,451 stage evaluations, 72,955 decided,
  0 mismatches.** No pass of three pipelines changed any result with respect to the device.
- **Level 2 (layouts): 28,483 ops checked over the corpus, 0 violations, 0 coverage gaps**, plus 57
  cases of Triton's own C++ unit tests for `LinearLayoutConversions.cpp` reproduced exactly.
- **Level 3 (races): 0 races with Membar's barriers in place over 9,372 distinct modules of the
  test suite, 1,435 races with every barrier stripped.** Membar is sound on everything the suite
  compiles, and about one barrier in six modules is individually necessary.

## How it works

**Level 1: TTIR and TTGIR with layouts erased.** Tensors are numpy arrays in logical index order,
pointers are int64 addresses into a flat byte-addressed memory, control flow is executed, async
copies are sequentialised in program order. `ttsem/mlir.py` parses the generic form, which is regular
enough that one parser covers every dialect; `ttsem/ops.py` gives each of 151 ops a meaning; `ttsem/interp.py`
runs a region, a function, or a whole grid, and raises `Unsupported` by name on anything it does
not model, so coverage is measurable instead of silent. `ttsem/harness.py` records real launches, copies
the arguments to the host at their device addresses, runs the grid, and compares every output
buffer: bitwise, always, for the verdict `match`, and under a documented float policy for `approx`.
`ttsem/validate.py` splits an `MLIR_ENABLE_DUMP=1` trace into one module per pass, runs each on the same
inputs, and names the first pass after which the answer changes; `ttsem/minimize.py` then shrinks that
module while the pass still changes its meaning, with no GPU and no `triton-opt` involved, because
the property is a disagreement between two interpretations of the same input.

**Level 2: distributed values.** Level 1 answers "does this kernel compute the right array".
Level 2 answers "and does each thread hold the right part of it". `ttsem/linear_layout.py` is a port of
Triton's GF(2) linear layout algebra and `ttsem/layouts.py` a port of its encoding-to-layout map, so a
distributed value is a pair of the same logical array and a layout saying that register `r` of lane
`l` in warp `w` holds `array[layout.apply(...)]`. Every op implementation is untouched: level 2 adds
an assertion about the layout, not a different value. `ttsem/layout_checks.py` states the properties a
lowering must keep, `ttg.convert_layout` being a bijection on the non-broadcast part, a shared
store-then-load under swizzling being the identity, a gather being warp local exactly when its
layout says so, a reduction's result carrying the slice of its operand's layout. These are the
classes level 1 is blind to, because the element still reaches memory once through some other lane,
or because a plain buffer makes two disagreeing swizzles cancel.

**Level 3: agents, shared memory and races.** The warps of a CTA are concurrent and shared memory
is where they meet. `ttsem/races.py` replays one execution and logs, for every shared-memory op and every
warp, the byte set that warp touches, computed from the layouts and not from an alias model, tagged
with the warp's epoch, the number of barriers its group has passed. Two accesses from different
warps, overlapping bytes, at least one a write, same epoch: a race. Asynchronous copies are pending
from the op that issues them to the op that completes them and race with anything that touches
their bytes in between. Across groups (`warp_specialize` partitions, the TMA engine) the ordering
goes through mbarrier phases and vector clocks. This is Membar's question, asked dynamically, on
kernels rather than on an abstraction of them.

## Limits

- **Inline PTX.** `tt.elementwise_inline_asm` is a deliberate rejection at the tile level, so it
  reads as "out of scope" rather than "not written yet". The LLVM continuation decides many of them
  and `llvm.inline_asm` templates it has no grammar for are the rest.
- **Warp-specialised TTGIR whose partitions wait on barriers** is rejected at level 1 by design:
  running the partitions in sequence would read shared buffers before the producer fills them.
  That is exactly the boundary between level 1 and level 3.
- **Clusters of more than one CTA** are declared unsupported rather than judged, at both levels.
- **`nvvm.ldmatrix` / `stmatrix`** in the m16n16 forms, and a small residue of NVVM intrinsics,
  are outside the LLVM continuation's fragment.
- **Everything below the LLVM conversion** is outside level 1; `ttsem/validate.py` stops at the first
  module containing `llvm.func` and reports "past lowering" once.
- **Device deviations are not gaps, they are findings**, and they stay mismatches. `SEMANTICS.md`
  lists them one by one: the `tt.clampf` NaN sign, decided by whether a peephole matched; a
  non-associative `tt.reduce` combiner, whose answer depends on the association the lowering picks;
  a left fold of narrow floats saturating where a balanced tree does not; `arith.remf` losing the
  sign of a zero remainder, because PTX has no `frem`; and the permutation of old values an atomic
  returns, which is the order the lanes arrive in.
- **A few ops raise `Unsupported` by name** rather than guess: `tt.mulhiui` at 64 bits, `e2m3` and
  `e3m2` microscaling formats, a `tt.atomic_poll` with no timeout, an im2col TMA copy, TMEM and
  warp-group MMA ops, and libdevice symbols outside the implemented table.

## Reading further

| file | what is in it |
|---|---|
| [LEDGER.md](LEDGER.md) | every upstream defect, with its reproducer, its dates and its fix |
| [docs/SEMANTICS.md](docs/SEMANTICS.md) | what each op is defined to mean, the float policy, and the device deviations |
| [docs/DESIGN.md](docs/DESIGN.md) | the three levels, the interfaces between them, and why they are drawn there |
| [docs/RESULTS.md](docs/RESULTS.md) | the corpora, the counts, and the calibration behind every number quoted above |
| [CONTRIBUTING.md](CONTRIBUTING.md) | how a new op, a new check or a new witness gets in |

## Citing

A report describing the semantics, the per-pass validator and the defects it found is in
preparation. Until it appears, cite this repository by URL and commit.

## Licence

MIT. See `LICENSE`.
