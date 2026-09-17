# SEMANTICS

Every choice the Triton and MLIR docs leave open, one line each: the rule this package picked
and why. The tie-breakers, in order: what the device does, then what
`triton/runtime/interpreter.py` does, then what LLVM does for the equivalent scalar op. Where
this package deliberately diverges from the shipped interpreter the line says so.

## What has been checked against a device

Seven of the eight fixture modules, parsed by `ttsem/mlir.py` and run by `Interp` on the recorded
pre-launch state, reproduce the recorded post-launch output **bit for bit**, with no
unsupported op: `p1`, `p2`, `p3` and `p4` in TTIR, and `p1`, `p2` and `p3` in TTGIR with
layouts erased. Between them that covers masked loads with `other`, masked stores, integer
division and remainder, truncation, `tt.reduce` and `tt.scan` combine regions, `tt.trans` and
`tt.sort` over rank-4 tensors, `scf.for` with three iter args, host descriptors clipping a
91-wide tensor into 16-element blocks, a bf16 `tt.dot`, and 3x2 and 5x2 grids.

The eighth, `p4.ttgir`, is rejected, and that rejection is the level-1 boundary rather than a
gap: its `ttg.warp_specialize` puts the TMA producer in a partition and the `tt.dot` consumer
in the default region, so running the regions in sequence would read the buffers before they
are filled. The barrier wait in the partition is what makes that detectable.

## Values

- Integers are stored **signed at their width** (`i8` -> `int8`, ... , `i1` -> `np.bool_`),
  because MLIR integer types are signless: the op, not the type, carries the signedness, so
  `arith.divsi` and `arith.divui` read the same `int32` differently.
- Pointers are `np.int64` byte addresses (the shipped interpreter uses `uint64`; `int64` is
  what `DESIGN.md` fixes, and the arithmetic wraps identically).
- `bf16` and the fp8 kinds are raw bit patterns in `uint16` / `uint8`, as in the shipped
  interpreter; `f16`, `f32`, `f64` are the native numpy floats.
- A scalar is a 0-d array, so every op is written once and works for both scalars and tensors.
  Every bit-level helper (`to_bits`, `from_bits`, `from_float`, `_unsigned`) goes through
  `values.viewable` rather than `np.ascontiguousarray`, which promotes a 0-d array to 1-d: a
  bf16 scalar add coming back with shape `(1,)` made the `tt.store` that consumed it fail to
  broadcast against a 0-d address, which is where 157 of the suite's errors came from.
- `index` is `int64`. Triton never emits it, but `arith.index_cast` is in the op list.
- A type that carries ordering and no data (`!ttg.async.token`, and anything of kind `other`
  whose spelling ends in `token`) is an opaque `int8` zero. The module the pipeliner produces
  has `!ttg.async.token` block arguments and `ub.poison` of that type, so a value of one has
  to exist; nothing reads it, because level 1 has already run every copy synchronously.
- A float literal MLIR printed in hexadecimal (`dense<0x7FC0> : tensor<...xbf16>`,
  `0xFF800000 : f32`) is a **bit pattern of the element type**, not a number: MLIR falls back
  to hex whenever the shortest decimal would not round-trip, which is every NaN and every
  infinity. The parser keeps it as `ir_types.FloatBits` and `arith.constant` writes the bits
  straight into the storage, so a NaN payload and the sign of an infinity survive. The quoted
  blob form `dense<"0x...">` is the raw little-endian data of every element, and a blob the
  size of one element is MLIR's shorthand for a splat.

## Integer arithmetic

- Every integer result wraps modulo `2**width`; `overflowFlags` (`nsw`/`nuw`) is read and
  ignored, because a wrapped result is a defined value and poison-on-overflow would reject
  programs the device runs.
- `divsi` and `remsi` truncate toward zero, computed as `(a - fmod(a, b)) // b` and
  `fmod(a, b)`, which is what LLVM and the shipped interpreter do.
- `arith.floordivsi` rounds toward minus infinity and `arith.ceildivsi` / `arith.ceildivui`
  toward plus infinity, so all three differ from `divsi` on every inexact division with a
  negative operand (`7 / -2` is `-3` truncated, `-4` floored and `-3` ceiled). numpy's `//`
  already floors, so the ceiling is the floor plus one whenever the remainder is not zero.
  Their poison cases are `divsi`'s. The `tritongpu-fuse-nested-loops` pass emits them.
- Division or remainder by zero raises `Poison`, and so does `divsi`/`remsi` of `INT_MIN` by
  `-1`: MLIR calls both undefined, and an exception makes the undefinedness visible instead of
  silently inventing a value. `Poison` is an exception, not a value, so the semantics never
  has to define what poison propagates into.
- A shift by an amount at or above the width gives `0`, or the sign for `shrsi`. LLVM calls
  this poison; the hardware gives 0 or the sign fill, and the fuzzer generates such shifts, so
  a defined answer is more useful than a rejection here.
- The shift amount is read **unsigned**, so a negative amount is a huge one and shifts out.
- `arith.extsi` of `i1` gives `-1` and `arith.extui` gives `1`; `arith.trunci` to `i1` keeps
  the **low bit**, not `!= 0`. This is LLVM's `trunc`/`sext`/`zext`, and it differs from a
  naive `astype(bool)`.
- An `i1` binary op wraps mod 2 like any other width: the operands are widened (`true` is `-1`
  to a signed op and `1` to an unsigned one), the op runs at that width, and the **low bit** is
  kept. numpy's own bool arithmetic is not that -- `np.add` is a logical or, so `True + True`
  would be `True`, and `np.subtract` refuses to run at all -- which is what
  `test_int1_bin_op_wraparound` and `test_sum_dtype`'s `int1` sum caught
  (triton-lang/triton#10919 is the same bug in the shipped interpreter).
- A signed `arith.cmpi` on `i1` reads `true` as `-1`, so `slt(true, false)` is true while
  `ult(true, false)` is false.
- `arith.fptosi` / `fptoui` truncate toward zero. A value out of the target's range, or a NaN,
  is left to numpy: MLIR calls it undefined and no fixture depends on it.
- `tt.mulhiui` computes in the doubled unsigned width; 64-bit inputs raise `Unsupported`
  because numpy has no 128-bit integer (the shipped interpreter falls back to a Python loop).

## Floating point

- A float op decodes its operands with `to_float` (identity for `f16`/`f32`/`f64`, widening to
  `float32` for `bf16` and fp8), computes in that dtype, and rounds **once** on the way back.
  So a `bf16` add is an `f32` add rounded to `bf16`, which is what the hardware does.
- `f32 -> bf16` uses **round-to-nearest-even** on the bit pattern. The shipped interpreter's
  `_convert_float` rounds half **up** and clamps the exponent instead; RTNE is what the
  hardware and the docs say, and the difference is invisible on the fixtures because their
  values are small integers.
- The fp8 kinds are decoded through a 256-entry table built from (exponent bits, mantissa
  bits, bias, family), and encoded by a nearest search over the finite values with ties to the
  **even code**. Families: `ieee` has infinities and NaN at the all-ones exponent, `fn` has no
  infinity and reserves the all-ones significand as NaN, `fnuz` has no infinity, no negative
  zero, and `0x80` alone as NaN.
- A finite value above the largest finite fp8 value **saturates**; only a real infinity becomes
  an infinity, and only in a kind that has one. This follows `_convert_float`'s exponent clamp.
- Encoding `-0.0` into an `ieee` or `fn` fp8 kind may give `+0`, because the search compares by
  value and the two zeros compare equal. No op distinguishes them at level 1.
- `f8E4M3B15` is treated as an `ieee` family (infinity and NaN at the all-ones exponent).
  Triton's `fp8e4b15` is a private format with no MLIR-side specification; nothing in the
  corpus uses it and this is a placeholder to be revisited if it ever appears.
- `maxnumf` / `minnumf` are `np.fmax` / `np.fmin` (a NaN operand loses); `maximumf` /
  `minimumf` are `np.maximum` / `np.minimum` (a NaN operand wins). This is the arith dialect's
  own distinction and matches the shipped interpreter.
- `arith.cmpf` implements the full 16-value `CmpFPredicate` table: the ordered predicates are
  `and`ed with "neither is NaN", the unordered ones are `or`ed with "one is NaN".
- `math.*` ops compute in the decoded dtype, so `f32` transcendentals are numpy's `f32` ones,
  as in the shipped interpreter. They do **not** model the device's fast approximations, so a
  program whose output depends on the last bit of `exp` will disagree with the device; the
  fuzzer avoids transcendentals for that reason.
- `math.erf` goes through `math.erf` per element (numpy has none), like the shipped
  interpreter's `np_erf_fp32`.
- `math.fma` is a **real** fused multiply-add: the product and the sum are formed exactly with
  `Fraction` and rounded once, and `float(Fraction)` goes through Python's correctly rounded
  integer division, so the single rounding, the subnormals and the overflow are right by
  construction. An unfused `a * b + c` in `float64` is not the same function: it rounds twice,
  and it overflows or underflows where the exact product does not. Two of the eight cases of
  `test_math_fma_op_special_values` show it -- `-DBL_MAX * DBL_MAX + inf` is `+inf` fused and
  `nan` unfused, and `-DBL_TRUE_MIN * 0.5 + 0.0` is `-0.0` fused and `+0.0` unfused.
- `tt.extern_elementwise` implements the common libdevice symbols by name, with `__nv_<name>`
  and `__nv_<name>f` generated from one table (`exp`, `log`, `sqrt`, `rsqrt`, the
  trigonometric and hyperbolic functions, `erf`, `rint`, `round`, `floor`, `ceil`, `trunc`,
  `fabs`, `fmin`, `fmax`, `fmod`, `copysign`, `hypot`, `pow`, `fdiv`, `fma`). Every other
  symbol raises `Unsupported` **with the symbol in the message**, so the gap is nameable. Like
  `math.*` these do not model the device's fast approximations.
- `math.absf` is `np.abs` on the decoded value, so a NaN's payload and sign are not preserved.
  The shipped interpreter masks the sign bit instead; only NaN payloads distinguish them.

## `tt.dot`

- Integer operands are widened to `int64`, multiplied and accumulated there, then wrapped into
  the result type. `int32` accumulation of `int8` inputs cannot round, so the wider intermediate
  changes nothing and removes an overflow question.
- Float operands are decoded and accumulated in `float32` (`float64` when the result is `f64`),
  then rounded into the result type.
- `bf16` operands are **decoded** to `float32`. The shipped interpreter multiplies the raw
  `uint16` bit patterns, which is a bug (the companion fuzzer's `mma.py` documents it and works
  around it in the generated programs). The device is the tie-breaker and `p4` matches.
- `inputPrecision = TF32` (0) masks the low 13 mantissa bits of `f32` operands before the
  multiply; `IEEE` (2) does not. The shipped interpreter never rounds, so it disagrees with the
  device on inputs that need more than 10 mantissa bits.
- `TF32x3` (1), `BF16x3` (3) and `BF16x6` (4) are treated as `IEEE`: they are error-compensated
  decompositions whose result is meant to be closer to `IEEE`, not further.
- `maxNumImpreciseAcc` is read and ignored: it bounds how many `fp8` products the hardware may
  accumulate in reduced precision, which is a hardware allowance, not a required behaviour.

## `tt.dot_scaled` and `ttg.fp4_to_fp`

The op says `d = matmul(scale(a, a_scale), scale(b, b_scale)) + c` and leaves the width of the
multiply open. The width is not a free choice: NVIDIA lowers the op by rewriting it, so the
steps of `DecomposeScaledBlocked::scaleArg` **are** the meaning the device gives it, and they
are what this semantics runs. Each operand is upcast to one *compute type*, scaled there, and
only then multiplied, with the accumulation of `tt.dot`.

- The compute type is `f16` when either `a_elem_type` or `b_elem_type` is `fp16`, and `bf16`
  otherwise (`getComputeType`). Every operand is rounded into it before anything else, so a
  `bf16` operand beside an `fp16` one loses the range `f16` does not have. That rounding is
  exact for every microscaling format on its own: `e2m1`, `e4m3` and `e5m2` carry at most four
  significand bits against bf16's eight, and a scale is a power of two.
- `e2m1` has no MLIR type. It travels two values per byte in an `i8` tensor, the **low nibble
  first**, and the nibble is a sign bit, two exponent bits with bias 1 and one mantissa bit:
  0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives, with the negative zero kept. `lhs_k_pack`
  and `rhs_k_pack` say which dimension the pair sits on: the K dimension when true (the
  default, and elided from the printed form when it is), the M dimension of the lhs or the N
  dimension of the rhs when false. That dimension is the one that doubles.
- `ttg.fp4_to_fp` is that unpacking on its own, along `axis`, and is what the decomposition
  leaves behind at TTGIR.
- The other formats decode as their own MLIR type when they have one. The op also admits an
  `i8` tensor for any format, and then `a_elem_type` is the authority and the bytes are
  reinterpreted; `e2m3` (2) and `e3m2` (3) are in the enum, no frontend produces them, and
  they raise `Unsupported`.
- A scale operand is `e8m0`: the byte **is** the exponent field of an f32, so 127 is 1.0 and
  the value is `2**(b - 127)`. Byte 0 is `+0.0`, not the `2**-127` of the microscaling
  specification, because the lowering builds the float by a shift (`scaleTo16`) and so does
  the shipped interpreter. A float-typed scale (the `f8E4M3FN` scales of nvfp4) decodes as
  itself.
- Byte 0xFF is a **NaN**, and it poisons its whole group: the shift makes it an infinity and
  `maskNan` then selects a NaN over the scaled value. `fastMath` skips that select, and then
  0xFF stays the infinity. The shipped interpreter has no `maskNan` step at all, so it returns
  the infinity in both cases; that is the one place these two disagree, and the device is the
  tie-breaker. `test_scaled_dot.py` pins both branches.
- One scale covers a **group** of consecutive K elements per row, and the group is the ratio
  of the shapes rather than a constant: `deduceScaleFactor` accepts 16 and 32. The lhs scale
  is `[..., M, K / group]`; the rhs scale is `[..., N, K / group]`, that is transposed with
  respect to its own operand, which `extendAndBroadcastScale` calls weird and does anyway.
- An absent scale operand is not a scale of 1 applied: it is no multiply at all, so the
  operand keeps the exact value the compute type holds.
- Accumulation is `tt.dot`'s: `float32` (`float64` for an `f64` result), plus `c`.

## Memory

- `tt.addptr` scales by `max(1, bits // 8)` of the pointee, so an `i1` pointer steps by a byte,
  as in the shipped interpreter.
- A masked load yields `other`, or `0` when `other` is absent. A masked-off lane never computes
  an address, so it never faults.
- A live lane whose address is outside every registered buffer raises `MemoryFault`. The device
  would fault too; making it an exception is what lets the harness report it rather than read
  someone else's memory.
- A store with two live lanes at the same address takes the **last** in row-major order. The
  hardware leaves this unspecified; row-major is the one order this package can name.
- Atomics apply one live lane at a time, in row-major order, so duplicate addresses compose in
  that order. This makes an atomic deterministic here, where the device leaves it unordered.
- An atomic on `bf16` or an fp8 kind passes `Memory.atomic` a `(decode, encode)` pair, because
  the storage of those types is a bit pattern and not the value: without it a `tl.atomic_add`
  of two bf16 numbers adds their `uint16` bit patterns, which is how `1.734 + 0.319` came out
  as `1.4e38` (`test_tensor_atomic_rmw` with `bfloat16`). `f16`, `f32` and `f64` need nothing,
  their storage *is* the value, and `cas` compares bits by definition and is left alone.
- `sem` and `scope` on an atomic are recorded and otherwise ignored: with one program running
  at a time there is nothing for a memory ordering to order.
- `Memory.atomic` takes the pair `(cmp, val)` in its `values` argument for `kind == "cas"`,
  which keeps the interface of `DESIGN.md` unchanged.
- A registered buffer must be C-contiguous, and `buffers()` hands back the very arrays that
  were registered, so a caller compares the same object it passed in.

## Descriptors

- A `!tt.tensordesc` is a base address, the global tensor's shape and element strides, and the
  block shape. A load clips to the shape and fills the outside with zero (or NaN when the
  descriptor says so); a store clips and drops the outside. This is what the docs promise and
  what `p4` confirms on a 91-wide tensor read in 16-element blocks.
- A host descriptor built with `round_f32_to_tf32=True` rounds every f32 a load brings in
  to tf32: nearest even at ten mantissa bits, Inf and NaN untouched, a carry out of the
  mantissa landing on the next exponent. That is `CU_TENSOR_MAP_DATA_TYPE_TFLOAT32` on the
  TMA unit and, word for word, the arithmetic `RewriteTensorDescriptorToPointer` inlines on
  targets without TMA; stores copy the bits. The flag reaches the semantics from the host
  object (`harness.descriptor_values`), since on TTGIR it lives only in the tensormap.
- `tt.reinterpret_tensor_descriptor` passes a `Descriptor` through and otherwise raises
  `Unsupported`: a device descriptor is an opaque 128-byte object and its shape is not
  recoverable from the pointer at level 1.

## Reductions and scans

- The combine region of `tt.reduce` / `tt.scan` is run once per index along the axis, on whole
  **slices** rather than element by element. Every op a combiner can contain is elementwise, so
  this is exact, and it is what makes a 128x64 reduction take microseconds instead of seconds.
- Combining associates left to right in increasing index order, with the accumulator as the
  left pair of block arguments and the incoming element as the right pair. Triton does not
  specify the order and the hardware's tree order can differ for a non-associative combiner;
  the fixtures use `+` and `max`, for which it does not matter. Float addition at a narrow
  width is not associative enough for that to stay invisible over a whole suite: see the
  device deviations at the end.
- A reverse `tt.scan` flips the input, scans, and flips back, as the shipped interpreter does.
- `tt.histogram` counts, in bin `i`, the live lanes **equal to** `i`; a value below zero or at
  or above the bin count belongs to no bin and is dropped. "Each bin has a width of 1 and bins
  start at 0" (`TT_HistogramOp`), so this is what the op says. It is a deliberate divergence
  from `triton/runtime/interpreter.py`, which uses `np.histogram`, whose last bin is closed on
  the right and therefore keeps a value equal to `bins`: the device drops it, and
  `test_histogram_out_of_range` and `test_histogram_silent_data_corruption` assert that it does.
- `tt.map_elementwise` applies its region to every element, or to every `pack` consecutive
  elements. With `pack = k` the block arguments are `k` consecutive elements of each operand,
  operand-major and pack-minor, and the region returns `k` elements of each result in the same
  order. A region that is one block of ops with no regions and no successors is run **once on
  whole columns** instead of once per element, which is exact because every op such a region
  can hold is elementwise; a region that branches is run per element.
- `tt.unsplat` is a one-element tensor read back as a scalar (a 0-d array); a wider source
  raises `Unsupported`.
- `tt.atomic_poll` is one relaxed load compared with `expected`. The op spins until every
  element matches or the shared timeout expires, and level 1 runs one program at a time, so
  nothing can change the flag between two iterations and the first load already decides.
  Without a timeout operand a poll that does not match would block forever waiting for a
  program this schedule will only run later: that raises `Unsupported` rather than inventing
  an answer, which is the level-1 boundary showing (`test_atomic_poll_waits_for_remote_cta`).
- The `tt.reduce` result is *not* re-encoded: the combine region already produced values in the
  storage encoding of the result type, and running them through `from_float` again would read
  the bit pattern as a number. A bf16 `256.0` is the `uint16` `0x4380`, which re-encoded
  becomes `0x4687`, i.e. `17280` (`test_sum_dtype`'s bf16 sum).

## Control flow

- Terminators raise a signal (`Yield`, `Condition`, `Branch`) that `Interp.run_region` catches,
  so the op table stays uniform and the block runner has no special cases.
- `scf.for` iterates `range(lb, ub, step)` on the operands as Python integers. A zero step
  raises `Poison`; MLIR requires a non-zero step and looping forever is the worse answer.
- `scf.if` with no else region and a true-only result yields nothing rather than failing, which
  only happens on malformed IR.
- `scf.while` runs the "before" region until its `scf.condition` is false and returns the values
  that `scf.condition` carried.
- `scf.index_switch` treats region 0 as the default and regions `1..n` as the `cases` attribute
  in order.
- `cf.cond_br` splits its operands with `operandSegmentSizes` when present and otherwise gives
  all of them to the taken branch, which is right for the one-successor-argument case.
- Names live in a stack of scopes, one per entered region, so a combine region's block
  arguments do not leak and a loop body rebinds its induction variable each iteration. A value
  defined in a dominating block stays visible, because the scope is not cleared on a branch.
- A multi-result op binds `%N#0 .. %N#k` when the parser gives it a single result name, and
  binds pairwise when the parser already numbered them. Both spellings appear in the corpus.
- `run_grid` visits program ids with `x` fastest, then `y`, then `z`, one program at a time.
  Any order is legal for a Triton launch; naming one makes the atomics deterministic.

## TritonGPU (level 1: layouts erased)

- `ttg.convert_layout` is the identity, and `Type.encoding` is carried but never read. That is
  the definition of level 1.
- A `!ttg.memdesc` is a plain numpy buffer. `memdesc_subview`, `memdesc_index`,
  `memdesc_reshape` and `memdesc_trans` return **aliasing views**, so a `local_store` through
  one is visible through the others, as shared memory is on the device.
- `ttg.async_copy_global_to_local` is a synchronous masked copy into the buffer;
  `async_commit_group`, `async_wait`, `async_bundle`, `local_dealloc`, `ttg.barrier` and
  `gpu.barrier` are no-ops, and the token they return is an opaque zero nobody reads.
- `ttg.warp_specialize` runs its default region, then each partition region in order, on the
  same memory. A partition that contains a barrier wait raises `Unsupported`, because running
  the partitions in sequence would deadlock a program that the device runs concurrently: this
  is exactly the boundary between level 1 and level 3.
- The `ttng` mbarrier ops (`init_barrier`, `inval_barrier`, `arrive_barrier`, `barrier_expect`,
  `wait_barrier`) and the fences (`fence_async_shared`, `async_tma_store_wait`) are no-ops in
  program order, and none of them touches the buffer it names: an mbarrier only orders the
  asynchronous copies against their consumers, and a copy that has already happened
  synchronously satisfies every barrier that follows it.
- `ttng.async_tma_copy_global_to_local` is a synchronous `tt.descriptor_load` landing in the
  destination `memdesc` buffer, with the same clipping and pad fill; the `pred` operand is
  honoured, so a false predicate leaves the buffer untouched. `ttng.async_tma_copy_local_to_global`
  is the matching synchronous store from the buffer.
- Both TMA copies read `operandSegmentSizes` rather than guessing positions, because the op has
  several variadic groups; a copy carrying im2col `offsets` raises `Unsupported`, since the
  gather it describes is not a plain block of the tensor.
- Everything else in `ttng` (warp-group MMA, TMEM, the cluster ops) is unregistered and raises
  `Unsupported` by name.
- `p4.ttgir` still does not run at level 1, but no longer for want of an op name: its
  `ttg.warp_specialize` puts the TMA producer in a partition and the `tt.dot` consumer in the
  default region, so running the regions in sequence would read buffers before they are filled.
  The barrier wait in the partition is what makes that detectable, and it is rejected.
- The `ttg.*_barrier` names an earlier draft registered do not exist in any dialect (TritonGPU
  has only `ttg.barrier`; the mbarrier family is `ttng`), so they were removed rather than left
  as registrations nothing can reach.

## Coverage

- An unregistered op name raises `Unsupported(name)` with the op text and is collected in
  `Interp.unsupported`, so a run reports what it could not model instead of silently skipping.
- `tt.elementwise_inline_asm` is registered as a deliberate rejection, so it reads as "out of
  scope" rather than "not written yet". The level-3 race detector is the one client that does
  not need the value: it binds the fragment's results to zero, counts them in `Report3.opaque`,
  and still finds the races around them (87 of the 469 `triton_kernels` modules carry one).
- `tt.fp_to_fp` implements round-to-nearest-even; an explicit round-toward-zero raises
  `Unsupported` rather than quietly rounding the other way.
- `tt.print` appends to `Interp.output` instead of writing to stdout, so a harness run over a
  corpus stays readable and the output is testable.
- `tt.assert` raises `AssertionError` with the op's message.

- `arith.divf` on f32: computed as the correctly rounded quotient. The NVIDIA lowering emits
  `div.full.f32`, within 2 ulp of it, so the harness reports a buffer that differs only by that
  much as `approx`, never as `match`; `tt.precise_divf` is the rounded one on both sides.
- `arith.remf`: C `fmod` (the sign of the dividend), which is what LLVM `frem` means.
- A `Poison` raised while running a launch (division by zero, `MIN / -1`, a zero loop step) is
  the verdict `poison`: the program is undefined by the semantics and the device value is not
  a reference for anything.
- `tt.call` runs the callee's body in place (the module before inlining still has them, and a
  `noinline` helper keeps its own `tt.func` all the way down: the kernel is then the function no
  `tt.call` targets, the root of the call graph, whatever suffix the frontend gave its name); `ub.poison`
  is a zero of its type, so a result that depends on it differs from the device and says so.
- `ttsem/validate.py` stops at the first module containing `llvm.func`: everything below the LLVM
  conversion is outside level 1 and is reported once as "past lowering".



## How a buffer is compared (`harness.compare`)

`match` means **bit for bit**, always, so the exactness picture stays visible: a launch that
reproduces the device exactly is never confused with one that only lands nearby. Everything
below decides whether a buffer that is *not* bit-identical is `approx` or `mismatch`.

- `harness.INEXACT_OPS` is the one table of ops whose device result a correctly rounded
  reference cannot be expected to reproduce, with a line per entry saying why, and a class:
  `"div"` for `arith.divf` (a bounded number of ulp) and `"wide"` for everything that
  reassociates or approximates (`tt.dot`, `tt.reduce`, `tt.scan`, float `tt.atomic_rmw`,
  `math.fma`, the transcendental `math.*`, `tt.extern_elementwise`). An op absent from the
  table is exact on both sides.
- `harness.scan_inexact` walks the module, including the regions, and keeps the classes it
  finds **on float operands** together with the float types those ops compute in. Nothing
  found means the buffers are compared bitwise, and only a NaN can still make them `approx`.
- Integer and boolean buffers are always bitwise. There is no such thing as an approximate
  integer, and pretending otherwise would hide the reduction-order and atomic-order findings
  below.
- Two NaNs are equal whatever their payloads and their signs, whether or not the launch has an
  inexact op. IEEE leaves the payload of a produced NaN unspecified, and the device
  canonicalises it in min/max and in conversions (`test_propagate_nan`: every launch of it
  differs from the semantics in exactly one NaN's bits). An infinity has to match exactly,
  which it does by falling outside every tolerance.
- A `"div"`-only launch is held to `APPROX_ULP` (2) ulp **of the narrowest float type any
  division computes in**, expressed as a relative bound. An ulp count is a relative bound, and
  taking it in the buffer's type would be the wrong yardstick when the result is widened on
  the way to memory: `test_bin_op` with `/` divides two integers in f32 and stores into an f64
  buffer, where one f32 ulp is 2**29 f64 ulp.
- Anything `"wide"` moves the launch to a relative band per element type -- f16 and bf16
  `rtol = atol = 1e-2`, f32 and f64 `rtol = atol = 1e-4`, the fp8 kinds one ulp of their own
  type (`2**-3` for e4m3, `2**-2` for e5m2) -- because no ulp count bounds a reassociated sum,
  and because a wide op's accumulation-order noise moves an f32 result that sits at an fp8
  rounding boundary to the neighbouring code: two elements in a million of a K=416 mxfp8 matmul
  (`triton_kernels/tests/test_matmul.py`), which no comparison of outputs can tell from a
  defect below one ulp of the type.
- The absolute tolerance is multiplied by the largest finite magnitude in the buffer. A
  reassociated sum is wrong by an amount proportional to the size of its *partial sums*, not
  to the size of the element that survives a cancellation, so a cancelled element has an
  unbounded relative error and a bounded absolute one, and the buffer's own dynamic range is
  the only proxy for the partial sums that a comparison of outputs can see. `test_dot3d` in
  f16 (a K=64 dot whose worst element cancels to 1.6 times its own value) and `test_scan2d`
  with `cumsum` in f32 are the two that need it.
- The diff record carries `max_ulp` (only for a buffer of a native numpy float dtype),
  `max_rel_err` and `max_abs_err`, so a `mismatch` says how far away it is.

## Device deviations

Cases where the device and the semantics disagree and the semantics is **not** changed to
follow, because the IR does not determine the answer or because the documented meaning is on
the semantics' side. Each is a launch of the upstream suite that survives the comparison
policy above.

- **`tt.clampf` means two different functions on NaN, depending on a peephole.**
  `test_propagate_nan[clamp-NONE-float16]` and `[clamp-NONE-float32]`. The generic lowering
  (`lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp`) emits `MinNumOp(MaxNumOp(x, lo),
  hi)`, which is what this semantics computes: with `x = NaN`, `lo = -0.2191162109375` and
  `hi = 0.2191162109375` that is `-0.2191162109375`. The NVIDIA backend recognises
  `clamp(x, -limit, limit)` and emits `min.xorsign.abs.f32(x, limit)` instead, whose sign is
  the xor of the two operands' signs, and the device returns `+0.2191162109375`. The same op
  with the same operands, one sign apart, decided by whether a pattern matched. The test only
  asserts `not isnan`, so both pass it.
- **A `tt.reduce` combine region that is not associative has no defined answer.**
  `test_argmax_argmin_with_nan` and `test_argmax_argmin_tie_break_fast_with_nan`: the argmax
  combiner keeps `(a, ia)` when `a > b` or (`a == b` and `ia < ib`), with *ordered* float
  predicates, so a NaN on the right wins and a NaN on the left loses. Over `[3, 5, nan, -inf]`
  the left fold this semantics runs ends at index 3, and the device's butterfly ends at index
  1, which is the index the test asserts. Both are legal evaluations of the same region:
  `tt.reduce` does not specify the association, and this combiner is not associative. The
  "argmax ignores NaN" property the test names holds only for the tree the lowering happens to
  pick.
- **A left fold of a narrow float saturates where a balanced tree does not.**
  `test_sum_dtype`'s last launch sums 1024 `bf16` ones with a `bf16` accumulator. bf16 has 8
  bits of significand, so `256 + 1` rounds back to `256` (a tie, to even) and the left fold
  this semantics runs stops at 256; the device's balanced tree doubles exactly at every level
  and reaches 1024, which is what the test asserts. Nothing in `tt.reduce` says which one it
  is, and the difference is a factor of four, not a rounding.
- **`arith.remf` loses the sign of a zero remainder.**
  `test_bin_op` with `%` and a `bfloat16` operand, 12 elements over four launches: where the
  dividend is a negative multiple of the divisor, C `fmod` and LLVM `frem` give `-0.0` (the
  sign of the dividend) and this semantics gives `-0.0`; the device gives `+0.0`, which is
  what the `x - trunc(x / y) * y` expansion of a remainder produces because PTX has no `frem`.
  The magnitudes agree everywhere, so only the bit pattern shows it.
- **The old value an atomic returns depends on the order the lanes arrive.**
  `test_atomic_rmw` (the `Old` buffer, e.g. `add-int32-all_neg`: the device returns `-11` and
  `-3` where the row-major order returns `-3` and `-7`) and `test_constexpr_if_return` (four
  programs race on a semaphore and the last one stores `program_id + prev`: the device stored
  `5`, the `x`-fastest order stores `6`). The set of updates is the same and the final value
  at the address agrees; only the permutation of the returned old values differs. `run_grid`
  and `Memory.atomic` name one order so that a run is reproducible, which is the most a
  single-program-at-a-time semantics can do.


## The cpu path's reference

On `--device cpu` the recorded post-launch state comes from the shipped interpreter
(`TRITON_INTERPRET=1`), not from a device. The interpreter computes bf16 arithmetic on the
bit patterns (`examples/witness_interp_bf16_add.py`: `g + g` for `g = bf16(-3)` is `0x8080`,
the 16-bit sum of two `0xC040`), so on the cpu path a bf16 or fp8 mismatch is the reference's
and not the compiler's; the calibration numbers of `RESULTS.md` are all from device records.


## Contraction

Triton compiles with `enable_fp_fusion` on by default, and LLVM's NVPTX backend then contracts
a float multiply feeding a float add or subtract into one `fma` with a single rounding. The
semantics rounds the product and the sum separately, as the IR reads. The difference is one
rounding of the compute type, and the float policy accounts for it: a module in which an
`arith.addf` or `arith.subf` takes an `arith.mulf` result is in the `contract` class and is
compared to a few ulp (`harness.APPROX_ULP`), like a division: relatively, or absolutely against
the buffer's largest magnitude (a fused `a * b + c` whose terms cancel leaves a result far
smaller than the rounding it absorbed), and in the widest of the compute type and the buffer's
own (one rounding of an f32 product still flips the last bit of the bf16 it is stored as). The
wide band of reductions and scans has an absolute part of a few hundred f32 ulp of the buffer's
magnitude for the same reason, a split scan's carry being the case that set it. The pattern is everywhere in the
kernels Inductor generates and almost nowhere in `test_core.py`, which is why it surfaced only
with the Inductor corpus.


## Buffers of atomic exchanges

A kernel that coordinates its programs through memory (Inductor's split scan and its decoupled
lookback: a dynamic block id from an atomic counter, then status words exchanged with
`atomic_xchg`) leaves in that workspace the trace of the order its programs arrived in. The
sequential replay fixes one order and the device another, so the words differ, and none of
them is a result of the kernel. `Memory` keeps the addresses that `xchg` and `cas` touched; a
buffer they landed in is compared, but its differences are reported `approx` with the note
"a buffer of atomic exchanges", the whole buffer, since which slot a program claims is itself
the schedule. The outputs of the same kernel stay under the ordinary policy.


## Two lanes, one address, two values

A store whose lanes carry the same address with different bytes is `poison`: the IR does not
say which lane lands last, and the hardware serialises them in an order nobody chose. The
replicated elements of a broadcast layout write the same bytes from every lane and pass. The
class was found one level down, by the companion LLVM-level interpreter, in the epilogue of a
`tt.reduce` whose
Welford combine leaves the 32 lanes of a warp with partials that differ in the last bit and
then stores them all to one word of shared memory; at TTGIR the reduction is a fold and the
epilogue is not visible, so here the rule guards `tt.store` and `local_store` of the program
itself, before and after every pass.

## Tensor memory and tcgen05 (Blackwell)

- `ttng.tmem_alloc` is a buffer like `ttg.local_alloc`: the `tensor_memory_encoding` (blockM,
  blockN, colStride) says how the hardware spreads the columns over lanes, not what the
  elements mean, so the semantics keeps the logical array and nothing else. `tmem_store`
  writes it when its predicate holds, `tmem_load` reads it (a `redOp` reduction result is
  unsupported), `tmem_subslice` is a view of `size` columns that aliases the buffer, and
  `tmem_copy` moves elements from shared memory when the counts agree; the blocked-scales
  layout, which duplicates every 32x128b chunk over four warps, changes the element count and
  is declined. Tokens are placeholders: level 1 runs in program order.
- `ttng.tc_gen5_mma` is `d = (useD ? d : 0) + a @ b`, done at once. Float operands accumulate
  in f32 (f64 for an f64 accumulator), f32 operands lose their low 13 mantissa bits first
  because the tensor core only multiplies tf32 (an `ieee` dot reaches the op already split into
  the three tf32 products of its emulation, so each MMA is exact on what it is given); integer
  operands accumulate in i32 and read as unsigned under `is_unsigned`. Every barrier operand
  whose predicate holds receives one arrival, which is what the hardware's commit does when
  the MMA lands; `tc_gen5_commit` is one arrival too, since every earlier MMA has already
  landed. `two_ctas` and `multicast` span the cluster and are unsupported.
- `ttng.tc_gen5_mma_scaled` is `d = (useD ? d : 0) + matmul(scale(a, a_scale), scale(b, b_scale))`
  with the scales read from tensor memory as the logical `[M, K / group]` and `[N, K / group]`
  arrays that `tt.dot_scaled` takes (a `tmem_alloc` from a register tensor keeps them; the
  blocked-scales `tmem_copy` path is declined). Decoding, the scale groups and the NaN rule are
  those of `tt.dot_scaled`; the products accumulate in f32.
- Checked against the CPU reference on three warp-specialized `desc_dot` programs compiled by
  `main` for sm_100 (arefs lowered to mbarriers, `tmem_alloc`/`store`/`load`, `tc_gen5_mma`,
  `tc_gen5_commit`): TTIR and final TTGIR both `match`.
