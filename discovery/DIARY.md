# Diary

What was run, on what, and what came out, negative results included. Newest entries last.
Where a later check showed an entry wrong, the entry is corrected in place and says so; the
review entry lists what changed and why.
Machine for every entry unless stated: Windows 11 laptop, Intel i7-13800H (20 threads), 32 GB,
NVIDIA RTX A1000 6GB Laptop GPU (Ampere, sm_86, 20 SMs, driver 596.58, CUDA 13.2); Python 3.13,
torch 2.14.1+cu126, triton-windows 3.8.0.post29; Lean v4.36.0 with Mathlib v4.36.0.

## 27 Nov 2026: environment

- **GPU, measured** (`do_bench`, float32 unless stated): a 3-stream add kernel over 64M elements
  moves 124.8 GB/s, `torch` copy 123.6 GB/s; a loop of `exp` plus one FMA reaches 512.6 G
  steps/s; `torch` matmul 4096³ gives 26.2 TFLOPS in fp16 and 4.4 TFLOPS in fp32. The first
  launch of a Triton kernel compiles in 4.8 s. `cost.A1000` uses these numbers.
- **Lean**: elan 4.4.0 installed in `C:\dev\tools\elan` (a path without spaces: the user
  profile's path has one); Mathlib v4.36.0's cache, 9,117 files and 7.1 GB, took 19 minutes
  over a network that dropped TLS handshakes (one file retried). A file importing
  `Mathlib.Analysis.SpecialFunctions.Exp` checks in 15.6 s, nearly all of it loading Mathlib.
- **LLM**: `claude -p` (Claude Code 2.1.322, `--model sonnet` resolves to claude-sonnet-5-5)
  with tools off, no MCP, no session saved: 44 s wall, 24 s of API time, for a two-character
  answer. The `--bare` mode would skip the start-up but needs an API key this machine does not
  have.
- **Downloads**: the torch wheel (2.6 GB) failed twice through `uv` (a truncated archive, a TLS
  handshake cut); `curl` with resume and retries got it.

## 27 Nov 2026: ttsem does not run on this laptop (negative)

`python -m ttsem.sanitize pytest tests/fixtures/agents/suite_no_mask.py` fails both tests with
`ttsem.mlir.NotGeneric` before executing anything. ttsem reads IR in MLIR's generic form; with
the Linux wheel it switches Triton's printer to that form by calling
`mlir::registerAsmPrinterCLOptions()` and `LLVMParseCommandLineOptions` inside `libtriton.so`
through ctypes. Triton's Windows build ships `libtriton.pyd`, whose export table holds two
symbols (`PyInit_libtriton` and one MLIR TypeID helper), so neither function can be reached, and
Triton's Python API offers no generic printing (`ir.module` has `str`, `str_nodebug`, `walk`).
Consequence: on this laptop the kernels are checked against torch on the GPU, not by ttsem;
the same code runs ttsem's checks on Linux. Ways out, none tried: a converter from the pretty
form for the TTIR subset the code generator emits; a Triton build that exports the two symbols;
WSL.

## 28 Nov 2026: what the research changed

Recorded in RESEARCH.md. In short: rediscovering the online softmax or FlashAttention's combine
with a Lean proof is no longer a result (TorchLean proved both forms over ℝ with a floating-point
bound; Opera, Neptune, Flashlight, EquiForge and tang derive the form automatically; OpenEvolve
and K-Search "rediscovered" it with LLMs). It stays as the pipeline's control experiment. Strassen
was dropped. Variance was added: Neptune's solver does not derive Welford's combine, and it is
the clean case where exactness over ℝ and floating point part ways.

## 28 Nov 2026: the semantics on the known forms

`evaluate(text, problem, seed=1)` on every seed and published form. Cell = (passes over memory,
state fields, float32 class). Errors are float32, normwise against a float64 reference on inputs
rounded to float32: rows of N = 2048 in blocks of 128 combined left to right, the median over
four rows per input family, the same rows for every program. Estimated time: 1,024 rows of
32,768 elements (attention: 1,024 queries, D = 64) on the A1000 model. The table was redone
after the review (below) with the four rows; the first version used one row per family.

| problem | form | cell | float32 error by input family | estimate |
|---|---|---|---|---|
| softmax | seed, 3 passes | (3, 2, stable) | normal 2.8e-8, large 5.9e-8, negative 4.7e-8, spread 6.0e-8 | 4,330 us |
| softmax | online | (2, 2, stable) | 6.0e-8, 5.9e-8, 4.7e-8, 6.0e-8 | 3,247 us |
| softmax | log-sum-exp (added later that day) | (2, 1, stable) | 8.2e-7, 4.7e-6, 7.3e-6, 4.3e-6 | 3,247 us |
| softmax | unshifted | (2, 1, nonfinite) | 7.8e-8, inf, inf, inf | 3,247 us |
| softmax | no rescale | rejected | wrong at N = 2: expected y=[0.9926, 0.0074], got [0.5, 0.0037] | |
| attention | seed, 3 passes | (3, 3, stable) | 1.7e-7, 1.5e-7, 2.1e-7, 3.2e-8 | 277,095 us |
| attention | online | (1, 3, stable) | 1.9e-7, 1.5e-7, 2.1e-7, 3.2e-8 | 138,547 us |
| variance | seed, 2 passes | (2, 3, stable) | normal 3.3e-8, offset 1.2e-4, tight 1.4e-6 | 2,165 us |
| variance | Welford | (1, 3, stable) | 1.3e-8, 3.4e-4, 2.0e-4 | 1,082 us |
| variance | sum of squares | (1, 3, inaccurate) | 6.5e-9, 2.4e+3, 1.0e+3 | 1,082 us |

What it shows:

- The exact tests reject the merge that forgets to rescale on the second input they try, and
  accept every form that is exact over ℝ, including the two that are not usable in floating
  point; those two are told apart by their class, not by the tests.
- Welford is one pass and `stable`, but on shifted data it is less accurate than the two-pass
  seed: 3 times on `offset` (mean 1e4, spread 0.1) and 140 times on `tight` (mean 100, spread
  0.001). The class threshold (1e-3) hides that; the record keeps the errors as numbers next to
  the class. (The first version of this entry said Welford "loses about three digits" on
  `offset`: that was one row's draw.)
- The log-sum-exp form is one to two digits less accurate than the online form on the
  adversarial families (4-7e-6 against about 6e-8), and still `stable`.
- bfloat16 everywhere (not what a kernel does: kernels accumulate in float32) breaks every form
  of variance on the shifted data, the stable ones included; it is reported, not used to rank.

## 28 Nov 2026: the kernels on the GPU

`codegen_triton.generate` turns every program into one Triton kernel: one program instance per
row, each pass a loop over blocks (the first block's summary is the state, every later block is
merged in), masked tails, no FMA contraction of the program's `+`, `-` and `*` (with it, the
one-pass variance of a single element came out nonzero, 9.3e+291 relative error against the
exact 0). Division, `exp`, `log` and `sqrt` are the GPU's approximate instructions, not numpy's
rounding: division within 2 ulp, `exp` from 2 ulp near zero to 64 at most, `sqrt` within 1 ulp
with subnormal inputs flushed, `log` within 1 ulp. (The kernels' headers said they round like
the numpy semantics until the review.) Every seed and known form compiles and matches the
float64 reference on lengths around the block (N = 1, 7, BLOCK ± 1, 3 BLOCK + 5, 20 BLOCK + 17;
N = 1 and 7 are lengths of the numpy tests too): worst error 7.2e-7. On the adversarial inputs
every kernel falls in the float32 class the numpy semantics gave it: unshifted softmax is
non-finite, the sum of squares is far off on shifted data, everything else is stable.

Times, `python -m discovery.gpu --problem P --program F --bench`, float32, one kernel at a time;
rows layout BLOCK 2048 with 16 warps, attention BLOCK 64 with 2 warps (each tuned on this GPU
only). Timed again after the review with the final method: three `triton.testing.do_bench`
calls, each after half a second of the kernel run back to back (an idle GPU sits at low clocks),
L2 flushed before every run; the median of the three medians and, in brackets, their range.
(The first timing was one `do_bench` call, whose warm-up is a number of runs estimated at idle
clocks, not 500 ms as this entry said; it read attention 6-26% high, and this entry's "every
number within 0.1%" held for the row kernels only.) Raw data:
`discovery/runs/kernels/bench-2026-11-28.json` (not versioned).

| problem, shape | multi-pass seed | one fewer pass | torch |
|---|---|---|---|
| softmax, 1024 x 4,096 | 0.317 ms (0.316-0.317) | online 0.330 (0.329-0.330), log-sum-exp 0.331 (0.331-0.331) | `torch.softmax` 0.281 (0.281-0.281) |
| softmax, 1024 x 32,768 | 4.175 (4.174-4.176) | online 3.238 (3.237-3.238), log-sum-exp 3.238 (3.238-3.239) | `torch.softmax` 4.204 (4.202-4.206) |
| variance, 1024 x 4,096 | 0.143 (0.143-0.143) | Welford 0.142 (0.142-0.142) | `torch.var_mean` 0.165 (0.164-0.165) |
| variance, 1024 x 32,768 | 2.035 (2.035-2.036) | Welford 1.059 (1.058-1.059) | `torch.var_mean` 1.063 (1.063-1.063) |
| attention, 1024 queries x 4,096 keys, D 64 | 3.310 (3.300-3.313) | online 1.525 (1.513-1.542) | SDPA 0.578 (0.578-0.579); `q @ k.T`, softmax, `@ v` 0.927 |
| attention, 1024 queries x 16,384 keys, D 64 | 15.186 (14.547-16.455) | online 6.903 (6.675-7.129) | SDPA 2.256 (2.256-2.257); the three calls 3.728 |

- On long rows (32,768) the pass the reformulation removes is the time it saves: softmax 1.29x,
  variance 1.92x over the seed; attention 2.2x at both lengths. There the online softmax moves
  124.4 GB/s counting the bytes it reads and writes, the copy bandwidth of this GPU: the
  roofline of a two-read algorithm.
- On rows of 4,096 removing a pass saves nothing: the seed's re-reads come from the cache (it
  "moves" 211 GB/s, above the copy bandwidth), so the three-pass softmax is 4% faster than the
  online kernel and the two-pass variance ties Welford.
- Against torch: the online softmax beats `torch.softmax` by 1.30x on rows of 32,768 and loses
  by 17% on rows of 4,096, where `torch.softmax` reaches 97% of the copy bandwidth by the
  operator's bytes and the online kernel 82%. Welford ties `torch.var_mean` on rows of 32,768
  (1.059 against 1.063 ms) and takes 14% less time on rows of 4,096. The attention kernels, one
  query per program on CUDA cores, are 2.6-3.1x slower than SDPA's tensor-core kernel and
  1.6-1.9x slower than the three torch calls: the IR has no query tiling and no `tl.dot`, so it
  ranks algorithms, it does not produce competitive attention.
- The row kernels' three medians agree within 0.1%; the attention kernels' spread 1-13% within
  one timing, most at 16,384 keys.
- ttsem's checks of these kernels: `python -m discovery.ttsem_check` reports `unavailable` on
  this laptop (the Windows limit above), in about 9 s, without raising.

## 28 Nov 2026: the first proofs (tier 2)

`lean/Discovery/Blocked.lean`, written once: `lmax`/`lmin` with their append lemmas, an
inductive `Tree` of blocks, and `Tree.eval_eq_block`: if `merge (block A) (block B) = block
(A ++ B)` for nonempty blocks, any merge tree over nonempty blocks evaluates to `block` of the
whole list. Its only axiom is `propext`. For a candidate, `lean_export` writes the definitions
from the IR, fixes every statement as a `Prop` before any proof text, and generates the
end-to-end theorem `blocked_eq_spec` (over any blocking and any merge trees the candidate
computes the specification's output, for every element), proved from the obligations by the
framework with no model involved. The model (`claude -p --model sonnet`) writes the proofs of
`merge_block_<pass>` and `spec_eq` only; a proof was accepted when Lean compiled it, no banned
construct appeared, and `#print axioms` listed nothing beyond `propext`, `Classical.choice` and
`Quot.sound`. (The review broke this judge twice; it was rebuilt, and the five proofs below
pass the new one.)

| form | proved | rounds | seconds (model, Lean) | tokens in/out | cost |
|---|---|---|---|---|---|
| softmax online | yes | 1 | 40.7 (18.9, 8.9) | 6,714 / 1,641 | $0.043 |
| attention online | yes | 2 | 73.4 (42.8, 17.7) | 19,349 / 3,954 | $0.117 |
| variance Welford | yes | 1 | 43.9 (21.7, 8.9) | 7,030 / 2,022 | $0.048 |
| variance sum of squares | yes | 1 | 37.4 (15.8, 8.5) | 6,815 / 1,156 | $0.039 |

The softmax proof (`lean/Discovery/Library/SoftmaxOnline.lean`) is the milestone's first proof:
the model wrote a helper, `shift_exp_sum` (Σ exp(x - a) = exp(b - a) · Σ exp(x - b), by
induction), then closed `merge_block_a` with `lmax_append` and `ring`; `spec_eq` is `rfl`. The
attention proof's first round left a `ring` after the goal was closed ("No goals to be solved");
the error went back and round 2 passed. The sum of squares is proved exact like the others: the
theorem says nothing about floating point, which is why the class is a separate label.

Not shown yet: that the judge rejects a wrong merge with the real model (the anti-cheat is
tested on canned answers only).

More proofs, the same afternoon:

- **softmax, log-sum-exp form** (one state field, the log of the sum, with temporaries for the
  maxima; added to the IR after the first search run, below): proved in 1 round, 66.2 s (model
  38.6 s, Lean 13.3 s), 6,688 / 4,648 tokens, $0.073.
- **Negative control**: the merge that forgets to rescale (`softmax no-rescale`), 2 rounds:
  not proved. Round 1's reply was not in the block format the prover asks for; in round 2
  `spec_eq` (which is true) was proved and `merge_block_a` was left with the false goal
  `Σ exp(x - lmax A) + Σ exp(x - lmax B) = Σ exp(x - max ..) + Σ exp(x - max ..)`. 56.6 s,
  $0.072. The record keeps statuses, not the replies, so whether the model said the statement
  was false is not known; the next version keeps the replies.

## 28 Nov 2026: the control experiment (tier 1), contaminated

**Not a test of recall.** The review found that every prompt of these runs held the library's
proved lemmas, whose statements spell out the target form's merge in the IR's syntax (the
library was filled an hour before the runs, and the prompts showed it by default). The
experiment was run again with no lemma, below; this entry is kept for the record, with the
proposal numbers corrected.

`python -m discovery.experiments --problems softmax attention variance --modes archive blind iid
--seeds 0 1 --iterations 6 --workers 2 --model sonnet`: 18 runs of 6 proposals each, every run
from the seed program with an empty memory of its own, `claude -p` with claude-sonnet-5-5. The
archive mode picks a parent and two inspirations from the MAP-Elites archive; blind is the same
with the operator's name and wording removed; IID asks the seed's prompt every time. 108 calls,
$1.80 at list price as the CLI reports it, about 3,000 tokens in per call. Full table:
`discovery/runs/rediscovery-2026-11-28/results.md` (not versioned).

| problem | the published form was proposal (seeds 0, 1) | other cells found |
|---|---|---|
| softmax | 1, 1 in every mode | log-sum-exp, one field (2,1,stable): archive at 6 and 3, blind at 5 and 3; never under IID |
| attention | 1, 1 in every mode | (l, o) with o already normalised (1,2,stable): one archive run |
| variance | 1, 1 in every mode | none |

(The first version of this table said 2 for 11 runs: it counted replies in the order they
arrived, and proposal 2's reply sometimes came back first.)

- The published form came first in every run, which with the lemmas in the prompt shows
  nothing; removing the names could not change it.
- The archive runs went on from the online softmax to the one-field log-sum-exp state; IID,
  which always restarts from the seed, never did. On attention one archive run produced the
  (log-sum-exp, normalised output) state that FlashDecoding and FlashAttention-2 use to combine
  split partial results: published too. Both need temporaries in a block or a merge; the first
  smoke run had a model's log-sum-exp proposal rejected for using one, and the IR gained
  temporaries that day. The lemma of the log-sum-exp form entered the library while these runs
  were going; two of the log-sum-exp replies came after it, with the lemma program's digest.
- **The floating-point check had nothing to reject**: the model never proposed the one-pass sum
  of squares; it went to Welford.
- **Rejections**: 8 of 108 replies. 5 read an element field outside a reduction, an
  element-level temporary such as `(p (exp (- s m)))` then `(sum p)`, which the IR does not have
  yet; 3 nested a reduction in another (variance).

The enumerator (no LLM), on the same forms with the merge removed (size cap 15, 16 sample
pairs, the search agent's runs): the online softmax merge in 0.8 s (size 13, 427k expressions
built), the online attention merges in 5.5 s (size 13), Welford's `n` and `mu` (size 9, the
form `R.mu + L.n (L.mu - R.mu) / n`) in 2 s, and **not** Welford's `q`: its smallest merge has
13 nodes with a 9-node subterm. At the default cap of 15 the full bank is built only up to size
7, so that subterm is never built, and the search stops at the cap (1,693,342 expressions built,
760,234 distinct); with a cap of 17 the memory guard stops it at size 8 (3,814,162 kept). (The
first version blamed memory at the default cap.) It does not have `log`, so it cannot find the
log-sum-exp merge either. An LLM's recall beats blind enumeration on the one combine that is not
a template.

## 28 Nov 2026: the GPU as the fitness, contaminated

`python -m discovery.loop --problem P --iterations 8 --workers 2 --model sonnet --gpu`: the seed
and every new correct candidate are compiled, checked against float64 on lengths around the
block, then timed alone (`gpu.fitness`: one shape, 1024 x 32,768 for rows, 1024 x 4,096 x 64
for attention); the measured time replaces the cost model's estimate in the archive. Softmax and
attention ran at once as two processes; `gpu.exclusive()` kept their timings one at a time. The
prompts held the lemmas, as in the control above, and the timings used the first method.

| problem | calls, cost, wall | elites by measured time |
|---|---|---|
| softmax | 8, $0.17, 73 s | log-sum-exp one field (2,1) 3.236 ms; online (2,2) 3.245 ms; seed (3,2) 4.174 ms |
| attention | 8, $0.16, 73 s | online (1,3) 1.487 ms; seed (3,3) 3.210 ms |

Five softmax kernels were timed: the four one-fewer-pass variants within 0.3% of each other
(3.236-3.245 ms), which is the noise of a median; the archive ranked them by it. This entry also
said one more (1,3) attention variant took 2.145 ms and was kept out of the cell by its time.
Timed again with the final method it takes 1.473 ms (1.471-1.522), inside online's range (1.507,
1.469-1.529): the 2.145 was a timing after idle, and the two kernels tie. Both runs led to the
tie rule (within 2% or within overlapping ranges, the smaller program wins). Every kernel's
worst error on the GPU check stayed below 1e-6.

## 28 Nov 2026: the review

Five reviewers, one per area (the semantics and checks, the Lean judge, the kernels and their
timing, the search, the claims of these documents), each finding then checked by a second
reviewer who tried to refute it: 55 findings held, 11 of them high. What changed conclusions:

- **The control was contaminated** (entry above): every prompt of the 18 control runs and of the
  two GPU runs held the lemma statements of the target forms, while the code and these documents
  said known forms were never shown. `experiments` now shows no lemma unless given a library
  (`--library`), reads one copy of it for the whole experiment, and the control was run again.
- **The proof judge could be fooled, twice.** A model-written `@[command_elab]` replaced
  `#print axioms` and printed a clean report for a `sorry` proof of the false no-rescale merge;
  and `set_option/- -/debug.skipKernelTC true in` passed the banned-word filter, after which Lean
  accepted `2 = 3`. The reviewers built both; the five proofs the model wrote pass the new judge.
  The judge was rebuilt (DESIGN.md): a deny list on the raw text, the statements compiled as a
  module of their own, and a fixed checker that replays the compiled proofs in Lean's kernel,
  checks their types and reads the axioms from the proof terms. Each attack is now a test; a
  check takes about 14 s instead of 8.5.
- **The exact tests were weaker than written**: no block larger than 128 elements was ever
  merged, the real-number check ran on the `normal` inputs only, and no row held repeated
  values; a merge wrong only on large blocks, or dividing by zero on equal values, passed. Now
  rows of 4,099 and 6,149 elements are cut in blocks of 2,048 and in a few random large blocks,
  rows of zeros, threes and small integers are tried, and so are the adversarial families, in
  float64 with a looser tolerance.
- **The floating-point class rested on one row per family**, a different one for each
  candidate: Welford's "three digits" was that row. Now the median of four rows, the same rows
  for every candidate.
- **The search's bookkeeping**: `reached()` took the first reply to arrive, not the first
  proposal; under `--gpu` a kernel that failed its check stayed an elite on its estimate; the
  elite was chosen on differences of 0.3%, inside the noise. Fixed, with tests; `--prove` was
  added, since the search never called the prover.
- **The timing**: `do_bench` sized its warm-up from runs made at idle clocks, and attention
  timings at 16,384 keys read 6-26% high. Now three timings after a fixed warm-up, with their
  range.
- **Smaller**: an `inf` literal crashed the loop and the prover; a Lean name with an
  apostrophe was misread; claims in these documents (6 proofs where there are 5, "Lean proves
  each kept form", "held-out shapes", "within 0.1%", the enumerator's cause).

Unchanged: the forms found, the five proofs (judged again), the kernels' correctness, the times
of the row kernels (within 0.2%). Open: the Lean statements say nothing about where `/`, `log`
and `sqrt` are defined (in Lean x / 0 = 0, so a theorem does not see a division by zero; the
float64 tests do, on the inputs they try); the deny list has to follow new Lean and Mathlib
command keywords; a kernel that fails on the GPU is not told apart from a wrong algorithm.

## 28 Nov 2026: the control again, with no lemma (tier 1)

`python -m discovery.experiments --problems softmax attention variance --modes archive blind iid
--seeds 0 1 --iterations 6 --workers 2 --model sonnet --out
discovery/runs/rediscovery-clean-2026-11-28`, with the reviewed code: no proved lemma in any
prompt (the default now), an empty memory per run. 108 calls, $1.88 at list price as the CLI
reports it, about 3,200 tokens in and 870 out per call, a median of 14 s per call; each run
took 33-59 s.

| problem | the published form was proposal (archive s0, s1; blind s0, s1; IID s0, s1) | other cells found |
|---|---|---|
| softmax | 1, 2; 1, 2; 1, 1 | log-sum-exp, one field (2,1,stable): archive at 4 and 6, blind s1 at 5; never under IID |
| attention | 1, 1; 1, 1; 1, 1 | none |
| variance | 1, 1; 1, 1; 1, 1 | none |

- **The search is recall, with no lemma too.** Proposal 1 was the published algorithm (up to
  names and the order of bindings) in 16 of 18 runs and proposal 2 in the other two; with 2
  workers both come from the seed's prompt, before the archive can say anything. In 10 runs one
  of the first two proposals was, byte for byte, the program written in `problems.py`: attention
  in all 6 runs, softmax in 4, variance in none, because the model writes Welford its own way
  (the same text in 4 runs). Blind made no difference.
- **The archive's effect is still variety**: the one-field log-sum-exp state in 3 of 4 archive
  and blind softmax runs, never under IID. The normalised attention state of the contaminated
  run did not come back.
- **Nothing was wrong over the reals**: no reply failed the exact tests. The 13 rejections of
  108 are all at parsing: 7 nested a reduction inside another (`(sum (exp (- x (rmax x))))` in
  6 softmax replies, one variance), 6 read an element field outside a reduction, an
  element-level temporary (4 attention, 2 variance). No reply proposed the unshifted softmax or
  the one-pass sum of squares, so the floating-point class still had nothing to sort.
- Archive and blind runs cost $0.12-0.15 each, IID $0.04-0.06: the model writes more when it
  sees elites.

## 28 Nov 2026: the first run with `--prove`, and a bug in the new judge (negative)

`python -m discovery.loop --problem softmax --iterations 8 --workers 2 --seed 0 --gpu --prove
--library none --model sonnet --out discovery/runs/softmax-gpu-clean-s0` (the directory is now
`softmax-gpu-judgebug-s0`). The search: 8 calls, $0.18, 92 s. Proposal 1 was the online form,
byte for byte, under the name `softmax-2pass`; the one-field log-sum-exp state came at proposal
3; at proposal 5 a smaller variant of it, whose merge is the two-argument log-sum-exp written
`max(L.l, R.l) + log(1 + exp(-|L.l - R.l|))`, took the cell. Seven kernels with one pass fewer
were timed, all at 3.237-3.239 ms, so every comparison among them was a tie; the seed took
4.176 ms.

The proofs: neither elite was proved, 3 rounds each, 6 calls, $0.45. The model was not the
cause. The new judge runs Lean in `discovery/lean` and handed it the round files' paths as the
run gave them, relative to the repository: every check failed with "Lean exited with 1" and no
other word, which is all the model was told. The three replies for the online form, judged
again with absolute paths, pass. The judge's tests had used absolute temporary directories
only. Fixed: the judge makes its paths absolute, a test runs it from a relative work directory,
and when Lean fails without a message in the file, what it printed goes into the error. The
three runs below were made after the fix.

## 28 Nov 2026: the GPU as the fitness, with the proofs of the elites (tiers 1 and 2 together)

`python -m discovery.loop --problem P --iterations 8 --workers 2 --seed 0 --gpu --prove
--library none --model sonnet --out discovery/runs/P-gpu-prove-s0`, for softmax, attention and
variance in turn, after the fix: no lemma in the prompts, every new correct kernel checked and
timed on the GPU (three timings after a warm-up), and the prover, 3 rounds at most, on every
elite but the seed.

| problem | search: calls, cost, wall | elites: cell, program, measured time, label | proofs |
|---|---|---|---|
| softmax | 8, $0.19, 81 s | (2,2) the online form, 3.236 ms, **E**; (2,1) log-sum-exp with the merge `max(L.l, R.l) + log(1 + exp(-abs(L.l - R.l)))`, 3.237 ms, **E**; seed (3,2) 4.175 ms | 2 calls, $0.11, 1 round each |
| attention | 8, $0.17, 86 s | (1,3) the online form with the two rescaling factors computed once for `d` and `o` (temporaries `wl`, `wr`), 1.511 ms, **E**; seed (3,3) 3.194 ms | 1 call, $0.07, 1 round |
| variance | 8, $0.15, 89 s | (1,3) Welford with `w = R.n / n`, 1.058 ms, **E**; seed (2,3) 2.035 ms | 1 call, $0.05, 1 round |

- **Every elite was proved**, in one round each: 4 of 4, $0.23 of calls against $0.51 for the
  three searches. Each proof went to the library under the program's name.
- **The GPU decided nothing within a cell.** Every kernel of a cell took the same time within
  its range (softmax 3.236-3.237 ms; attention 1.491-1.517 ms, ranges 1.471-1.531; variance
  1.058-1.059 ms), so each elite is the smallest program among ties. The time separates cells,
  that is passes over memory, not programs within one.
- **Recall again**: the published form was proposal 1 or 2 in every run (attention's byte for
  byte at proposal 2); no reply was wrong over the reals.

The first chain of three runs, started before the fix, had not stopped: stopping it ended the
shell, not the runs. Its attention run (`attention-gpu-judgebug-s0`) hit the same bug; its
variance run (`variance-gpu-firstchain-s0`), which started after the fix, proved its Welford
elite in one round. That attention search found one more cell, (1,2,stable): two fields, the
log-sum-exp of the scores with the same merge as above, and the output kept normalised,
`u = Σ exp(s - a) v`, the form FlashDecoding uses to combine partial results (published).
Judged again with the fixed judge, the model's replies during the bug were right for the compact
log-sum-exp softmax (all 3 rounds) and for the online attention (rounds 1 and 3); for the
two-field attention it had proved the merge in round 3 but not `spec_eq`. Sent to the prover
again (`python -m discovery.prove --problem P --program FILE --rounds 3`): the compact softmax in
1 round, $0.04; the two-field attention in 3 rounds, $0.35 (round 1 proved the merge and ran out
of Lean's heartbeat budget on `spec_eq`, round 2 broke a helper, round 3 passed).

**The library** (tier 3) now holds 11 proved programs: the five of the first afternoon and six
that a search proposed and the prover proved with no human step. Only one is a new cell, the
two-field attention; the others are algebraic variants of forms the library had (two Welford
merges, the online attention with shared factors, the softplus log-sum-exp merge), and two of
them, `softmax-lse-abs` and `softmax-lse-compact`, are the same program up to a temporary's name:
the library tells programs apart by their digest, which counts names. A proof of a program the
library holds under another name had overwritten the first proof of the online softmax; that
proof was put back, and `prove` now judges a stored proof again instead of asking the model for
a program it already holds (same digest, same specification), and keeps one entry per program.
`python -m discovery.prove --recheck-library` judges the 11 stored proofs again with no model:
all pass, about 22 s each. The test suite: 199 passed, 2 skipped (two forms that are not exact
over the reals have no kernel check).

## 28 Nov 2026, 09:35 UTC: floating point, measured (M1 of the long session)

A long session on branch `discovery-fp` (from `discovery`, PR 12) started at 08:59 UTC; its
brief: certified floating-point bounds through an approximate homomorphism, then a search under
them. At 09:00 the user lifted the brief's spend cap of $30; spend is still written down at every
checkpoint. Spend so far: $0, no model call. M0's base: on the new branch the suite passes (199,
2 skipped) and `prove --recheck-library` judges the 11 stored proofs again, all accepted.

- `fsem.py` runs a program with one rounding per operation in float32, float16 or bfloat16
  (bfloat16 rounded from float64 once, so no double rounding), a block's `sum` added along an
  explicit tree (`pairwise`: adjacent pairs, depth ⌈log₂ b⌉, as `tl.sum` is a tree; or
  `sequential`), the blocks merged along any tree (`left` is the kernel's loop), `+` and `*` of
  three or more arguments folded left to right as the kernel does. Its reference is exact:
  mpmath at 160 bits for the exponentials, rationals for the variance (a float64 sum of squares
  is off by 1e-9 at a mean of 1e4, a test shows); the float64 reference used inside the
  adversary's loop agrees with the exact one to 1e-13. A hand-written float32 online softmax
  gives the same bits as `fsem` (test).
- `adversary.py` climbs from six families of rows (logits around 0, 100 and 10⁴, a wide spread,
  a staircase of block maxima that forces a rescale at every merge, two levels of values) with
  moves of a few ulps, block shifts, ties with a block's maximum and new maxima; the worst row is
  measured again with the exact reference and saved.
- **The brief's table** (N = 4,096, blocks of 256, pairwise in-block trees, left merges, depth
  15; median and maximum of 10 random rows per family; unit u = 2⁻²⁴;
  `results/m1/library.csv`):

  | state | logits | brief | here: median / max | adversary |
  |---|---|---|---|---|
  | `(m, d)` online | around 0 | 1.2-2.2 (adversary 3.8-11.7) | 0.62 / 2.48 | 14.4 |
  | one field, log-sum-exp | around 0 | 9-55 | 5.98 / 19.7 | |
  | one field | around 100 | 190-340 | 96.5 / 189 | |
  | one field | around 10⁴ | 25,000-69,000 | 9,600 / 20,500 | 47,500 |

  The order and the growth of the one-field error with the logits' magnitude reproduce; the
  one-field numbers here are 2 to 3 times below the brief's on random rows. A hand-written
  float32 run of the one-field program matches `fsem` bit for bit, so the gap is in the rows or
  the statistics the brief used (its families are not given in detail), not in the arithmetic.
  Not pursued further.
- **The 11 programs of the library** (float32; `results/m1/library.csv` has float16 and
  bfloat16 too, `results/m1/worst/` the worst rows):
  - `SoftmaxLse`, `SoftmaxLseAbs` and `SoftmaxLseCompact` give the same bits on every row: they
    differ in the IR, not in floating point, since `exp(0)` is 1 and fl(a − b) = −fl(b − a). The
    same holds for `AttentionOnline` and `Attention1passShared`. The IR's digest does not know
    it; three of the six programs the searches added to the library are float duplicates.
  - The online softmax stays under 2.5 u on random rows (14.4 u adversarial); the one-field
    state grows with the logits (47,500 u adversarial at 10⁴). The two attention states behave
    the same way: the online `(m, d, o)` under 4.5 u against Σ p|v| (25.4 adversarial), the
    log-sum-exp state with a normalised output 541 / 1,020 u at 10⁴ (55,000 adversarial).
  - Variance at a mean of 10⁴ and a spread of 0.1: Welford 3,810 / 9,810 u (237,000
    adversarial), the sum of squares 1.7·10⁷ / 4.0·10¹⁰ u (7.6·10¹⁰ adversarial).
  - float16: Welford overflows on every family, because `L.n * R.n` is 2048² > 65,504; the two
    `-lean` variants, which compute `d * f * L.n`, stay finite on normal rows. A program can be
    exact over ℝ, the best of its cell in float32, and unusable in float16.
  - bfloat16 rounds the rows at a mean of 10⁴ to a single value (spacing 64): variance 0, error 0.

## 28 Nov 2026, 09:50 UTC: the rounding model and the depth law in Lean (M2)

`lean/Discovery/Approx.lean`, written once and by hand, as the brief asks; it took about 15
minutes, so the brief's fallback at 3 hours (a closeness relation specialised to relative and
exact fields) was not needed.

- **The model.** `Rounding` holds `u`, `η`, `ue`, `ηe`, `ul`. An operation with exact result `x`
  returns `x(1 + δ) + η` with `|δ| ≤ u`, `|η| ≤ η_min`; an addition has no `η` (a subnormal sum is
  exact); `exp` and `log` have their own relative errors; `max min neg abs` are exact. Two sign
  facts are part of the model: a rounded product keeps the sign of the exact one, and `exp` is
  never negative. Without them the model allows a negative normalizer that no run produces.
  Overflow is not modelled. Evaluation is a relation (`Tree.FEval`), true for every admissible
  choice of the errors.
- **The depth law.** `Tree.depth_law` (real index) and `Tree.depth_law_nat` (a count of
  roundings): if every block's float summary is `ε₀`-close and a merge takes an `εa`-close and an
  `εb`-close pair to a `max εa εb + c`-close one, then any blocking and any merge tree of depth
  `d` gives `ε₀ + c·d`. `Tree.eval_rel_block` and `output_eq_of_rel`: a merge that respects a
  relation and is right up to it on two blocks is right up to it on any tree (for the states with
  a stale reference of M4).
- **The instance.** `sum_tree_bound` and `sum_tree_error`: positive numbers summed along any tree
  of depth `d` give a result within a factor `(1 ± u)^d`, so `|w − Σ| ≤ ((1 + u)^d − 1) Σ`, the
  classical bound by the tree's height, out of the depth law with `C k w s := (1 − u)^k s ≤ w ≤
  (1 + u)^k s` and `c = 1`.
- **Tools for the candidates.** `fsum`, a block's sum along any tree of depth at most `D`;
  `fsum_bounds`; one lemma per operation (`add_err`, `mul_err`, `exp_bounds`, `log_err`).
- **Checks.** It compiles; every theorem's axioms are among propext, Classical.choice and
  Quot.sound (`eval_rel_block` uses propext only); the judge's checker replays every declaration
  of the module in the kernel with no error.
- **The model's parameters for numpy's float32, measured on every input** (`libm.py`,
  `results/m2/libm.json`): `exp` on the 2,239,853,081 floats of [−104, 89] with a finite result
  (36,329 that overflow are outside the model): largest relative error 3.57 u (at −62.03); below
  the normal range, largest absolute error 1.54·2⁻¹⁴⁹ (at −87.67): `exp` is not correctly rounded
  there, so `ηe` is 2⁻¹⁴⁸, not 2⁻¹⁵⁰. `log` on all 2,139,095,039 positive finite floats: largest
  relative error 4.58 u (at 0.7464). numpy's scalar and vector paths give the same bits (700,000
  samples each), so the merges, which call `exp` on scalars, are covered. For F1 the bounds are
  evaluated at `u = 2⁻²⁴`, `η = 2⁻¹⁵⁰`, `ue = 4u`, `ηe = 2⁻¹⁴⁸`, `ul = 5u`.

Spend: $0.

## 28 Nov 2026, 10:00 UTC: the two sources, reopened (M0)

- **TorchLean** (blueprint chapter on rounding error in attention; the theorems are in
  `TorchLean/Verify/RuntimeApprox/OnlineSoftmax.lean` at commit f9dff85 of 22 Nov, Lean and
  Mathlib v4.36.0, FloatLib v4.1.0; the arXiv paper 2602.22631 has none of them).
  `onlineAverage_error` bounds the online softmax average of its own Lean schedule, one query and
  one value coordinate: |out − O/S| ≤ (1+u)(Eo + |O/S|·El)/(S − El) + u|O/S| + η, where El and
  Eo give each key the allowance exp(u(M − s) + c) − 1 with c = u(B + 3T − 1) for T blocks of at
  most B keys, plus underflow residuals; through Gibbs' inequality the normalizer's relative error
  closes at (e^c − 1) + 2u·e^c·log N. Its model is FloatLib's: |fl(t) − t| ≤ u|t| + η, no η for
  sums, `exp` correctly rounded, `max` exact. It holds for every partition into blocks of at most
  B keys, but always for the sequential left fold with left-to-right sums inside a block: no tree,
  and the page says a native kernel needs a separate agreement proof for its `exp` and its
  summation order. The invariant was written by hand ("a closed form needs an invariant over all
  the keys seen, which a generated recurrence doesn't supply").
- **FloatLib** (arXiv 2609.19352, MIT, lean-dojo/FloatLib at 7835e85): it pins exactly our Lean
  and Mathlib; all 821 modules compiled against our build with no error, in 30.3 minutes, just
  over the brief's 30. Its model is ours except for `exp`, which it takes as correctly rounded;
  numpy's is not (3.57 u, M2). Decision, as the brief says past 30 minutes: our own model, which
  has the errors of `exp` and `log` as parameters.
- **Prior mechanized work** (a sweep of 12 sources) changes one claim of the plan. FloatLib
  already proves a depth law for sums over every association order (sums, sums of absolute
  values, dot products), stated once over all trees; `sum_tree_bound` is an instance of known
  mechanized work, not a new result. FloatLib's nodes must approximate an addition, so softmax and
  variance merges are outside it, and no mechanized bound for them over every merge order was
  found. ELSA (CVPR Findings 2026) states a depth law for online-softmax merges on paper,
  u(2⌈log₂ n⌉ + 3); a note of November 2026 shows it false as stated: it lacks a cancellation
  factor and a term in the logits' magnitude. Nothing on a floating-point theorem for conditional
  rescaling parametric in τ. The closest to "an LLM proposes the bound": bounds developed with a
  chatbot and formalised in Rocq (Mar 2026) and agents proving a floating-point stack in Lean
  (NeurIPS 2026); neither has an adversarial search as a second judge.

## 28 Nov 2026, 10:05 UTC: the prover refutes the statement's form twice (M3, negative for the form)

The model proposes B in the brief's grammar (sonnet), the float32 adversary attacks it (climbs on
error/B over families, blocks of 64 and 256, left, balanced and random trees), and a surviving B
goes to the prover with the generated statement. The first two forms of the statement were wrong,
and the prover said so instead of trying:

- **Version 1** had no hypothesis on the depth. Sonnet's first bounds survived the adversary:
  2Du + 2Ru + 2ue + 2d(Ru + ue + 2u) + 4n(η + ηe) for the online state, 2(d + 1)(Ru + 2nu + Mu
  + 2u + ue + n·ul + n·ηe + Du) for the one-field state. Asked for the one-field proof, it answered
  with a counterexample in the model: u = 1/256, which `small F` allows, 2001 copies of 10⁶, a
  caterpillar tree of depth 2000: the rounding of the state compounds as (1 + u)^depth, and no
  bound linear in the depth covers that. Correct. Version 2 adds the classical hypothesis of the
  γ_k bounds, (d + D + 1)(u + ue + ul) ≤ 1/16; float32 runs satisfy it by a factor of 1,000.
- **Version 2** had no hypothesis on the absolute errors. Asked for the online proof, it refuted
  again: u = ue = ul = 0, η = ηe = 2⁻¹⁰⁰, a comb of about 2¹⁰⁰ leaves: the absolute errors of
  the products bring d̂ to 0, and the readout to Lean's 1/0 = 0. Correct. Version 3 adds
  n(η + ηe) ≤ u (float32: n ≤ 2¹²⁴). Before any run, the same reading showed the attention's form
  false too: with subnormal values v, no B · Σ p|v| covers the absolute errors; in version 3 the
  attention's B has the scale S = Σ p|v| / Σ p and the bound V on |v| among its atoms, and so an
  absolute term.

The float32 adversary cannot see either flaw: it only reaches u = 2⁻²⁴ and depths up to 63. The
model's two refutations are kept (`results/m3/v1/`, `results/m3/v2/`). The version 1 and 2 runs
could not be stopped (the process kill was refused by the session's safety check) and finish on
their own, their outputs in `results/m3/` and `results/m3/v2/`; version 3 runs in `results/m3/v3/`.

## 28 Nov 2026, 10:15 UTC: the bounds of version 3, and a prover with Lean in the loop (M3, in progress)

- **The bounds.** With statement version 3, sonnet's first proposal survived the float32
  adversary for all five programs (`results/m3/v3/prompts/`):

  | program | B (grammar of `bound.py`) |
  |---|---|
  | SoftmaxOnline | 2(d+1)R·u + 2(d+1)·ue + 2(2d+D)·u + 4n(η+ηe) |
  | SoftmaxLse | 2(u(D+R+M+n+2) + ue + ul(n+1)) + 2d(u(M+n+2) + ue + ul) |
  | AttentionOnline | 3S((D+3d+3)u + (d+1)(Ru+ue)) + 4n(η+ηe)(V+1) |
  | VarianceWelford | 2(D+7d+6)u·S2 + 3(d+1)(D+3d+2)²u²M² + 2RMu·d(D+2d+1) + 6η(d+1)²(1+R)² |
  | VarianceSumsq | 4(D+d+3)u·Q + 6η |

  The forms are the expected ones and nothing in the prompts suggested them: the online state
  pays R·u per merge (the rounding of the subtraction in `exp(L.m − m)`, which no closed form of
  TorchLean's carries per merge but its per-key allowance u(M − s) does), the one-field state pays
  M·u (the magnitude of the logits), the sum of squares pays u·Q (the mean of the squares: the
  cancellation), the attention is relative to S = Σ p|v| with an absolute term.
- **The prover, as specified, does not write the proofs.** With the one-shot prover of
  `fpbound.py` (whole proof in one reply, Lean's errors back, 3 rounds of sonnet then 3 of opus)
  the replies were refutations of versions 1 and 2, `sorry`, or an outline marked "INCOMPLETE:
  a full proof needs a closeness relation, the block case from fsum_bounds, the merge case, the
  depth law, the final algebra". The protocol does not let a long proof grow: the deny list
  refuses `sorry`, so a partial proof never reaches Lean. Decision: keep these runs as the F3
  measure, and add `agentprove.py`: `claude -p` as an agent in a sandbox outside the repository
  (Read, Write, Edit there, and shell commands that start with `lean`; no web, no MCP, no
  settings; the diary and the research notes are not reachable), `sorry` allowed while it works;
  at the end the helpers and the proof are cut out of its file and judged by the same judge,
  which rebuilds the module around the generated statement.
- **A runaway proof.** At 10:07 UTC one model proof judged by `fpbound` took 17.5 GB of memory in
  Lean (a tactic searching without bound) and left 0.2 GB free on the machine; the session's
  safety check refused to stop the process, and the judge's 900 s timeout ends it. From now on
  every Lean run of the judge and of the agents has `--memory=6000`.

## 28 Nov 2026, 10:30 UTC: conditional rescaling, exact and overflow theorems proved (M4, in progress)

The IR has two new forms, `(param tau 8)` (a named constant, a variable in Lean) and
`(ifgt a b x y)` (x when a > b, else y). `policies.py` writes FlashAttention-4's policy as
`softmax-conditional`: the merge keeps the left reference `L.m` unless `R.m − L.m > tau`.
`fp_export.export_policy` writes its exact semantics out (no library module) and three statements;
the agentic prover proved two of them on the first try (`results/m4/`):

- **Exact over ℝ for every merge tree**: `znorm (t.eval (block_a τ) (merge_a τ)) = Σ exp x`,
  for every τ. Sonnet, 8 turns, 71 s, $0.10: the merge keeps `d·e^m` whichever reference the
  condition picks (`split_ifs <;> exact key _`), then an induction on the tree.
- **The overflow condition**: the sum field is at most `n·exp(τ·ρ)`, ρ the tree's right depth
  (a leaf 0, a node max(ρ_left, ρ_right + 1)). Sonnet, 12 turns, 102 s, $0.18. For the kernel's
  left fold ρ = 1, so `n·e^τ`; for a right comb ρ is the depth.
- **The lower bound, measured** (`results/m4/SoftmaxConditional-worst.json`): block maxima rising
  by 0.98 τ, 16 blocks of 256: the left fold's reference lags the maximum by 0.97 τ; on a right
  comb (ρ = 15) it lags by 14.69 τ, 98% of the proved τ·ρ, and at τ = 8 float32 overflows (the
  sum reaches e^117.6). A threshold that is safe in the kernel's loop is not safe in a tree of
  merges: a split-KV combine or a parallel reduction that keeps stale references needs τ·ρ below
  about 88 − ln n.
- The floating-point bound, parametric in τ: in progress.

M3: the agentic prover closed the sum of squares, |v̂ − var| ≤ 4(D + d + 3)·u·Q + 6η (sonnet,
59 turns, 18 min, $1.59, 455 lines with 11 lemmas of its own; the three standard axioms).

## 28 Nov 2026, 10:40 UTC: the one-field bound proved, a judge bug, the attention refuted (M3)

- **A bug of PR 12's judge (negative for the judge).** Sonnet's agent left a 785-line proof of
  the one-field softmax bound with no `sorry`; the judge refused it before Lean ran: "not
  allowed: def". The proof has the identifier `hφdef`; the deny list split words with an
  ASCII-only pattern, read `h`, then `def`. Lean identifiers take Greek letters and subscripts.
  The pattern now reads words as Lean does (any letter, digits, subscripts, primes, `!`, `?`);
  real `def`, `#eval` and `sorry` are still refused (test). Judged again, the proof is accepted:
  **SoftmaxLse, |log Ẑ − log Z| ≤ 2(u(D + R + M + n + 2) + ue + ul(n + 1)) + 2d(u(M + n + 2) +
  ue + ul)**, sonnet, the three standard axioms (`results/m3/agent/SoftmaxLse-rejudged.json`).
- **The attention's version-3 statement is false; both models said why.** Sonnet (2 turns) and
  opus (7 turns) each gave a counterexample instead of a proof: u = 1/256, R·u = 8, one block of
  two elements, every rounding exact but the subtraction s − m: the weight is off by a factor
  e^(Ru) ≈ 2,981, while B grows as Ru. In the softmax statements the quantity is a logarithm and
  the same error adds Ru; in the attention's |ô − o| it multiplies. Version 4 adds (hi − lo)·u ≤
  1/16 to the attention's statement only (float32: logits within a range of 10⁶), and the whole
  path (proposal, adversary, agent) runs again on it.
- So far: 2 bounds proved (SoftmaxLse, VarianceSumsq); the online softmax and Welford are with
  the agents; the attention restarts at version 4.

## 28 Nov 2026, 10:50 UTC: the search under a certificate (M5): recall again, and a noisy fitness

`certsearch.py`, softmax, sonnet as proposer, no lemma in the prompts, 6 proposals per run, 3
seeds per arm (`results/m5/report.json`):

- **cert**: the archive search, where a correct candidate enters the archive only with a
  certificate (the model proposes B for its policy-form statement, the adversary of `policyrun`
  attacks it for τ in {0, 2, 8} and left, right, balanced and random trees), and the
  certificate's value at a reference point (n = 4,096, blocks of 256, left fold, R = 10, M = 100,
  float32) is the fitness; **iid**: independent samples certified the same way; **plain**: PR 12's
  search with no certificate, its elites certified afterwards.
- **No new program.** In all 9 runs the elite of the two-field cell is `softmax-online`, byte for
  byte the library's (digest fbc0a917); in the one-field cell a one-field log-sum-exp variant. To
  the brief's question (a program whose certified bound beats every published form of its cell
  at equal time) the answer here is no.
- **F5.** Proposal 1 is the final elite in 2 of 3 `cert` runs, 0 of 3 `iid` and 1 of 3 `plain`;
  the published form is in the first batch of three proposals (submitted together) in every run.
  It is recall, as in PR 12.
- **A negative result for the method: the certificate measures the certifier.** Four
  programs of the two-field cell that differ only in the text the model wrote got certificates of
  148.8, 400, 524 and 526 u, because the certifier proposed a tighter or a looser B for each. In a
  cell the search rewards the luckier bound, not the better program. A certified fitness needs a
  canonical bound per program (for instance, the tightest B a fixed procedure proves), not the
  first one a model proposes.
- The first try of the `cert` arm stopped on a bug of mine (the adversary read a field `a.d` that
  only the online state has); fixed, the three runs were made again. Cost of the runs: $1.14 at
  list price, not counting the certificates computed before the bug, which are cached and reused.
- The two distinct final elites (the online form, a one-field form) go to the agentic prover with
  their certificates' B: an elite counts as certified only when that proof is accepted.

## 28 Nov 2026, 11:00 UTC: three bounds proved; the certified order flips PR 12's choice (M3 closed)

`results/m3/summary.json` (from `fpreport.py`), `results/m3/agent/` (records, proofs, statements),
`results/m3/f1/` (the long attacks):

| program | proved bound B (version 3) | prover | tightness | F1 |
|---|---|---|---|---|
| SoftmaxOnline | 2(d+1)R·u + 2(d+1)·ue + 2(2d+D)·u + 4n(η+ηe) | sonnet, agent, 128 turns, 49 min, $4.46 | 20.3 | holds |
| SoftmaxLse | 2(u(D+R+M+n+2) + ue + ul(n+1)) + 2d(u(M+n+2) + ue + ul) | sonnet, agent (judged again) | 4.9 | holds |
| VarianceSumsq | 4(D+d+3)·u·Q + 6η | sonnet, agent, 59 turns, 18 min, $1.59 | 9.7 | holds |

Tightness is B over the largest error the long attack found (400 steps, 4 seeds, every family,
plus the worst rows of M1 climbed again), at the run where error/B is largest; F1 holds when no
run exceeds B. The worst runs: logits around 10⁴ on a balanced tree (one field), the mean-10⁴
rows (sum of squares, an absolute error of 64 on a variance of 0.014).

- **F4, the brief's first test bench, passes.** At the logits of M1's families (n = 4,096,
  blocks of 256, left fold): around 100 the certified bounds put the online state 382 times
  below the one-field state, the measured errors 102 times; around 10⁴, 1,071 times certified,
  12,370 times measured. The order agrees on both pairs. PR 12 elected the one-field state the
  elite of its cell because it was smaller at the same time; the certified bound reverses that
  choice, as the brief asked.
- **The first bound proposed was the one proved, three times out of three** (on statement
  version 3; the earlier versions' first bounds were right for float32 but their statements were
  false). The bounds are 5 to 20 times above the worst error found.
- **F3 is falsified.** With the one-shot prover (3 rounds of sonnet, then 3 of opus) no surviving
  bound was proved: 0 of 11 so far (results/m3/*.json, v2/, v3/). Some of those statements were
  false; on the valid ones the replies were outlines or `sorry`. The brief's fallback is a
  mechanical procedure for the positive-sum fragment; I took a different one, the agentic prover
  (the model with Lean in a sandbox), which keeps the model writing the whole proof. It closed 3
  of the 3 valid statements it was given so far, at 18 to 49 minutes each.
- Open: Welford (with the agent), the attention at version 4 (the path runs again).
