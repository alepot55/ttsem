# Discovery: design

A system that searches for reformulations of GPU operators, proves each one equivalent to the
operator it replaces, and measures whether it is faster. Three tiers, each usable alone:

1. **Search.** An LLM proposes variants of a program; checks reject the wrong ones; a cost model
   or the time measured on the GPU is the fitness; the survivors go into a MAP-Elites archive.
2. **Proof.** A variant is a program in a small IR with a real-number semantics. Its correctness
   becomes a Lean 4 theorem; the LLM writes the proof and Lean checks it. Exact equivalence over
   ℝ and behaviour in floating point are two separate labels.
3. **Self-improvement.** Proved lemmas and recorded failures are fed back to the next run.

This file says what is built and why. [RESEARCH.md](RESEARCH.md) has the sources behind each
decision, [DIARY.md](DIARY.md) what was run and what came out, negative results included.

## What the research changed

The literature of 2024-26 already contains every piece of the first milestone as it was first
written (sources in RESEARCH.md):

- deriving the online softmax and FlashAttention's rescaled combine automatically from the
  multi-pass form: Opera (PLDI 2024), Neptune (PLDI 2026), Flashlight, EquiForge, tang;
- a Lean 4 proof that the blocked online form equals the two-pass one over ℝ, for every
  blocking, with a floating-point error bound: TorchLean (the bound written by hand);
- LLM-driven evolution that "rediscovers" the online softmax: OpenEvolve's 2025 Metal kernels,
  K-Search; every model has seen FlashAttention and Triton's fused-attention tutorial.

So the rediscovery is run as a **control experiment** for the pipeline, with the contamination
stated, and is not claimed as a result. Strassen on small blocks is dropped: on GPUs it pays only
for large matrices (SubCuber, 2026) and its Lean proof is one tactic (`noncomm_ring`).

What is still open, and what the system is shaped for:

- finding the state and the associative combine of a reduction **outside the schemas compilers
  hard-code** (an invertible repair function, a factorizable update). Neptune's solver, for
  example, does not derive Welford's mean-and-variance combine;
- producing, for each discovered form, both the **exact proof** and a **floating-point
  classification**, automatically. Two programs equal over ℝ can differ completely in floating
  point: one-pass variance from `(n, Σx, Σx²)` is exact over ℝ and loses every digit on data with
  a large mean; Welford's `(n, mean, M2)` is exact and stable.

## The IR: blocked folds

A candidate is an S-expression program: a sequence of **passes** over a stream of elements, each
pass a fold with a **state**, a **block summary** (what one contiguous block of elements reduces
to) and a **merge** (how two adjacent summaries combine), followed by an **output** stage.

```lisp
(program softmax-online
  (input x)
  (pass a
    (state m d)
    (block (m (rmax x)) (d (sum (exp (- x m)))))
    (merge (m (max L.m R.m))
           (d (+ (* L.d (exp (- L.m m))) (* R.d (exp (- R.m m)))))))
  (output elementwise (/ (exp (- x a.m)) a.d)))
```

The same program has three readings:

| reading | used for | where |
|---|---|---|
| real numbers | the Lean theorem | `lean_export.py` |
| numpy at a chosen dtype, over any blocking and any merge tree | tests, the floating-point class | `semantics.py` |
| a Triton kernel, one program instance per row, a loop over blocks | ttsem, the GPU timing | `codegen_triton.py` |

The kernel does not round exactly as the numpy semantics does: `+`, `-` and `*` are rounded to
float32 one by one (no FMA contraction), but its division, `exp`, `log` and `sqrt` are the GPU's
approximate instructions (division within 2 ulp, `exp` from 2 ulp near zero to 64 at most), so
a kernel's errors are measured on the GPU, not taken from the semantics.

The fitness descriptors come from the IR, not from the kernel text: the number of passes over
memory, the size of the state, the numeric class. A kernel is an instance of a plan, so a slow
or broken kernel does not lose a good algorithm (the split K-Search argues for).

## The checks, in order

Every candidate of the search goes through the first four, each cheaper than the next and run
only on what the previous one kept:

1. **parse and type** the program;
2. **exact tests**, in float64 against the problem's reference: short rows (1 to 257 elements)
   over many random blockings (one block, one per element, random cuts) and random merge trees;
   rows of 4,099 and 6,149 elements in blocks of 2,048 (the kernel's) and in a few random large
   blocks; rows of repeated values (all 0, all 3, small integers); and the floating-point
   families' inputs, where float64 itself may lose digits on an ill-conditioned problem (a
   looser tolerance there). Blockings are sampled: a merge that is wrong only on blockings never
   tried can pass; only a proof covers them all. The stage also records whether every merge is a
   homomorphism (`merge (block A) (block B) = block (A ++ B)` on random splits): a program right
   end to end without it cannot be proved by the proof tier;
3. **floating-point tests**: float32, float16 and bfloat16 (emulated) on four rows of each
   input family (ordinary and adversarial: large logits, large mean with a small spread), the
   same rows for every candidate, error against float64, median over the rows. The float32
   errors give the class: `stable`, `inaccurate` (finite, above 1e-3 on some family) or
   `nonfinite` (an inf or a NaN, overflow or 0/0);
4. **cost model**: bytes moved and operations per element against constants measured once on
   the RTX A1000 (`cost.A1000`); on these problems the bytes decide it.

Two more stages are opt-in, because they are slower:

5. **kernel** (`loop --gpu`): the Triton instance is checked on the GPU against float64 on
   lengths around the kernel's block (1, 7, the block ± 1, longer rows), then timed alone (the
   GPU is the clock): three `do_bench` calls, each after half a second of warm-up, their median
   and their range. The median becomes the fitness, and a candidate whose kernel fails is taken
   out of the elites. `ttsem_check.py` runs every launch of the instance under `ttsem.sanitize`
   (out-of-bounds accesses, races, undefined values) where ttsem runs, on Linux;
6. **proof** (`prove.py`, or `loop --prove` on a run's elites): the Lean obligations of the
   candidate, proved by the LLM and checked by Lean. A proved elite is labelled `E`, one that
   passed the tests only `T`.

The checks, the specification and the shapes the kernels are checked on are outside anything
the proposer can edit.

## The proof tier

`lean/Discovery/Blocked.lean` is written once: if `merge (block A) (block B) = block (A ++ B)` for
nonempty `A` and `B`, folding `merge` over the summaries of any nonempty blocking of `X` gives
`block X`. For each candidate, `lean_export.py` generates:

- per pass, the theorem `merge_block` (the only place the merge's algebra lives);
- `spec_eq`: the candidate's output on one block equals the specification's.

The framework then proves, with no model involved, `blocked_eq_spec`: over every merge tree of
nonempty blocks, the candidate computes the specification's output.

The LLM returns proof bodies and helper lemmas. The judge (`prove.py`'s docstring has the
details) does not believe anything Lean prints while it elaborates the model's text, because
that text can change how Lean elaborates (a review of this prototype forged a clean
`#print axioms` report that way, and turned the kernel's type check off with a `set_option`):

1. the model's raw text must pass a deny list: in the helpers, theorem and lemma declarations
   only; in a proof, tactics only; no attribute, no `#` command, no other command or modifier,
   no `sorry`, no native evaluation, `set_option` only for `maxHeartbeats`;
2. the statements are compiled first, from the generated text alone, as a module of their own;
   the model's proofs go into a second module that imports it;
3. a fixed checker (`lean/Discovery/Checker.lean`, run with `lean --run`, importing only Lean's
   core) loads the compiled proofs with extensions off, replays every declaration in Lean's
   kernel, checks that each `X_holds` has exactly the type `X_statement` of the first module,
   and collects the axioms from the proof terms: nothing beyond `propext`, `Classical.choice`
   and `Quot.sound`.

`python -m discovery.prove --recheck-library` judges every stored proof again with no model.

The theorems are about Lean's real numbers, where division is total (x / 0 = 0) and `Real.log`
and `Real.sqrt` are defined everywhere. A program that divides by zero on some input is
therefore not refuted by its theorem; the float64 tests reject it on the inputs they try, and
definedness obligations are left for later.

Floating point is not proved in this first version. It is measured (stage 3), and the design
leaves room for a bound proved over an abstract rounding model (FloatLib's, which TorchLean
uses) as a third label.

## The search tier

A minimal loop of our own, not a framework: the research found that archives, islands and
ensembles give no stable gain over simple sampling (Simple Baselines, 2026), and the parts that
matter here (the IR, the checks, a GPU that must time one thing at a time) are ours anyway.

- **Proposer.** `claude -p` with tools off (any model the account allows), or any
  OpenAI-compatible endpoint (a local model on the RTX 4070). The prompt holds the IR's grammar,
  the problem, the parent and two other elites with their scores, the failures so far and, when
  a library is given, the proved lemmas; the answer is a whole program.
- **Archive.** MAP-Elites over (passes, state size, numeric class); fitness is the measured time
  with `--gpu`, the cost model otherwise. Two times closer than 2%, or whose ranges of repeated
  timings overlap, count as a tie, broken by the smaller program: on this laptop a row kernel's
  timings repeat within 0.1%, the attention kernel's within a few percent at 4,096 keys and
  only within about 15% at 16,384.
- **Baselines**, run with the same budget: independent samples from the seed with the same prompt
  (no archive), and a bottom-up enumerator of merge expressions with observational equivalence
  (no LLM), given the state.
- **Contamination.** A proved lemma states a known form's merge, so the control runs use no
  library. A blind mode replaces the operator's name and the programs' names; the field names
  (`m`, `mean`, `var`) are shown as they are. It cannot hide a structure the model knows; the
  diary says which runs were blind.

## Self-improvement

Every proved form goes into `lean/Discovery/Library/` with its statements in an index that the
proposer's prompt includes when a library is given; a program the library already holds (the
same digest, against the same specification) is not sent to the model again: its stored proof is
judged again. Every rejected candidate leaves its reason and counterexample in the run log, and
the proposer sees the last ones. Whether this helps has to be measured at equal
compute (an EACL 2026 study found that library learning's gains often vanish then); this
version records, it does not claim a gain.

## Hardware and cost

| | this laptop | target |
|---|---|---|
| GPU | RTX A1000 6 GB (sm_86): 124 GB/s copy, 26 TFLOPS fp16 matmul, measured | RTX 4070 12 GB |
| OS | Windows 11, no WSL | Linux, where ttsem's checks run |
| LLM | `claude -p`, a median of about 13 s per call in the runs (44 s for a first, cold one) | the same, or a local 7-8B model |
| Lean | v4.36.0, Mathlib v4.36.0; the judge takes about 14 s per proof attempt, 22 s for a candidate's first | the same |

ttsem's checks do not run on this laptop: Triton's Windows build does not export the two MLIR
symbols ttsem uses to print IR in the generic form. The kernels generated here are checked
against torch on the GPU, which sees wrong values but not an out-of-bounds read that lands in
padding; on Linux the same instances go through ttsem. Triton 3.8's FpSan
(`TRITON_INSTRUMENTATION_MODE=fpsan`) is a candidate cheap algebraic gate on the GPU, not
evaluated yet.

Scale, if one wanted it: a campaign of 5 operators x 10 seeds x 500 proposals is 25,000 LLM
calls. Measured here, a proposal costs about 3,000 tokens in and 400 to 1,200 out, $0.017 on
average at list price as `claude -p` reports it: about $425 for the campaign, or several days of
a local 8B model on the 4070. A proof costs $0.04-0.12 and about a minute in one or two rounds
($0.35 and four minutes for the hardest so far, three rounds); the GPU check and timing of a
candidate about 10 s. Neither is the bottleneck.

## Risks

- The proposer knows the answers to the control problems; only operators without a published
  single-pass form test discovery. Finding them is part of the next step.
- A theorem certifies its statement, not that the specification is the right one: the
  specifications are written by hand, short, and reviewed.
- Floating-point behaviour is measured, not proved, in this version.
- The cost model ranks algorithms by memory passes and exps; only the GPU decides speed.
