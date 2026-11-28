# An honest evaluation

Written on 28 Nov 2026, after the runs in [DIARY.md](DIARY.md) and a review of the whole
prototype (DIARY.md, "the review"), against the prior work in [RESEARCH.md](RESEARCH.md).

## In one paragraph

The pipeline works end to end on one laptop GPU: an LLM proposes a program, the checks keep the
correct ones and label their floating-point behaviour, the GPU times their kernels, and Lean
proves a kept form equal to its specification over every blocking when asked to (`--prove`).
Nothing it found is new. With no lemma in its prompt, the model writes the published
reformulation of each of the three control operators as its first proposal in 16 of 18 runs and
as its second in the other two, blind or not, with or without the archive, often byte for byte
as this repository writes it. The proofs are proofs of published facts, which TorchLean, tang
and others proved in Lean in 2026. What this prototype adds is a way of putting the three tiers
on one object (a program in an IR of blocked folds) and a proof judge rebuilt against the two
attacks its review found: a tool, not a result. The step most likely to give a strong paper is
not more search on known operators but **certified floating-point bounds produced
automatically**, used as the constraint of a search whose answers the model cannot recall.

## What was asked, and what was delivered

| asked | delivered | where |
|---|---|---|
| a short design doc | the three tiers, the IR, the checks, the proof judge, costs | DESIGN.md |
| a working tier-1 prototype | proposer (`claude -p` or any OpenAI-compatible model), exact tests over sampled blockings and merge trees, floating-point class, cost model, MAP-Elites archive, IID and enumerative baselines, Triton kernels checked and timed on the GPU as the fitness (`--gpu`); the 18-run control, run twice (the first run was contaminated) | `loop.py`, `experiments.py`, DIARY.md |
| a first Lean proof of tier 2 | the online softmax, proved by the model in one round and checked by Lean, and the end-to-end theorem for any merge tree, closed by the framework; also attention, two variance forms and the log-sum-exp softmax (5 forms); a wrong merge not proved; then, with `loop --gpu --prove`, every elite of three searches proved in one round, and six programs found by searches added to the library, 11 in all | `lean/Discovery/Library/` |
| exact versus floating-point equivalence | two labels: proved over ℝ (`E`) or tested only (`T`); a measured float32 class (`stable`, `inaccurate`, `nonfinite`) on adversarial inputs, the errors kept as numbers. Not a proved bound | `evaluate.py` |
| self-improvement | proved lemmas fed to the proposer (opt-in, since a lemma about a known form gives the form away), rejected candidates fed back with their counterexamples, accepted ones recorded; a program the library already holds is not proved again, its stored proof is judged again; no gain measured | `knowledge.py`, `prove.py` |
| a research pass first | 5 topic searches, 3 searches for prior work; 18 sources opened again to check them, 15 of them cited | RESEARCH.md |
| a diary with the negative results | the contaminated control, the judge that was fooled, the weaker tests, the timing errors, each with what changed | DIARY.md |

## What is not new

- **The forms**: online softmax (2018), its blocked combine (FlashAttention, 2022), the
  log-sum-exp state, FlashDecoding's normalised combine, Welford (1962) and its parallel merge.
- **Deriving them automatically**: Opera, Neptune, Flashlight, EquiForge, tang. The model's
  "rediscovery" is recall: with no lemma in its prompt, one of its first two proposals was the
  program written in this repository, byte for byte, in 10 of 18 runs.
- **Proving them in Lean**: TorchLean (with a floating-point bound), tang, VeriTile, TensorLean.
  Our proofs are cheap (one to three rounds, 4 to 35 cents each) because the facts are
  simple and the framework does the composition.
- **An LLM loop with proofs and GPU timing**: TAIC keeps a PTX candidate only if a verifier
  proves it equivalent; a VeriTile fork keeps a Triton variant only with a measured gain and a
  Lean proof; KernelHarness measures, applies a trick and proves it on a PTX model; Tessella
  proposes the proof and contract tiers for GPU kernels. (AutoAssumption, also an LLM search over
  Lean proofs, finds ML-systems algorithms, not kernels.)

## What is, at most, new

None of these is a paper on its own.

- One IR read three ways (numpy over any blocking, Lean, Triton), so one candidate carries its
  tests, its theorem and its kernel. Close in spirit to tang's monoid IR, but with a search on
  top and floating point measured rather than ignored.
- The end-to-end theorem (`blocked_eq_spec`: over every merge tree of nonempty blocks, the
  blocked computation equals the specification) closed automatically from two small
  obligations: the model does not have to prove the composition.
- A judge whose verdict does not depend on anything Lean prints while it elaborates the model's
  text: statements compiled apart, every declaration replayed in Lean's kernel, types and axioms
  read from the terms by a fixed checker. The first judge, which believed `#print axioms`, fell
  to two attacks its reviewers wrote; prover pipelines that trust Lean's messages have the same
  hole.
- Search, GPU and proof on the same object with no human step: in one run the search proposed
  the attention state that keeps the output normalised (the combine FlashDecoding uses), the GPU
  timed it, and the prover proved it equal to three-pass attention over every blocking. The form
  is published; the chain is the point.
- The pairing of a Lean proof with a measured floating-point class makes one point concrete: the
  one-pass sum of squares is proved exact like Welford and is off by a factor of 1,000-2,400 in
  float32 on shifted data, while Welford, proved by a theorem of the same shape, stays `stable`
  but is 3-140 times less accurate than two passes. Known to numerical analysts; rarely shown by
  a pipeline.

## What the numbers say

- **Search**, the clean control: 108 calls, $1.88. Proposal 1 is the published algorithm in 16
  of 18 runs and proposal 2 in the other two, both from the seed's prompt, before the archive
  can act; blind changes nothing. The archive's only effect is variety: the one-field
  log-sum-exp state in 3 of 4 archive or blind softmax runs, never under IID. No reply was wrong
  over the reals, and none was unusable in floating point, so the floating-point class had
  nothing to sort. On known operators the search tier has nothing to do.
- **Enumeration without an LLM** finds the template merges (online softmax, attention) in under
  6 s and does not find Welford's `q`: at the default size cap its 9-node subterm is never
  built. The one combine that is not a template is out of its reach and within the model's
  recall.
- **GPU** (RTX A1000, medians of three timings): on rows of 32,768 one pass fewer is the
  expected gain (softmax 1.29x, variance 1.92x over the seed), and the online softmax runs at
  the copy bandwidth; on rows of 4,096 it gains nothing, because the re-reads come from the
  cache. Attention: 2.2x over the seed. Against PyTorch: the online softmax wins by 1.30x on
  long rows and loses by 17% on short ones; Welford ties `torch.var_mean` on long rows and takes
  14% less time on short ones; the attention kernels are 2.6-3.1x slower than SDPA, because the
  code generator has no query tiling and no tensor cores.
- **Proofs**: 5 forms proved by hand-started runs, 1 to 2 rounds, $0.04-0.12 each; one false
  statement not proved. With `--gpu --prove`, every elite of three searches was proved in one
  round (4 of 4, $0.23); two more elites, from runs that hit a bug of the new judge, were proved
  after the fix, in 1 and 3 rounds. That bug (the judge gave Lean relative paths from another
  directory) made the first `--prove` run prove nothing while the model's replies were right.
- **Within a cell the GPU cannot rank**: every kernel of a cell took the same time within its
  range, so the elite is the smallest program among ties; the measured time separates cells.

## Limits that matter

- ttsem's own checks (out of bounds, races, undefined values) do not run on this Windows
  laptop; the kernels were checked against float64 on the GPU only.
- Floating point is measured, not proved. The class threshold hides real differences (Welford is
  3-140 times less accurate than two passes on shifted data and still `stable`).
- The exact tests sample blockings; only a proof covers them all, and the framework proves only
  merges that are homomorphisms. A correct program whose state is not canonical (keeping the
  first block's maximum, say) passes every test, cannot be proved this way and is not sent to
  the prover.
- The theorems are about Lean's reals, where x / 0 = 0: they say nothing about where `/`, `log`
  and `sqrt` are defined. The float64 tests catch a division by zero only on the inputs they try.
- The IR has no element-level temporaries, no conditional, no 2D tiling and no `tl.dot`: it
  cannot express FlashAttention-4's conditional rescaling or a competitive attention kernel.
  Half the replies it rejects in the clean control (6 of 13) wanted element-level temporaries.
- One model, two seeds, six proposals per run: enough for a control, not for statistics. One
  laptop GPU: the attention timings repeat within a few percent at 4,096 keys and only within
  about 15% at 16,384.
- The judge resists the attacks found so far, not every attack: its first step is a deny list
  of Lean's command keywords, which has to follow new versions of Lean and Mathlib. It has not
  been run against a model told to cheat.
- The library tells programs apart by their digest, which counts names. Of the six programs the
  searches added, one is a new cell (the two-field attention); the other five are algebraic
  variants of forms it had, and two of those are the same program up to a temporary's name. It
  grows; it does not yet merge what is the same algorithm.

## The next step

**Make the floating-point bound a theorem the pipeline produces automatically, then search
under it.** Concretely:

1. A rounding model in Lean (the standard model per operation with exp's error as a parameter,
   or FloatLib's) and, per candidate, a generated statement "the float32 kernel's output is
   within B(x) of the real one", with B proposed by the model and checked by Lean. Control: get
   TorchLean's online-softmax bound back with no hand proof; then Welford's bound, independent
   of the data's mean, against the sum of squares', which grows with mean²/variance (the
   classification above, made a theorem).
2. Extend the IR with a conditional and element-level temporaries, and target what has no
   floating-point theorem today: FlashAttention-4's conditional rescaling (rescale only when the
   maximum grows by more than τ) for a bound parametric in τ and the block size, and variants a
   model can propose but cannot recall the proof of (power-of-two shifts that make rescaling
   exact in floating point, rescaling every k blocks, accumulating in lower precision).
3. Search with "certified error at most ε" as the constraint and the time on the RTX 4070 as
   the fitness, on Linux so ttsem checks every kernel.

Why this one rather than more operators: it is the step the 2026 work left to hand proofs
(TorchLean says its bound was proved by hand because the automatic tactic did not find the
invariant); it turns the search from recall into discovery, since a certified bound for a new
policy is in no training set; the control shows that on anything published the model answers
from memory, so only a target with no published answer can show discovery; it needs a CPU for
Lean and the one GPU there is; and it builds on everything here. A first result in two to four
weeks of runs is plausible: the bounds of the five forms already proved exact, then the
τ-parametric theorem. The risk is that the model cannot write the inequality proofs; the
fallback is a decision procedure for the positive-sum case (Olver-style relative precision,
where budgets add), which makes most of the proof mechanical.

Considered and ranked lower: new operators of 2024-26 without a fused form (Selective
Attention, Multi-Token Attention) are the higher payoff but need 2D tiling first and may admit no
constant-state form at all (then the result is an impossibility proof); a benchmark of
certified liftings across 30-50 reductions is the safest publication but a weaker headline.

What it would cost at scale: a proposal costs about $0.017 at list price (216 calls over the two
controls, $3.68) and a proof $0.04-0.35, so a campaign of 5 problems x 10 seeds x 500 proposals
is about $425 of model calls, or several days of a local 8B model on the 4070; Lean and the GPU
take seconds per candidate and are not the bottleneck.
