# What already exists, and what it decides here

A targeted search of 2025-26 work, run on 27-28 Nov 2026 before building: five topic searches
(evolutionary search with LLMs, LLMs with Lean, kernel and tensor-program verification, deriving
GPU algorithms, formal floating point) and three searches whose only job was to find prior work
that already does what this project set out to do. Only what bears on a decision is kept. A
separate pass opened 18 of the sources again to check that they exist, carry the date given and
say what is attributed to them, and all 18 held; ✓ marks the 15 of them cited here. The others
were opened once, by the search that cites them.

## The first milestone is not new

| work | date | what it already does |
|---|---|---|
| [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) | 2018-05 | the running max and the rescaled sum, by hand |
| [Opera](https://arxiv.org/abs/2404.04743) ✓ (PLDI 2024) | 2024-04 | synthesises online algorithms from multi-pass ones, log-sum-exp among them; a sequential fold, no associative block combine |
| [Neptune](https://arxiv.org/abs/2510.08726) (PLDI 2026) | 2025-10 | fuses attention's reductions and derives the repair term exp(m_old - m_new) with a solver, from a user template; fails on Welford's mean and variance, by the authors' account |
| [Flashlight](https://arxiv.org/abs/2511.02043) (MLSys 2026) | 2025-11 | turns the stable two-pass softmax into the online form inside TorchInductor, as a homomorphism, proof on paper |
| [EquiForge](https://arxiv.org/abs/2609.12330) | 2026-09 | derives tiled attention with the online update by equality saturation; the repair rule is Neptune's, written by hand |
| [tang, "Don't Write Kernels, Derive Them"](https://muni.works/research/tang) ✓ | 2026-11 | recognises the online softmax as a monoid; Lean 4 + Mathlib proofs over ℝ of the fusion, the tiled reductions, the online form's equality with two passes; no floating point |
| [TorchLean](https://lean-dojo.github.io/TorchLean/blueprint/Verification-and-Certificates/Rounding-Error-In-Attention/) ✓ ([arXiv 2602.22631](https://arxiv.org/abs/2602.22631)) | 2026-02, chapter updated 2026-11 | Lean 4: `Tensor.softmax_online_eq`, `Tensor.attention_online_eq` (online = two-pass over ℝ for every blocking, mask and scale) and `onlineAverage_error`, a rounding bound in every IEEE format; the bound proved by hand because the automatic tactic does not find the invariant |
| [VeriTile](https://veriinfra-lab.github.io/VeriTile/) ✓ | 2026-09 | Triton embedded in Lean 4; agents write the proofs; FlashAttention-1 forward equals naive attention over ℝ |
| [TensorLean](https://tensorlean.ai/) ✓ | 2026-11 | Lean + SMT on the PTX of production kernels (all 37 FlashAttention 2-4 kernels): 83 of 149 instances proved, 44 bugs; loop invariants of the online softmax found automatically |
| [OpenEvolve on Metal kernels](https://huggingface.co/blog/codelion/openevolve-gpu-kernel-discovery) | 2025-06 | an LLM-evolved two-pass fused online softmax, announced as an "algorithmic breakthrough" |
| [K-Search on MLX](https://bair.berkeley.edu/blog/2026/07/29/cuda-to-mlx-k-search/) ✓ | 2026-07 | an evolved attention kernel "independently discovers" the online softmax, with FlashAttention-2 knowledge in its context |
| [Agentic Kernel Generation](https://github.com/puneethgv/Agentic-Kernel-Generation) ✓ | 2026-07 | an LLM proposes Triton kernels, MAP-Elites with speed as fitness, on an 8 GB laptop GPU: "the search discovered the flash algorithm" |

Strassen on small blocks does not pay on a GPU: Strassen-like CUDA kernels win only on large
matrices, by 12-22% in fp32 and at most 16% in fp16 on tensor cores (SubCuber, PACMPL 2026); its
proof in Lean is one tactic (`noncomm_ring`, as in CLRS-Lean), and small-rank schemes are a
crowded race run on CPUs (flip graphs, AlphaEvolve's 4x4 with 48, a 3x3 rank-23 scheme with 55
additions proved in Lean, [arXiv 2610.27451](https://arxiv.org/abs/2610.27451) ✓).

**Decision.** The online-softmax rediscovery is run as a control of the pipeline, its
contamination stated; the proof is ours only in that the pipeline produced it.

## The loop: LLM, proof, GPU

| work | date | what it does | what it does not |
|---|---|---|---|
| [AlphaEvolve](https://arxiv.org/abs/2506.13131) | 2025-06 | evolution with an LLM ensemble, evaluation cascades, a MAP-Elites/island database; kernels: +23% Pallas matmul tiling, +32% on an XLA FlashAttention kernel | correctness of kernels by construction or by random tests and human review |
| [Mathematical exploration and discovery at scale](https://arxiv.org/abs/2511.02864) | 2025-11 | 67 problems; one full chain AlphaEvolve → Deep Think → AlphaProof (Lean) | that chain worked "in rare cases"; verifiers that leak get exploited |
| [TAIC](https://arxiv.org/abs/2609.36800) ✓ | 2026-09 | an LLM writes PTX from Triton; with "formal verification in the loop" a candidate counts only if Volta proves it equivalent | lowering of a given algorithm, floats treated as reals |
| [A Lean proof let agents cut a workload's runtime by 26%](https://casella.dev/blog_pytorch_narrowing.html) ✓ | 2026-09 | agents keep a variant of a Triton kernel only with a measured gain and a Lean proof (VeriTile fork) | one class of transformation (index width), no floating point |
| [KernelHarness](https://github.com/pythongiant/KernelHarness) ✓ | 2026-11 | measure on an H100, apply a trick, prove it in Lean on a PTX model, commit | ideas from a fixed catalogue, proofs at tiny sizes, no floating point |
| [AutoAssumption](https://arxiv.org/abs/2611.14844) ✓ | 2026-11 | LLM search over the assumptions of Lean proof graphs finds state-of-the-art ML-systems algorithms | not operators or kernels |
| [Growing Languages for Self-Improving Agents](https://geometricagi.github.io/2026/10/23/growing-languages-for-self-improving-agents.html) (Tessella) ✓ | 2026-10 | proposes tiers 2 and 3 for GPU kernels: rewrites citing Lean lemmas, numerical contracts | a proposal, no experiments yet |

Two findings about the loop itself:

- Archives, islands and ensembles give no stable gain over simple sampling
  ([Simple Baselines are Competitive with Code Evolution](https://arxiv.org/abs/2602.16805);
  [Hill Sampling](https://arxiv.org/abs/2609.25510)); the search space and the evaluator matter.
  OpenEvolve and ShinkaEvolve are the baselines reviewers expect.
- Every test-based fitness gets gamed: KernelBench's own reports, Sakana's AI CUDA Engineer,
  [robust-kbench](https://arxiv.org/abs/2509.14279) (mean speedup 3.13x → 1.49x without the
  contaminated tasks), [TritonRL](https://arxiv.org/abs/2510.17891),
  [KernelBench-Verified](https://arxiv.org/abs/2607.16241) (time against PyTorch with TF32 on),
  [Gaming Without an Attacker](https://arxiv.org/abs/2608.08722) (30% of wins do not transfer to
  a held-out configuration).

**Decision.** A minimal loop of our own, IID sampling as the baseline at equal budget, checks
outside the evolvable text, held-out shapes for the kernels, several seeds per claim.

## Finding the state and the combine

Turning a multi-pass reduction into a one-pass, blockable one is a lifting problem: find extra
state and an associative combine. Divide-and-conquer synthesis for loops (PLDI 2017, 2019)
synthesises them and shows that a constant-size lifting may not exist; [AutoLifter](https://arxiv.org/abs/2202.12193)
(TOPLAS 2024) solves 82 of 96 such tasks on integer lists; [Ink](https://arxiv.org/abs/2508.15109)
(OOPSLA 2025) also refutes non-homomorphisms. With LLMs: LLMLift (NeurIPS 2024), STAGG (PLDI 2025,
where the LLM alone solves 34 of 77 and with enumeration 76). None of these works on exp and max
over floats, on attention, on GPUs, or with a proof assistant.

**Decision.** The IR is a language of blocked folds, so a candidate *is* a lifting; the
bottom-up enumerator is the no-LLM baseline for the combine.

## Floating point

- [Accurately computing the log-sum-exp and softmax functions](https://arxiv.org/abs/1909.03469)
  (2021): error of the shifted log-sum-exp and softmax; the shift does not hurt because large
  coefficients multiply small exponentials. A formal follow-up in Why3 appeared in FMSD 2026.
- [FloatLib](https://arxiv.org/abs/2609.19352) (2026-09): IEEE and ML formats in Lean 4 with a
  `roundoff` tactic producing mixed error theorems; TorchLean's bounds are built on it. Flean and
  lean-ieee754 are lighter alternatives.
- A purely relative bound on attention's output is false (the case of ELSA,
  [arXiv 2604.23798](https://arxiv.org/abs/2604.23798)); state it against Σ p|v| or absolutely.
- FlashAttention-4's conditional rescaling (rescale only when the maximum grows by more than τ)
  is exact over ℝ and has no floating-point theorem parametric in τ that the search found.

**Decision.** Two labels: exact (Lean, over ℝ) and a measured floating-point class; a proved
bound over an abstract rounding model is the next layer, not this one.

## Lean tooling

LeanInteract (works on Windows), the Lean REPL, kimina-lean-server; open provers that fit 12 GB:
Goedel-Prover-V2-8B, Pythagoras-Prover-4B. Judge proofs by more than "Lean printed no error":
DeepSeek-Prover-V2-7B exploited a Lean 4.9.0 bug; check `#print axioms` and keep the statement
out of the model's reach. Library learning: LEGO-Prover's lemma reuse did not hold up
([EACL 2026](https://doi.org/10.18653/v1/2026.eacl-long.163) ✓: gains
vanish or reverse at equal compute); [ProofEvolve](https://arxiv.org/abs/2608.26334) ✓ reports a
2.7x gain on synthetic families with a growing library.

## Where something new is plausible

From the searches, not yet checked by building anything:

- operators of 2024-26 with no fused or one-pass form: Selective Attention (a prefix sum over
  the queries, no flash kernel found), Multi-Token Attention (key-query convolution, "not
  compatible with optimised kernels" by its authors);
- online forms released without a proof: softpick, stick-breaking attention, the chunked delta
  rule family (DeltaNet, Gated DeltaNet, KDA);
- a floating-point theorem for FlashAttention-4's conditional rescaling, parametric in τ;
- impossibility: a proof that α-entmax or post-softmax MTA has no constant-state one-pass form;
- error invariants proposed by the LLM and checked by FloatLib, the step TorchLean did by hand.
