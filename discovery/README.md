# discovery

A research prototype on top of ttsem: an LLM proposes reformulations of GPU operators, the
checks keep the correct ones, the GPU (or a cost model) ranks them, and Lean 4 proves a kept one
equivalent to the operator it replaces when asked to (`prove.py`, or `loop --prove` on a run's
elites). It lives in this repository, outside the `ttsem` package (the wheel does not ship it).

- [DESIGN.md](DESIGN.md): the three tiers, the IR, the checks, the proof tier, what it costs.
- [RESEARCH.md](RESEARCH.md): what already exists (most of the first milestone does) and the
  decisions it drove.
- [DIARY.md](DIARY.md): every run, with its numbers, negative results included.
- [EVALUATION.md](EVALUATION.md): what is new, what is not, and the next step.

## What a candidate looks like

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

A sequence of passes over the elements of a row; each pass is a fold with a state, a summary of
one block and a merge of two adjacent summaries. The same text is run by numpy over any blocking
(`semantics.py`), compiled to a Triton kernel (`codegen_triton.py`) and stated in Lean
(`lean_export.py`). The grammar is in `ir.py`'s docstring.

## Setup

Python 3.10 or later, from the repository root:

```console
$ python -m venv .venv && .venv/bin/pip install -e . pytest      # ttsem and numpy
$ .venv/bin/pip install torch triton      # Linux; on Windows: the CUDA torch wheel and triton-windows
```

Lean (for the proof tier): install elan, then fetch Mathlib's cache once (about 7 GB):

```console
$ cd discovery/lean && lake exe cache get && lake build
```

On Windows put elan in a path without spaces and set `ELAN_HOME` (this prototype used
`C:\dev\tools\elan`); `prove.py` finds `lake` there when it is not on `PATH`.

The model: `claude -p` (Claude Code, logged in) by default, or any OpenAI-compatible endpoint,
such as a local model on the GPU machine: `--model openai:MODEL@http://localhost:8080/v1`, which
`loop`, `experiments` and `prove` all accept.

## Running it

```console
$ python -m pytest discovery/tests -q                          # about 4 minutes; GPU and Lean tests skip without them
$ python -m discovery.loop --problem softmax --iterations 12 --model sonnet          # tier 1: the search
$ python -m discovery.loop --problem softmax --iterations 8 --gpu --prove            # timed on the GPU, elites proved
$ python -m discovery.experiments --seeds 0 1 --iterations 6   # the control: archive, blind, IID, no lemma shown
$ python -m discovery.enumerate --problem attention --form online                     # the no-LLM baseline
$ python -m discovery.gpu --problem softmax --program online --bench                  # the kernel, timed
$ python -m discovery.prove --problem softmax --program known:online --model sonnet   # tier 2: the proof
$ python -m discovery.prove --recheck-library                                         # judge the stored proofs again
$ python -m discovery.ttsem_check --problem softmax --program online                  # ttsem's checks (Linux)
```

Floating point (the branch `discovery-fp`):

```console
$ python -m discovery.fptable --out discovery/results/m1                 # every library program's error in u, measured
$ python -m discovery.adversary --problem softmax --program known:online # the worst rows a climb finds
$ python -m discovery.libm --out discovery/results/m2/libm.json           # numpy's float32 exp and log, every input
$ python -m discovery.fpbound --program SoftmaxOnline                     # B proposed, attacked, proved in one-shot rounds
$ python -m discovery.agentprove --program SoftmaxLse                     # the same with the agentic prover (Lean in a sandbox)
$ python -m discovery.fpcheck --program SoftmaxLse --bound "..."          # F1: a long attack on a proved bound
$ python -m discovery.fpreport                                            # proved bounds, tightness, the certified order (F4)
$ python -m discovery.policyrun --policy conditional --kind exact         # rescaling policies: exact, overflow, fp, worst
$ python -m discovery.certsearch --arm cert --seed 0                      # the search under a certificate, and its baselines
```

Run output goes under `discovery/runs/` (not versioned); proved forms go to
`lean/Discovery/Library/` with an index the search shows the model when given `--library`
(tier 3). `loop` shows it by default; `experiments` shows no lemma unless given one, since a
lemma about a known form would put that form in every prompt.

## Modules

| module | what it does |
|---|---|
| `ir.py` | the IR: reader, AST, scoping and types, printer |
| `semantics.py` | numpy runs over any blocking and merge tree, in float64, float32, float16 or emulated bfloat16 |
| `problems.py` | softmax, single-query attention, mean and variance: seed (the specification), reference, inputs |
| `evaluate.py` | the checks in order: parse, exact tests, floating-point class, cost |
| `cost.py` | a roofline estimate from bytes and exps per element |
| `codegen_triton.py`, `gpu.py` | the Triton kernel of a program; its check on lengths around the block and its timing next to torch |
| `ttsem_check.py` | the kernel under `ttsem.sanitize` (out-of-bounds, races, undefined values) |
| `lean_export.py`, `prove.py` | the Lean statements of a candidate; the proof loop with the model and its judge |
| `lean/Discovery/Blocked.lean` | the framework: any merge tree over nonempty blocks evaluates to the summary of the whole |
| `lean/Discovery/Checker.lean` | the judge's last step: replays the compiled proofs in Lean's kernel, checks their types and axioms |
| `archive.py`, `propose.py`, `loop.py` | MAP-Elites archive, the proposer's prompt, the search loop |
| `enumerate.py` | bottom-up synthesis of a merge with observational equivalence, no LLM |
| `knowledge.py` | tier 3's memory: rejected candidates, accepted ones, proved lemmas |
| `llm.py` | `claude -p`, OpenAI-compatible endpoints, scripted answers for the tests |
| `experiments.py` | the rediscovery control experiment |
| `fsem.py` | floating-point semantics: one rounding per operation, a block's sum along a tree, merges along any tree, an exact reference |
| `adversary.py`, `fptable.py` | inputs that maximise the error; the library's errors in units of u |
| `libm.py` | the errors of numpy's float32 `exp` and `log` on every input: the rounding model's parameters |
| `lean/Discovery/Approx.lean` | the rounding model, floating-point evaluation of a merge tree, the depth law, the framework up to a relation |
| `fp_export.py`, `bound.py` | the floating-point statement of a program with its bound left open; the grammar of the bound |
| `fpbound.py` | the proposal of a bound, the adversary on error/bound, the judge, one-shot proof rounds |
| `agentprove.py` | the agentic prover: the model with Lean in a sandbox outside the repository; its proof judged as above |
| `fpcheck.py`, `fpreport.py` | F1 (a long attack on a proved bound) and F2 (tightness); the summary and the certified order (F4) |
| `policies.py`, `policyrun.py` | rescaling policies (FlashAttention-4's conditional rescaling): exactness, overflow, bound, worst cases |
| `certsearch.py` | the search with a certified bound as the fitness, and its two baselines |
