# KernelBench-style answers, judged without a GPU

Two public sets of model-written Triton answers to [KernelBench](https://github.com/ScalingIntelligence/KernelBench)
tasks, each answer judged on a CPU by `python -m ttsem judge` under the correctness rule of the
benchmark that labelled it, and the verdicts set against the labels the datasets ship. Nothing of
the datasets is in this repository: not the tasks, not the answers, not a list of the answers
flagged.

| dataset | what it is | label | rule the judge applies |
|---|---|---|---|
| [Infatoshi/kernelbench-v3-runs](https://huggingface.co/datasets/Infatoshi/kernelbench-v3-runs) (CC-BY-4.0) with [kernelbench-v3-problems](https://huggingface.co/datasets/Infatoshi/kernelbench-v3-problems) (MIT) | the winning solution of each (model, GPU, problem) of an agent sweep of frontier models (Feb 2026); only those with a Triton kernel are taken | `correct`: the release's own harness passed it | the KernelBench-v3 harness's (`--rule v3`: five seeds, `max\|diff\| < atol + rtol * max\|ref\|`), the task's own dtypes, the run's GPU |
| [makora-ai/triton-gpu-latency](https://huggingface.co/datasets/makora-ai/triton-gpu-latency) (Apache-2.0), test split | a KernelBench problem followed by a candidate `ModelNew` written with Triton, and its measured latency | `correct`: a latency was measured (the dataset leaves it null when the candidate did not compile, did not match the reference, or failed) | KernelBench's (five seeded trials, `allclose` at 1e-2), fp32, an H100's shared memory |

## What you need

- Linux with `bwrap` (bubblewrap) and unprivileged user namespaces: every answer is model-written
  code and runs only inside the sandbox (read-only filesystem, no network, its own pid namespace).
- Python 3.12 and a clone of this repository, installed with the Triton wheel and CPU torch:

  ```console
  $ pip install --index-url https://download.pytorch.org/whl/cpu \
                --extra-index-url https://pypi.org/simple -e '.[triton]' torch==2.14.0
  ```

  The audit used Triton 3.8.0, torch 2.14.0+cpu, numpy 2.5.3, the judge at `9e42531` (committed on
  24 Sep 2026, run on 26 Sep), a memory cap of 6 GB per run (`TTSEM_MEM_GB`, the default) and 900 s
  per pass (the batch default is 300 s). The judge's fixes on `main` since then (autotune configs that do not fit the GPU or do
  not compile, the capture of the IR trace) can move a few verdicts.
- `huggingface_hub` for the `hf` command (`pip install huggingface_hub`; older releases call it
  `huggingface-cli download`), and `pyarrow` to read the makora parquet. No account is needed.

## Run it

KernelBench-v3, at the revisions the audit used:

```console
$ hf download Infatoshi/kernelbench-v3-runs --repo-type dataset \
      --revision 6767f50a1145df2868f655ad48d12575303a2efc --local-dir kb/v3runs
$ hf download Infatoshi/kernelbench-v3-problems --repo-type dataset \
      --revision 3f419ad821f5a1645c00f8a4487e32e3d630c215 --local-dir kb/v3problems
$ python audits/kernelbench/prepare.py v3 kb/v3runs kb/v3problems kb/v3
$ python -m ttsem judge --manifest kb/v3/manifest.jsonl --out kb/v3/out --jobs 2 --timeout 900
$ python audits/kernelbench/table.py kb/v3/out --manifest kb/v3/manifest.jsonl --by model
```

makora, the test split (241 MB, sha256 `1163472651257b0ed4d7baf38c6b93b8dcf24d65336bb0d7e62e4cc7f61ae6d3`),
sampled per problem of levels 1 and 2 as the audit sampled it: a first sample of 30 rows labelled
correct and 10 labelled failed per problem, then a second one of 15 and 5 from the rows left.

```console
$ curl -L -o kb/makora_test.parquet https://huggingface.co/datasets/makora-ai/triton-gpu-latency/resolve/3b30911350fa433e1cb97ec2a41bd8908dd35d5d/data/test.parquet
$ python audits/kernelbench/prepare.py makora kb/makora_test.parquet kb/mk --per-problem 30,10 --seed 0
$ python audits/kernelbench/prepare.py makora kb/makora_test.parquet kb/mk2 --per-problem 15,5 --seed 1 \
      --skip kb/mk/manifest.jsonl
$ cat kb/mk/manifest.jsonl kb/mk2/manifest.jsonl > kb/makora.jsonl
$ python -m ttsem judge --manifest kb/makora.jsonl --out kb/makora_out --jobs 2 --timeout 900
$ python audits/kernelbench/table.py kb/makora_out --manifest kb/makora.jsonl
```

- `prepare.py` writes the task and answer files and `manifest.jsonl`: one line per answer with its
  task, its answer, the dataset's label, and for v3 the rule, precision and GPU the judge applies.
  It prints how many answers it took. The manifests are shuffled, so a run stopped early (`--limit
  N`, or a kill) is still a sample of every problem; the audit judged the makora sample until its
  time ran out, not the whole of it.
- The judge runs each answer once at sizes reduced to what an interpreter runs (the `scaled` pass),
  and again at the benchmark's own shape where that is small enough (the `full` pass). With 2 jobs it
  takes about half a minute per answer on a 16-core laptop, with a long tail: the v3 release took
  about five hours. The batch can be stopped and started again: it reads back the reports already in
  the output directory.
- `table.py` sets the verdicts against the labels.

## What comes out

On the terminal, one line per answer and pass (`<id>: verified: 5 trial(s), max abs diff 3.1e-06
[sizes / 8]`, `wrong in trial 1 (seed 42): max abs diff 0.5`, `unsafe: out-of-bounds read in kernel
..., <answer>.py:<line>: <source>`, ...) and the counts per pass. In the output directory: one
report per answer and pass (`<id>.scaled.json`, `<id>.full.json`: the verdict, the trials, the
sizes, the autotune configs checked, and for a fault the kernel and the answer's line) and
`rows.jsonl`. `table.py` prints, per dataset:

```text
## v3: N answers judged

| verdict | labelled correct | labelled failed |
|---|---|---|
| verified | . | . |
| wrong | . | . |
| unsafe | . | . |
| no_kernel | . | . |
| error | . | . |
| too_slow | . | . |
| not_judged | . | . |
| total | . | . |

| verdict, in detail | correct | failed |
|---|---|---|
| wrong: values | . | . |
| unsafe: oob_read | . | . |
...

labelled correct: the judge decides on N, verifies N (P%), and judges N wrong, unsafe or not a kernel
```

then the full-shape pass where it ran and, with `--by model`, the answers labelled correct by model.
A v3 file that the release evaluated twice, once correct and once not, is labelled `ambiguous`. The
verdicts are those of `python -m ttsem judge --help`: `verified`, `wrong`, `unsafe`, `no_kernel`
(PyTorch does the work), `error`, `too_slow`, `not_judged` (the tool cannot say). A verdict at
reduced sizes says what the kernel does at those sizes: an answer that hard-codes the benchmark's
shape can be right there and wrong at others, and an access out of bounds that only the real shape
reaches is missed. Running the answers on a GPU through the benchmarks' own checks is not part of
this folder.
