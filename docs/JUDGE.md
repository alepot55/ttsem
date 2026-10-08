# Judging model-written kernels

`python -m ttsem judge` judges an answer a model wrote for a kernel benchmark, on a CPU, with every
Triton launch executed by ttsem. It needs the Triton extra (see the README) and, for the sandbox,
`bwrap`. `python -m ttsem judge --help` lists every option.

## KernelBench answers

`python -m ttsem judge task.py answer.py` checks a
[KernelBench](https://github.com/ScalingIntelligence/KernelBench) answer written with Triton: the
answer's `ModelNew` against the task's `Model`, by KernelBench's own rule (five seeded trials,
`allclose` at 1e-2; `--rule v3` for the KernelBench-v3 harness), every Triton launch executed by
ttsem, large tasks shrunk to sizes an interpreter runs. The verdict is `verified` (exit 0), `wrong`,
`unsafe` (out-of-bounds access, race between program instances, or memory nobody wrote, at the
answer's own line), `no_kernel` (PyTorch does the work), `error`, `too_slow` or `not_judged`.
`--json` prints the whole record; `--manifest M.jsonl --out DIR` judges a batch (one verdict per id;
`--rule`, `--precision` and `--gpu` apply to the lines that do not set them).

`error` with `why: shared_memory` means every config of an autotuner was skipped and the one it
falls back to needs more shared memory than the GPU (`--gpu`, an H100 by default) has, by Triton
3.8.0's count. The count moves between Triton versions, so this verdict can disagree with a label
produced under another Triton. Launches outside an autotuner are left unchecked against the limit,
by choice of scope, and whether a config compiles is decided for sm_90a whatever `--gpu` says. The
other autotune configs are forced in turn after the trials, except those the autotuner itself would
never run: one it skips, or one its `prune_configs_by` drops (`not_applicable`, `pruned`).

A verdict at reduced sizes says what the kernel does at those sizes: an answer that hard-codes the
benchmark's shape can be right there and wrong at others, and an access out of bounds that only the
real shape reaches is missed. Where the benchmark's own shape is small enough, the judge runs it
too.

## TritonBench answers

`python -m ttsem judge --tritonbench task.py answer.py` does the same for a
[TritonBench](https://github.com/thunlp/TritonBench) answer. The task is a file of the benchmark's own
data (`data/TritonBench_{G,T}_v1/*.py`: the reference operator, a line of `#`, the test); the answer
is the code the benchmark runs for a model (its imports and functions for G, its last fenced block
for T), and the judge puts the task's test after it, as the benchmark does. TritonBench's own check
compares the stdout of the two files and its tests print nothing, so it only asks that the answer
runs. Here the reference file and the answer file each run in their own sandbox, and what the two
tests keep is compared by value: `verified` within rtol 1e-4 or 1e-2 (or, with `tolerance:
reduced`, 5e-2, a float16 accumulator), `wrong`, `unsafe`, `no_kernel` (right values, no Triton
launch), `error`, `too_slow`, or `not_judged` with the reason in `class` (a package or a CPU
operator this machine lacks, a task that draws random numbers, a reference that cannot be compared
with: it does not run, keeps nothing, returns memory it never wrote, or faults under its own test
where the answer does). `--manifest M.jsonl --out DIR` judges a batch and prints the counts per
verdict and in columns (verified, reduced precision, wrong, unsafe, not compared, not judged). The
references run once per task, on a Triton cache of their own; an answer's sandbox sees neither their
results nor the task file, nor the directory tree the task sits in.

## The sandbox

Each answer is model-written code and runs in its own `bwrap` sandbox: a read-only filesystem, your
home directory and the host's sockets in /run hidden, of your environment only the few variables a
run needs, no network, its own pid namespace. Without `bwrap` the judge refuses to run unless given
`--no-sandbox`.

## What the judge has been run on

The commands that redo the TritonBench audit and the two KernelBench-style audits from a clone are
in [audits/](https://github.com/alepot55/ttsem/tree/main/audits); the README's "Evidence" section
has the TritonBench numbers and the agreement with an RTX 4070 control.
