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

The defaults, as the code sets them: `--rule kernelbench`; `--precision fp32`, which casts every
tensor input to float32 as KernelBench does (`keep` leaves the task's dtypes, and `--rule v3`
always keeps them); `--scale auto`; `--full-max` 2**20 elements; `--timeout` 600 s for each run of
one pair, 300 s in a batch; `--jobs 2` in a batch. Every run also has an address-space cap of 6 GB
(`TTSEM_MEM_GB`).

`error` with `why: shared_memory` means every config of an autotuner was skipped and the one it
falls back to needs more shared memory than the GPU (`--gpu`, an H100 by default) has, by Triton
3.8.0's count. The count moves between Triton versions, so this verdict can disagree with a label
produced under another Triton. Launches outside an autotuner are left unchecked against the limit,
by choice of scope, and whether a config compiles is decided for sm_90a whatever `--gpu` says. The
other autotune configs are forced in turn after the trials, except those the autotuner itself would
never run: one it skips, or one its `prune_configs_by` drops (`not_applicable`, `pruned`).

With `--scale auto`, every size constant of the task is divided by the same power of two until its
largest tensor holds at most 2**17 elements, and that reduced pass always runs. The real sizes run
too when the task's largest tensor holds at most `--full-max` elements and the reference model's
parameters at most 32 times that, and always after a `no_kernel` verdict (an answer may launch its
kernel only at the benchmark's own shape). `--scale none` runs the real sizes only. A verdict at
reduced sizes says what the kernel does at those sizes: an answer that hard-codes the benchmark's
shape can be right there and wrong at others, and an access out of bounds that only the real shape
reaches is missed.

## TritonBench answers

`python -m ttsem judge --tritonbench task.py answer.py` does the same for a
[TritonBench](https://github.com/thunlp/TritonBench) answer. The task is a file of the benchmark's own
data (`data/TritonBench_{G,T}_v1/*.py`: the reference operator, a line of `#`, the test); the answer
is the code the benchmark runs for a model (its imports and functions for G, its last fenced block
for T), and the judge puts the task's test after it, as the benchmark does. TritonBench's own check
compares the stdout of the two files and its tests print nothing, so it only asks that the answer
runs. Here the reference file and the answer file each run in their own sandbox, and what the two
tests keep is compared by value. The judge tries three tolerances in turn and the first that holds
sets `tolerance` in the record: rtol 1e-4 and atol 1e-5 (`strict`), else rtol 1e-2 and atol 1e-3
(`lenient`), else rtol and atol 5e-2 (`reduced`, what a float16 accumulator does to a float32
result); the absolute tolerance shrinks with the reference's largest value when that is below 1,
down to 1e-6. Within any of the three the verdict is `verified` (exit 0), or `no_kernel` when no
Triton kernel was launched; the batch counts `reduced` apart. Otherwise the verdict is `wrong`,
`unsafe`, `error`, `too_slow`, or `not_judged` with the reason in `class` (a package or a CPU
operator this machine lacks, a task that draws random numbers, a reference that cannot be compared
with: it does not run, keeps nothing, returns memory it never wrote, or faults under its own test
where the answer does). `--manifest M.jsonl --out DIR` judges a batch and prints the counts per
verdict and in columns (verified, reduced precision, wrong, unsafe, not compared, not judged). The
references run once per task, on a Triton cache of their own; an answer's sandbox sees neither their
results nor the task file, nor the directory tree the task sits in.

## The sandbox

Each answer is model-written code and runs in its own `bwrap` sandbox: a read-only filesystem, your
home directory and the host's sockets in /run hidden, of your environment only the few variables a
run needs, no network, its own pid namespace, no capabilities, and a session of its own with no
controlling terminal. Without `bwrap` the judge refuses to run unless given `--no-sandbox`.

The verdict is not the answer's to write. The answer and the task's reference run in separate
sandboxes; each writes only raw outputs, in a directory of its own, and the judge compares them in
the parent process. A kernel fault reaches the judge on a pipe opened before the answer is
imported, and the run ends there, so an answer cannot catch the fault and fall back on PyTorch to
turn `unsafe` into `verified`. A run that ends normally reports `done` and exits at once, so no exit
handler the answer registered runs after that report; and the only results the judge reads back
from an earlier run are records it wrote itself, in a directory no run can see.

What a run can still read is the host's filesystem, read-only, except the hidden directories above:
system files, your world-readable files, other checkouts. It cannot reach the network, your home
directory, or the records and outputs of the judge's other runs. `--strict` narrows that: the run
sees only what it needs (the Python running the judge and its packages, this package, the task and
the answer, and the few system files the C library reads), nothing else of the host. Use it when the
machine holds data no model-written answer should read; it is off by default because it depends on
the layout of your Python install.

## What the judge has been run on

The commands that redo the TritonBench audit and the two KernelBench-style audits from a clone are
in [audits/](https://github.com/alepot55/ttsem/tree/main/audits); the README's "Evidence" section
has the TritonBench numbers and the agreement with an RTX 4070 control.
