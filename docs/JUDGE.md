# Judging model-written kernels

`python -m ttsem judge` judges an answer a model wrote for a kernel benchmark, on a CPU, with every
Triton launch executed by ttsem. It needs the Triton extra (see the README) and, for the sandbox,
`bwrap`. `python -m ttsem judge --help` lists every option.

## KernelBench answers

`python -m ttsem judge task.py answer.py` checks a
[KernelBench](https://github.com/ScalingIntelligence/KernelBench) answer written with Triton: the
answer's `ModelNew` against the task's `Model`, by KernelBench's own rule (five seeded trials,
`allclose` at 1e-2; `--rule v3` for the KernelBench-v3 harness), every Triton launch executed by
ttsem, large tasks shrunk to sizes an interpreter runs. The verdict is `verified` (exit 0), `wrong`
(where memory nobody wrote shows up, as NaN in the values), `unsafe` (an out-of-bounds access or a
race between program instances at the answer's own line, or an operation with an undefined result:
a division by zero, MIN / -1, two lanes of one store writing different values), `no_kernel`
(no trial launched a Triton kernel), `error`, `too_slow` or `not_judged`.
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
where the answer does), or right values from a launch too large to be checked for races
(`races_unchecked`, below). `--manifest M.jsonl --out DIR` judges a batch and prints the counts per
verdict and in columns (verified, reduced precision, wrong, unsafe, not compared, not judged). The
references run once per task, on a Triton cache of their own; an answer's sandbox sees neither their
results nor the task file, nor the directory tree the task sits in.

## The sandbox

Each answer is model-written code and runs in its own `bwrap` sandbox: a read-only filesystem, your
home directory and the host's sockets in /run hidden, of your environment only the few variables a
run needs, no network, its own pid namespace, no capabilities, and a session of its own with no
controlling terminal. Without `bwrap` the judge refuses to run unless given `--no-sandbox`.

The answer does not write its verdict. The answer and the task's reference run in separate
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

## What the verdicts guarantee

The answer runs in the same Python process as the semantics, so in principle its own code can reach
into the checker. What each verdict holds against is therefore worth stating plainly.

- `wrong` and `verified` tell whether the values are right, and hold against an adversarial answer.
  The values are compared, in the judge's own process, against a reference computed in a separate
  sandbox the answer cannot touch, so no code the answer runs in its own process makes wrong values
  read as right. `verified` means the values are right; an answer that launches a kernel and then
  computes the result in PyTorch is still `verified` (it got the right answer). `no_kernel` is the
  one case that turns on a launch rather than a value: it covers only a run in which no trial
  launched a Triton kernel at all.
- `unsafe` and `no_kernel` rest on two stop-gaps, not a proof. They block every attack we know and
  test (`tests/test_judge_integrity.py`), but a determined adversary with the source can likely
  find another way until the answer and the checker run in separate processes (below). The two
  stop-gaps, in one line each:
  - **the scan** reads the answer's source before it runs and refuses it (`not_judged`) when it
    reaches the checker without an ordinary import: importing the judge or the import machinery,
    code built from strings or bytes (`exec`, `eval`, `compile`, `__import__` in any spelling,
    `base64`/`marshal`/`pickle` fed to them), the interpreter's frames, object graph or module
    table, `ctypes` or other native code, threads, signals, child processes, a dunder attribute
    outside a short list, an attribute name computed at run time, or rebinding a library's own
    attribute. A module bound to another plain name (`w = os`, a default argument, a chain of such
    bindings) is followed as the module itself; one reached through any other expression (a
    lambda's return value, a container, an object's attribute) is left to the integrity check;
  - **the integrity check** snapshots, before the answer is imported, the identity and code of
    every function and class of the whole checker (not a hand-picked list), of the numpy, Triton,
    torch and standard-library attributes it uses, and of the process's threads, signal handlers
    and import hooks, and ends the run (`not_judged`, `tampered`) if any changed before a launch
    runs or before the account is written; the fault, the launch count and the process's exit all
    travel out of band over a pipe carrying a token the answer does not know, captured as closures
    when the run starts, so rebinding the channel's class does not reach the run's own exit, and a
    forged report or a mismatched launch count is `not_judged` (`forged`).
- `verified` also means that every launch was checked for races between program instances. A race
  here is a pair of accesses by two instances, at least one a write, whose result depends on the
  order the instances run in; a store of the bytes the address already holds changes nothing a
  load can see, so it is no conflict for a load, and two atomic updates are no race. A
  launch of more than 4,000,000 memory accesses (`ttsem.memory.ACCESS_BUDGET`) is checked by the
  range of bytes each access touched: where no instance writes into a range another instance
  touches, no race is possible. Where the ranges meet, or one launch leaves more than 2,000,000 of
  them (`ttsem.memory.SPAN_BUDGET`), the launch runs and its values are compared, but no race is
  looked for, and a run with such a launch is `not_judged` (`races_unchecked`) where it would have
  been `verified`; it is still `wrong` or `unsafe` where its values differ or a fault was found.
  The record keeps the count of such launches, and the verdict line names it.
- The race check leaves out no buffer of an answer. Outside the judge, `python -m ttsem.sanitize`
  and `sanitize.session()` leave out one buffer, for TorchInductor's own code: the scratch buffer of
  Inductor's split scan, through which each block passes its partial to the next under a flag that
  the next block spins on, which forces their order. They do so only where the kernel's code calls
  Inductor's helper `exclusive_scan_decoupled_lookback*` with that buffer (the callee resolved
  through the kernel's globals to the function in `torch._inductor.runtime.triton_helpers`; a
  comment or a string that names it is not a call) and the launch exchanged a flag in it. The judges
  turn this off (`session(inductor_lookback=False)`): an answer is untrusted and could race on the
  very buffer it hands the helper, so an answer that calls the helper has that buffer checked like
  any other, and the 64-bit helper's protocol (a plain store and load under the flag) reads as a
  race; the 32-bit helper packs value and flag into one word updated only atomically, which is
  never a race. None of the 7,752 published TritonBench answers calls either.
- Full separation of the answer from the checker (running the semantics in a process the answer's
  code cannot reach at all) is planned for 0.2. Until then the judge is fit for auditing published
  answers and for evaluation, where the answers were not written to defeat it. Used as an RL reward
  against an optimiser that is free to write anything, treat `unsafe` and `no_kernel` with this
  caveat: the two stop-gaps block every attack in the test suite, but they are not a proof, and
  `wrong` versus `verified` is the only distinction that holds unconditionally against adversarial
  code.

## What the judge has been run on

The commands that redo the TritonBench audit and the two KernelBench-style audits from a clone are
in [audits/](https://github.com/alepot55/ttsem/tree/main/audits); the README's "Evidence" section
has the TritonBench numbers and the agreement with an RTX 4070 control.
