# TritonBench, judged by value

Every answer [TritonBench](https://github.com/thunlp/TritonBench) publishes for the models of its
paper (`LLM_generated/`: 7,752 answers, models of 2024-25) is run as the benchmark runs it and judged
by value with the public judge, `python -m ttsem judge --tritonbench`, on a CPU. TritonBench's own
check compares the stdout of the answer's file with the reference's, and 342 of its 350 tests print
nothing, so for those it asks only that the file runs. Here the answer file and the reference file each run in their own
`bwrap` sandbox, every Triton launch is executed by ttsem, and what the two tests keep is compared by
value. The README's numbers come from these commands. Nothing of the benchmark is in this
repository: not its data, not its answers, not a list of the answers flagged.

## What you need

- Linux with `bwrap` (bubblewrap) and unprivileged user namespaces: every answer is model-written
  code and runs only inside the sandbox (read-only filesystem, your home directory and the host's
  sockets in /run hidden, of your environment only the few variables a run needs, no network, its
  own pid namespace).
- Python 3.12 and a clone of this repository, installed with the Triton wheel and CPU torch:

  ```console
  $ pip install --index-url https://download.pytorch.org/whl/cpu \
                --extra-index-url https://pypi.org/simple -e '.[triton]' torch==2.14.0
  ```

  The published numbers were made with Triton 3.8.0, torch 2.14.0+cpu, numpy 2.5.3, the judge as it
  is on `main` since `bf51173` (27 Sep 2026), a memory cap of 6 GB per run (`TTSEM_MEM_GB`, the
  default) and 300 s per run.
- About 160 MB for the benchmark and 5 GB for the output (Triton caches and the arrays the tests keep).
- The benchmark, at the commit the numbers were made on (Apache-2.0):

  ```console
  $ git clone https://github.com/thunlp/TritonBench
  $ git -C TritonBench checkout 603e28a5050e8c268f6883a69709d477a272d49a
  ```

## Run it

```console
$ python audits/tritonbench/prepare.py TritonBench tb/pred
$ python audits/tritonbench/manifest.py tb/pred TritonBench tb
$ python -m ttsem judge --tritonbench --manifest tb/all.jsonl --out tb/out --jobs 2 --timeout 300
$ python audits/tritonbench/table.py tb/out/rows.jsonl
```

- `prepare.py` writes the file the benchmark runs for each answer (the answer as the benchmark
  extracts it, the separator, the task's test), with the benchmark's own mapping from answer to task
  reimplemented so that nothing of theirs is executed: 7,752 files in a few seconds, one folder per
  model under `tb/pred/{G,T}`.
- `manifest.py` writes `tb/G.jsonl`, `tb/T.jsonl` and `tb/all.jsonl`: one line per answer, with its
  task (the benchmark's own file), its id `<bench>.<model>.<task>`, the benchmark and the model.
- The judge runs each task's reference once (and a second time under another poison pattern when an
  answer's values are to be compared with it) and each answer once. With 2 jobs on a 16-core laptop
  it took about 1 h 30 min for G and 45 min for T. More jobs go faster where memory allows. The batch
  can be stopped and started again: it reads back the records already in `tb/out`.
- `table.py` prints the summary table.

## What comes out

On the terminal, one line per answer (`G.<model>.<task>: verified: ...`, `wrong: the test's results
differ from the reference's: ...`, `unsafe: out-of-bounds read in kernel ..., <file>:<line>: <source>`,
...) and the counts. In `tb/out`: one record per answer, `<id>.tritonbench.json` (the verdict, its
finer class, and for a fault the kernel, the buffer and the answer's line), the file run
(`<id>.tritonbench.py`), its report and arrays, the reference runs under `reference/`, and
`rows.jsonl`, one row per answer. `table.py` prints one table per benchmark, one line per model:

```text
| model | answers | ran | verified | reduced precision | wrong | unsafe | not compared | not judged | wrong or unsafe |
...
| **total** | **4,764** | **596** | **335** | **7** | **58** | **48** | **148** | **53** | **106 of 448** |

G: of the 448 answers that ran and could be judged, 106 (24%) return wrong values or are unsafe.
...
| **total** | **2,988** | **406** | **180** | **0** | **89** | **50** | **87** | **76** | **139 of 319** |

T: of the 319 answers that ran and could be judged, 139 (44%) return wrong values or are unsafe.

All: 7,752 answers; of the 767 that ran and could be judged, 245 (32%) return wrong values (147) or are unsafe (98).
```

"ran" is what TritonBench's own check sees: the file ran to the end, plus the answers the semantics
stopped for a fault a GPU runs past. "not compared" is a task or answer that draws random numbers,
a reference that cannot be compared with, a fault the task's own test causes in the reference too,
or right values with no Triton kernel launched. "not judged" is what this machine or the tool
lacks, a timeout or a crash. The verdicts and their classes are those of
`python -m ttsem judge --help` and of the docstring of `ttsem/judge_tritonbench.py`.

## The control on a GPU (optional)

The same files on an NVIDIA GPU, without ttsem, judged by value against the reference run on the
same device by the judge's own rules, then set next to the judge's verdicts. It needs a Linux box
with the card, the same Triton wheel, a CUDA build of torch, and `bwrap`: the answers run in the
judge's sandbox with the GPU's device nodes let in.

```console
$ python audits/tritonbench/device.py tb/all.jsonl tb/device --jobs 3
$ python audits/tritonbench/agree.py tb/out/rows.jsonl tb/device/rows.jsonl
```

`agree.py` sets both sides in coarse classes (did not run, right, wrong, fault, not comparable, not
judged), prints the matrix and ends with the agreement over the answers both sides decide:

```text
Both sides decide on 7,280 answers and agree on 7,264 (99.8%).
```

That figure comes from the control runs of 19 Sep 2026 on an RTX 4070 (about two and a half hours,
three processes on the card), which `device.py` ports onto the judge's current rules; the port itself
has been run without a GPU only. A rerun can move a few answers: the 19 Sep table counts three T
answers as random that the current rules compare, and a card with other limits runs other answers
(on the 4070, 11 references stop at `OutOfResources`).
