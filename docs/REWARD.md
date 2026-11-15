# The judge as a reward

`ttsem.reward` turns the judge's verdict on a model-written Triton kernel into a reward for
training: it runs the answer on a CPU, in the judge's sandbox, against the task's reference, and
says whether the kernel is right, whether it reads or writes outside its buffers or races, and
where, in a paragraph the model can be shown. No GPU is needed in the reward loop.

**Status: new in 0.2, prototype-grade.** The API may still change. It is tested against a fake
worker (every verdict-to-reward rule, the tiers, the cap), against the judge's own fixtures with
the real worker and sandbox, and as a module loaded by path; it has not been tested inside a verl
release, and this release shows no training run that used it. [What it does not
catch](#what-it-does-not-catch) is part of the status, not a footnote.

It judges two kinds of task, the ones [docs/JUDGE.md](JUDGE.md) describes: a KernelBench-style task
(the reference defines `Model`, `get_inputs` and `get_init_inputs`; the answer defines `ModelNew`)
and, with `format="tritonbench"`, a TritonBench task (the reference, a line of 146 `#`, the test).

## A first call

```python
from pathlib import Path

from ttsem.reward import Judge, extract_answer

judge = Judge(
    cache_dir=Path("~/.cache/ttsem_reward").expanduser(),  # on disk, with room: see "Cache"
    python=Path("~/venvs/judge/bin/python").expanduser(),  # the judge's own venv: see "The worker"
    workers=4,                                             # passes judged at once
    mode="gate",
)
verdict = judge.score(reference_src, extract_answer(completion))
print(verdict.score, verdict.verdict, verdict.kind)
print(verdict.feedback)                       # one paragraph, for the model's next turn
reward = judge.gate(gpu_reward, verdict)      # never above gpu_reward
judge.close()
```

## Gate mode and score mode

`mode` says what `Verdict.score` is.

- **`mode="gate"`** (the default) is for a reward you already have, usually a GPU reward that
  measures speed. `score` is a multiplier: 1.0 for a kernel the judge verified, 0.0 for one it
  rejected, `None` for an abstention. `Judge.gate(base, verdict)` applies it and can only lower the
  reward: it returns `base` for a verified answer, `min(base, fail_value)` for a rejected one
  (`fail_value` is 0.0 unless you set it), and for an abstention `base` (`on_abstain="keep"`, the
  default) or `min(base, fail_value)` (`on_abstain="zero"`). The worst an answer that fools the
  judge can get is the reward it would have had without it.
- **`mode="cpu"`**, score mode, is for a trainer with no GPU reward: `score` is the reward, 1.0 for
  verified, 0.0 for rejected, and `None` for an abstention, which a trainer that needs a float
  replaces by `abstain_value` (the verl adapter uses 0.0).

The two modes differ only where a failure may be the judge's environment rather than the answer
(next section): score mode gives 0.0 where gate mode abstains.

## What a verdict becomes

`scores(verdict, kind)` is the whole table, `(score-mode reward, gate multiplier)`; `kind_of`
derives `kind`, the finer reason, from the judge's record.

| judge verdict | `kind` | score mode | gate mode |
|---|---|---|---|
| `verified` | none | 1.0 | 1.0 |
| `wrong` | `values`, `unwritten` (memory nobody wrote shows as NaN), `autotune` (right under the first config only), `weights`, `shape`, `not_tensor` | 0.0 | 0.0 |
| `unsafe` | `oob_read`, `oob_write`, `race`, `poison` | 0.0 | 0.0 |
| `no_kernel` | none (no Triton launch: the work was done in PyTorch) | 0.0 | 0.0 |
| `not_judged` | `tampered`, `forged` (the answer reached into the checker; `Verdict.hack`) | 0.0 | 0.0 |
| `error` | `raised` (the answer raised an exception), `no_code` (the completion holds no code) | 0.0 | 0.0 |
| `not_judged` | `native_code` (the answer builds C++ or CUDA of its own) | 0.0 | abstain |
| `error` | `version` (a compile or import error that may be drift between your Triton and the judge's), `shared_memory` | 0.0 | abstain |
| `error` | `crash`, `configs` (the judge's sweep forced a config the autotuner would skip), `compare_dtype` | abstain | abstain |
| `too_slow` | `timeout` | abstain | abstain |
| `not_judged` | `races_unchecked`, `not_at_task_shape`, `tool_error`, unsupported operations, others | abstain | abstain |

### Abstention

`score is None` means the judge cannot say whose fault the outcome is. It is not a pass and not a
fail, and `Verdict.feedback` then starts with `(the judge abstains)` and names why. An abstention
leaves the reward to the other terms: the base reward in gate mode (or zero with
`on_abstain="zero"`), `abstain_value` in score mode.

Some abstentions can be provoked by the answer itself: a launch too large for the race check
(`races_unchecked`), inline C++ or CUDA (`native_code`), an output dtype the judge cannot compare
under the v3 rule (`compare_dtype`), a run that does not finish in `timeout_s`. In gate mode with
`on_abstain="keep"` such an answer keeps its base reward, so a policy that learns to provoke one
escapes the gate. Count the abstentions per step (the verl adapter logs `ttsem_abstain`), and use
`on_abstain="zero"` where an abstention should not pay, at the price of zeroing the abstentions
that are not the answer's doing.

## The API

### `Judge`

`Judge(*, cache_dir, python=None, workers=None, mode="gate", tiers=("scale", "medium"),
contract="operator", rule="kernelbench", precision="fp32", gpu="", timeout_s=120, cap=16,
on_abstain="keep", abstain_value=None, fail_value=0.0, strict_fs=True, real_max=2**20,
ttsem_root=None, mem_gb=None)`

| argument | meaning |
|---|---|
| `cache_dir` | where the tasks, the answers, the reference runs and every pass's record are kept (see "Cache") |
| `python` | the interpreter of the judge's venv (default: this interpreter); see "The worker" |
| `workers` | passes the worker runs at once (default: the CPU count) |
| `mode` | `gate` or `cpu` |
| `tiers` | `scale` always, then any of `medium`, `real` |
| `contract` | `operator` (default) or `task`, see "Shapes" |
| `rule`, `precision`, `gpu` | the judge's own: `kernelbench` (`allclose` at 1e-2) or `v3` (the KernelBench-v3 release's rule); `fp32` casts every tensor input to float32 as KernelBench does, `keep` leaves the task's dtypes; `gpu` names the device whose shared-memory limit applies. `score` takes them per call too |
| `timeout_s` | wall seconds for each pass's run |
| `cap` | the block size the race cap uses; `None` turns the cap off |
| `on_abstain`, `abstain_value`, `fail_value` | see "Gate mode and score mode" |
| `strict_fs` | the sandbox shows a run only what it needs (the judge's `--strict`); on by default |
| `real_max` | largest tensor, in elements, the `real` tier will run |
| `ttsem_root` | a checkout of ttsem the worker imports ahead of the venv's own |
| `mem_gb` | the address-space cap of every run (`TTSEM_MEM_GB` of the judge) |

Methods:

- `score(reference_src, answer_src, **overrides) -> Verdict`: one answer against one task's
  reference. `answer_src` may be `None` or empty: that is `error` / `no_code`, scored 0.0 without a
  run. `overrides`: `rule`, `precision`, `gpu`, `format` (`kernelbench` or `tritonbench`),
  `contract`. A repeated `(task, answer)` pair is judged once and read back; a failure of the
  worker itself (it died, a request raised) is an abstention with `kind="tool_error"`, not an
  exception. Two things do raise: an unknown option or setting (`ValueError`), and a machine that
  cannot make the sandbox (`RuntimeError`, see "The sandbox").
- `score_many(items, **overrides) -> list[Verdict]` for `(reference_src, answer_src)` pairs, and
  `await ascore(...)`, which runs `score` in a thread.
- `gate(base, verdict) -> float`, above.
- `close()` ends the worker process. `Judge.from_env(**overrides)` reads the variables listed at the
  end of this page. `judge.info` holds what the worker reported on its first call (versions, the
  judge's root).

### `Verdict`

A frozen dataclass.

| field | meaning |
|---|---|
| `score` | the mode's score: 1.0, 0.0 or `None` |
| `verdict` | `verified`, `wrong`, `unsafe`, `no_kernel`, `error`, `too_slow` or `not_judged` |
| `kind` | the finer reason (the table above), or `None` |
| `feedback` | one paragraph for the model: the judge's own message (kernel, buffer, line, source line, what to check), the sizes it was found at and why those sizes count |
| `tier` | the tier that decided: `scale`, `scale_cap`, `medium`, `real` (`test`, `test_cap` for TritonBench) |
| `seconds` | CPU seconds the answer's sandboxed runs cost, the references' excluded |
| `record` | the judge's full record of every pass, for logs |

`Verdict.abstain` is `score is None`; `Verdict.hack` is true for `tampered` and `forged`, which a
policy that learns to attack the checker produces first: log it per step.

### `extract_answer(completion) -> str | None`

The answer's code in a model's completion: the last fenced Python block that defines `ModelNew`,
else the last fenced Python block, else the whole completion if it is Python that defines
`ModelNew`; `None` when there is none.

### `scores(verdict, kind)` and `kind_of(record, fmt="kernelbench")`

The table above as functions, for code that has the judge's record and not a `Verdict`.

## Shapes: what a verdict at a small size means

The judge runs the task's program with every size constant divided by the same power of two, so
that an interpreter finishes, and the reward builds its tiers on that.

1. `scale`: the largest tensor is shrunk to at most 2**13 elements, one trial, small enough that
   the race check can usually cover every launch.
2. `medium`, only after `scale` verified: 2**17 elements. It catches what grows with the shape.
3. `real`, optional (`tiers=("scale", "medium", "real")`): the task's own sizes, when its largest
   tensor holds at most `real_max` elements.

`contract="operator"` (default) holds an answer to compute the operator at the task's shape and at
the scaled ones, so a kernel specialised to the benchmark's shape (no bound on a dimension that is
a multiple of the block there) is rejected, with feedback that says the sizes. That is a choice
about what to pay for and it changes the reward. `contract="task"` holds it to the task's shape
only: a rejection at a scaled shape stands only if the task's shape confirms it, and where that
shape is too large to run the verdict is an abstention (`not_at_task_shape`).

### The cap

A kernel launched as one program instance has nothing to race with. When `scale` verifies and the
answer has a launch whose grid is a ceiling division over a block constant (`cap.py` lists the
forms it recognises), the pass runs again on a copy whose blocks are capped at `cap`, so that the
small test launches many instances; only an `unsafe` verdict of that copy counts (`scale_cap`). A
launch whose autotune configs all list blocks above the cap is not capped, since no config the
answer's own tuner can pick launches a block that small. The cap is insurance against an optimiser
that learns a race, not a claim that models write them.

## The verl adapter

`ttsem.reward.verl_adapter` is a custom reward function in the shape verl documents: four
arguments in, a dict with a `score` key out. `ground_truth` is the task's reference module as a
string. The judge is configured by environment variables ([the list](#environment-variables)), so
the trainer needs no code of ours.

Score mode, the reward being the judge's:

```yaml
reward:
  custom_reward_function:
    path: /path/to/site-packages/ttsem/reward/verl_adapter.py   # python -c "import ttsem.reward.verl_adapter as m; print(m.__file__)"
    name: compute_score          # acompute_score under verl's async reward loop
```

That is the key verl reads from 0.7.1 on (verl 0.9.1 was the latest release when this was
written); in 0.6 it was a top-level `custom_reward_function`, and a trainer on 0.6 needs the old
form.

Gate mode over a reward of your own, in a file of yours that `custom_reward_function.path` names:

```python
from ttsem.reward.verl_adapter import gated
from my_rewards import compute_score as base     # the GPU reward, verl's signature

compute_score = gated(base)                       # gated(base, lazy=True): judge only what base accepts
```

By default the base reward and the judge run at once; with `lazy=True` the judge runs only when the
base is above `fail_value`, which saves CPU and adds the two latencies.

Every call returns `{"score": float, "ttsem_verdict", "ttsem_kind", "ttsem_tier", "ttsem_abstain",
"ttsem_hack", "ttsem_seconds", "ttsem_feedback"}`, the same keys for a judged and a skipped sample
(verl stacks them across a batch); gate mode adds `base_score` and keeps the base's own keys.
`ttsem_feedback` is the paragraph for a multi-turn rollout's next turn. The judge judges one
sample per call, so use a reward manager that calls the function concurrently.

The function signatures and the dict return follow the source of verl 0.9.1; its documentation
page gives only the signature and an example that returns a number. This repository tests the
adapter against a fake worker and loads it by path as verl does, not against verl itself: check
the reward manager and the dict convention on the verl version you pin.

Two things to plan for on a real job, read from verl's source and not yet run: verl starts several
reward workers (`reward.num_workers`, 8 by default), each of which builds its own `Judge`, so the
real ceiling on concurrent passes is that number times `TTSEM_REWARD_WORKERS`; and the workers are
spread over the nodes of the job, so ttsem, the judge's venv, `bwrap` and the `TTSEM_REWARD_*`
variables must be present on each of them.

## The OpenEnv adapter

`ttsem.reward.openenv_adapter` serves the judge as an [OpenEnv](https://github.com/openenv/OpenEnv)
environment (BSD-3; `pip install "ttsem[openenv]"`, written against `openenv` 0.13.0, the extra stops
at that minor): `reset` hands out a KernelBench-style task as a prompt, `step` takes a completion,
extracts its code, judges it in score mode and returns the observation with the verdict, the
judge's paragraph and the reward.

```python
from ttsem.reward.openenv_adapter import KernelAction, KernelEnv, create_kernel_app

env = KernelEnv({"train": [reference_src, ...]}, max_attempts=3)
obs = env.reset(split="train", index=0)           # obs.prompt: what the model is shown
obs = env.step(KernelAction(completion=text))     # obs.reward, obs.verdict, obs.feedback, obs.done
app = create_kernel_app({"train": [...]}, max_concurrent_envs=8)   # uvicorn.run(app)
```

- **Reward.** The judge's CPU-only score (1.0 verified, 0.0 rejected), taken through a `Rubric`
  (`env.rubric.last_score`). The judge must be a `Judge(mode="cpu")`: a judge in gate mode is
  refused, because its score is a multiplier and abstains where score mode gives 0.0. An abstention
  is the reward `abstain_value`: the environment's, else the judge's own, else 0.0 as in the verl
  adapter. It sets `abstain` on the observation, so a trainer that can drop those samples should,
  and it uses up an attempt.
- **Attempts.** With `max_attempts` above 1 an episode lasts until a kernel is verified or the
  attempts are spent; `feedback` is what the next attempt is shown. Only the observation of
  `reset` carries the prompt: the trainer keeps it. `hack` on an observation is that step's; the
  state counts them (`hacks`).
- **Tasks.** `list_splits`, `num_tasks`, `get_task` and `get_task_range` answer OpenEnv's task
  discovery without building a judge; `reset(split=, index=, task=, seed=)` picks the task, and
  takes nothing else. `reset(task=...)` takes the reference from the caller: it runs in the same
  sandbox as an answer, but a reward from a reference the caller chose says nothing about the
  caller's answer.
- **Judge.** Pass one, or it is built once per process from the `TTSEM_REWARD_*` variables, as for
  verl; every answer and reference runs in its `bwrap` sandbox.
- **Sessions.** An episode lives in one WebSocket session (`/ws`, which OpenEnv's `EnvClient`
  uses). A server takes one session at a time unless `max_concurrent_envs` is given (OpenEnv's
  default; the second session is refused with `CAPACITY_REACHED`): parallel rollouts need it above
  1, and the sessions share the judge's worker pool, so `Judge(workers=)` is the other ceiling.
  OpenEnv's REST routes make a fresh environment for each request, so a REST `/step` fails (HTTP
  500; the server log says "step before reset").

Tested here against a fake worker, directly and through OpenEnv's app over a WebSocket (CI has a
job for it, since the tests skip without `openenv`); not in a training run, and not against an
OpenEnv version other than 0.13.0.

## The worker: the judge runs in its own venv

The judge needs Triton and torch (CPU), pinned in ttsem's `triton` extra, and a trainer usually
has its own CUDA builds of both. So the trainer's process imports neither: `Judge` starts one
long-lived worker process, `python -I .../ttsem/reward/worker.py`, with the interpreter you give
as `python` (`TTSEM_REWARD_PYTHON`), and talks to it in JSON lines over a pipe. The worker hands
each pass to `ttsem.judge`, which runs the answer and the reference each in a `bwrap` sandbox of
its own, and sends back the judge's record. A worker that dies fails its pending requests (they
become abstentions, `tool_error`) and the next call starts another; a forked reward process starts
its own.

Make the judge's venv once, with the pinned wheels:

```console
$ python -m venv ~/venvs/judge
$ ~/venvs/judge/bin/pip install --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple 'ttsem[triton]'
$ export TTSEM_REWARD_PYTHON=~/venvs/judge/bin/python
```

Make it from an interpreter that sits outside your home directory (the system Python, or one whose
real path is not under `$HOME`). The sandbox hides the home, so a venv whose `python` is a symlink
into a Python that a tool keeps there does not start inside it; `Judge` checks this on its first
call and raises instead of letting every pass abstain.

The trainer's side needs ttsem and numpy only. Keep the two ttsem installs at one version: the
worker file is the trainer's own and talks to the `ttsem.judge` of the venv. `ttsem_root`
(`TTSEM_REWARD_TTSEM_ROOT`) names a checkout the worker imports ahead of the venv's, for working on
ttsem itself. No new extra is needed: the worker's requirements are the `triton` extra.

Each pass costs CPU seconds, not GPU: the sandboxed run imports torch and Triton, runs the
reference once per task and tier (kept, see below) and runs the answer. `Verdict.seconds` is the
CPU time of the answer's runs, measured inside the sandbox on the process tree, so it stays honest
on a machine shared with other work. It is telemetry from the sandbox: an answer could forge it,
never the verdict.

### Cache

`cache_dir` holds the tasks and answers (named by the hash of their text), one reference run per
task and tier, reused across a GRPO group and across epochs, and every pass's record. It also holds
every model-written answer the run saw. Put it on disk, not on a RAM-backed `/tmp`, and clear it
when the run is done. No answer's sandbox sees it, except its own working directory.

## The sandbox

The reward never runs an answer without a sandbox and has no switch for it. It needs Linux and
[`bwrap`](https://github.com/containers/bubblewrap) (bubblewrap) on `PATH`, able to create its
namespaces: unprivileged user namespaces enabled, or root. The first call asks the worker to make
one, and raises `RuntimeError` ("the judge cannot run here") instead of judging if it cannot. The
sandbox is the judge's, described in [docs/JUDGE.md](JUDGE.md#the-sandbox): a read-only filesystem
(with `strict_fs`, only what the run needs), the home directory and the host's sockets hidden, no
network, its own pid namespace, no capabilities.

Not inside a container whose seccomp profile forbids namespaces, which is Docker's default: `bwrap`
cannot start there. Run the judge on the host, or start the container with a profile that allows
them. Switching the sandbox off is not an answer: the answer's process could then overwrite the
reference's outputs on disk and turn `wrong` into `verified`.

## What it does not catch

- **The judge's own caveat applies unchanged.** `wrong` versus `verified` holds against an
  adversarial answer. `unsafe` and `no_kernel` rest on a source scan and a run-time integrity
  check, not on a separation of the answer from the checker, which this release does not have
  ([docs/JUDGE.md](JUDGE.md#what-the-verdicts-guarantee)). Against an optimiser free to write
  anything, they bound the damage (gate mode never raises a reward) and do not prove anything.
  Every `tampered` and `forged` verdict is counted as a hack attempt.
- **Speed.** Nothing is timed. The reward says a kernel is right and safe, not fast; timing hacks
  are the job of the GPU reward it gates.
- **The device.** Triton's IR is interpreted on a CPU. Below the LLVM conversion (ptxas, the
  hardware, the order of atomic updates, fast-math, a tensor core's accumulation) nothing runs, so a
  kernel right here can differ on a device, and the comparison tolerance of the rule decides the
  rest.
- **Sizes that did not run.** A verdict holds for the inputs and sizes that ran. With the default
  tiers the task's own shape is not run; a fault only that shape reaches is missed, and under
  `contract="task"` a kernel wrong only at a scaled shape is let through.
- **Races in large launches.** Races are looked for between program instances, and past a limit on
  the accesses of one launch they cannot be: such a launch is an abstention (`races_unchecked`),
  not a pass.
- **Anything outside Triton.** Inline C++ or CUDA is not run (`native_code`), and inline PTX and the
  operations the semantics does not model stop as `unsupported`, by name: abstentions, not
  verdicts.
- **A compile error under the judge's Triton.** It may be drift between the trainer's Triton and
  the judge's, not the answer's fault: gate mode abstains, and score mode scores 0.0, which is
  where score mode risks rejecting a good answer.
- **The v3 rule and an output of another dtype.** The judge's comparison calls `torch.allclose`
  under every rule, which raises when the answer's output dtype is not the reference's, where the
  v3 harness casts both to float32 first. Under `rule="v3"` that is an abstention
  (`compare_dtype`), so an answer can return another dtype to escape a gate with
  `on_abstain="keep"`; under `rule="kernelbench"` the error stands, as KernelBench's own comparison
  raises it.
- **Tasks the precision cast breaks.** `precision="fp32"` casts every tensor input to float32, which
  breaks a reference that takes integer indices (gather, embedding): its run fails and the pass is
  an abstention. Use `precision="keep"` for those.
- **What it does to training is not shown here.** The rules are tested; whether a policy trained on
  them improves, or learns around them, is not.

## Environment variables

`Judge.from_env()`, the verl adapter and the OpenEnv adapter read:

| variable | argument |
|---|---|
| `TTSEM_REWARD_CACHE` | `cache_dir` (default `~/.cache/ttsem_reward`) |
| `TTSEM_REWARD_PYTHON` | `python` |
| `TTSEM_REWARD_TTSEM_ROOT` | `ttsem_root` |
| `TTSEM_REWARD_WORKERS` | `workers` |
| `TTSEM_REWARD_MODE` | `mode` (the adapter sets it per function) |
| `TTSEM_REWARD_CONTRACT` | `contract` |
| `TTSEM_REWARD_CAP` | `cap`; `0` turns it off |
| `TTSEM_REWARD_TIERS` | `tiers`, comma-separated |
| `TTSEM_REWARD_TIMEOUT` | `timeout_s` |
| `TTSEM_REWARD_RULE`, `TTSEM_REWARD_PRECISION`, `TTSEM_REWARD_GPU` | `rule`, `precision`, `gpu` |
| `TTSEM_REWARD_MEM_GB` | `mem_gb` |

`on_abstain`, `abstain_value`, `fail_value`, `strict_fs` and `real_max` are not read from the
environment: pass them to `Judge.from_env(...)`.

## Tests

`pytest tests/test_reward*.py` runs the unit tests with no GPU and no Triton (the OpenEnv adapter's
need the `openenv` extra and skip without it), and the integration
tests where the judge's interpreter has torch and Triton and `bwrap` works (they skip otherwise).
`tests/test_reward_hygiene.py` fails if the package or this page names a path of the machine it was
written on or anything that does not belong in public text.
