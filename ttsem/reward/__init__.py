"""ttsem.reward: the exact judge as a reward for kernel RL (new in 0.2, prototype-grade).

`Judge` scores a model's answer against a task's reference by running both in the judge's own
sandboxed processes; `Verdict` is what it returns, `extract_answer` takes the code out of a
completion, `kind_of` and `scores` are the table from a judge's record to a reward. The core is
`core.py`; the cap that exposes races between program instances `cap.py`; the verl adapter
`verl_adapter.py`; the OpenEnv adapter (optional, needs the `openenv` extra) `openenv_adapter.py`;
the worker, which runs in the judge's own venv, `worker.py`. See
`docs/REWARD.md`: what it checks, what it does not, and what it needs.

Importing this package imports neither torch nor Triton.
"""

from ttsem.reward.core import Judge, Verdict, extract_answer, kind_of, scores

__all__ = ["Judge", "Verdict", "extract_answer", "kind_of", "scores"]
