"""The corpus driver keeps the runners' failures instead of counting them silently."""

from __future__ import annotations

from ttsem import validate_corpus


def test_a_failed_runner_is_kept_with_its_last_stderr_line() -> None:
    reports = [
        {
            "program": "ok.py",
            "launches": [
                {
                    "passes": [{"pass_name": "Before A: a", "verdict": "match"}],
                    "first_bad_pass": None,
                }
            ],
        },
        {
            "program": "oom.py",
            "error": (
                "Traceback (most recent call last):\n  ...\n"
                "torch.AcceleratorError: CUDA error: out of memory\n"
            ),
        },
        {"program": "slow.py", "error": "timeout"},
    ]
    summary = validate_corpus.summarize(reports)
    assert summary["programs"] == 3
    assert summary["errors"] == 2
    assert summary["per_pass"] == {"Before A: a": {"match": 1}}
    assert summary["error_programs"] == {
        "oom.py": "torch.AcceleratorError: CUDA error: out of memory",
        "slow.py": "timeout",
    }


def test_an_empty_message_still_names_the_program() -> None:
    summary = validate_corpus.summarize([{"program": "p.py", "error": ""}])
    assert summary["error_programs"] == {"p.py": "unknown error"}
