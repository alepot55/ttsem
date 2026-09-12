"""The demo, and the one line of it that CI is there to pin.

`python -m ttsem demo` ends by naming the pass the recorded verdicts point at. That name is
not written anywhere in `demo.py`: it comes out of `ttsem.validate.culprit_of` run over the
recording. If the recording is edited, or either reading of a stage list changes, this test is
what says so.
"""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ttsem import demo

CULPRIT_LINE = (
    "first pass that changes the meaning: tritongpu-fuse-nested-loops "
    "(triton#11519, fixed upstream in #11521)"
)


def _run(argv: list[str]) -> tuple[int, list[str]]:
    out = io.StringIO()
    code = demo.main(argv, out=out)
    return code, out.getvalue().splitlines()


def test_the_default_demo_names_the_culprit_on_its_last_line() -> None:
    code, lines = _run([])
    assert code == 0
    assert lines[-1] == CULPRIT_LINE


def test_the_culprit_is_computed_from_the_recording_and_not_written_down() -> None:
    """The name in the last line must not be a literal anywhere in the module."""
    source = Path(demo.__file__).read_text()
    assert CULPRIT_LINE not in source


def test_the_demo_prints_one_line_per_group_of_consecutive_equal_verdicts() -> None:
    _code, lines = _run([])
    stage_lines = [line for line in lines if line.lstrip().startswith("stages ")]
    launch = json.loads(demo.DEFAULT_RECORDING.read_text())["launches"][0]
    groups = demo.group_stages(demo.stages_of(launch))
    assert len(stage_lines) == len(groups)
    assert sum(g.last - g.first + 1 for g in groups) == len(launch["passes"])


def test_every_group_line_carries_its_verdict_and_its_differing_elements() -> None:
    _code, lines = _run([])
    mismatch = [line for line in lines if " mismatch " in line]
    assert len(mismatch) == 1
    assert "n_diff=320" in mismatch[0]
    assert "tritongpu-fuse-nested-loops" in mismatch[0]


def test_short_name_keeps_the_pipeline_name_and_drops_the_options() -> None:
    assert (
        demo.short_name("Before TritonGPUFuseNestedLoops: tritongpu-fuse-nested-loops")
        == "tritongpu-fuse-nested-loops"
    )
    canonicalize = "Before CanonicalizerPass: canonicalize{top-down=true}"
    assert demo.short_name(canonicalize) == "canonicalize"
    assert demo.short_name("Before P: pipeline{n=3} (step 1)") == "pipeline (step 1)"


def test_a_recording_whose_stages_all_agree_names_no_pass() -> None:
    stages = [demo.Stage("Before A: a", "match", 0), demo.Stage("Before B: b", "match", 0)]
    assert demo.culprit(stages) is None
    assert demo.verdict_line(None).startswith("no pass changes the meaning")


def test_a_pass_with_no_public_bug_number_is_named_without_a_citation() -> None:
    stages = [
        demo.Stage("Before A: a", "match", 0),
        demo.Stage("Before B: some-other-pass", "mismatch", 7),
    ]
    assert demo.verdict_line(demo.culprit(stages)) == "first pass that changes the meaning: a"


def test_a_missing_recording_is_an_error_and_not_a_traceback() -> None:
    code, lines = _run(["--recording", "/nonexistent/recording.json"])
    assert code == 1
    assert "no recording" in lines[0]


def test_live_without_the_wheel_explains_itself_instead_of_failing_to_import() -> None:
    """Without the wheel `--live` is a paragraph of instructions, not an ImportError."""
    if importlib.util.find_spec("triton") is not None:
        pytest.skip("the wheel is installed here: --live is the slow CI job, not this one")
    code, lines = _run(["--live"])
    assert code == 1
    assert "needs the Triton wheel" in lines[0]


def test_the_module_entry_point_routes_demo() -> None:
    """`python -m ttsem demo` and `python -m ttsem.demo` are the same program."""
    result = subprocess.run(
        [sys.executable, "-m", "ttsem", "demo"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0
    assert result.stdout.rstrip().splitlines()[-1] == CULPRIT_LINE


def test_an_unknown_subcommand_is_refused() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ttsem", "nonsense"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 2
    assert "unknown command" in result.stderr
