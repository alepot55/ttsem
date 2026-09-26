"""`judge`: KernelBench answers judged without a GPU, each in its own sandboxed process.

The four answers under `fixtures/judge/` are the four outcomes a model-written kernel most often
has: right, wrong by value, right by value but reading past the end of its input, and no kernel
at all. They run through the judge's own `bwrap` sandbox like any answer, so those tests skip
where the sandbox cannot run.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ttsem import judge, judge_shapes

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "judge"
TASK = FIXTURES / "task_relu.py"
ROOT = Path(__file__).resolve().parent.parent

needs_sandbox = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("triton") is None
    or judge.sandbox_problem() is not None,
    reason="needs torch, Triton and a working bwrap",
)


@pytest.fixture(scope="module")
def batch(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The four answers judged once, as a manifest, two at a time. The manifest's paths are
    relative to its own directory."""
    here = tmp_path_factory.mktemp("judge")
    manifest = here / "manifest.jsonl"
    lines = []
    for name in ("ok", "wrong", "unsafe", "no_kernel"):
        answer = os.path.relpath(FIXTURES / f"answer_{name}.py", here)
        item = {"id": name, "task": os.path.relpath(TASK, here), "answer": answer, "label": "x"}
        lines.append(json.dumps(item))
    manifest.write_text("\n".join(lines) + "\n")
    out = here / "out"
    done = subprocess.run(
        [sys.executable, "-m", "ttsem", "judge", "--manifest", str(manifest), "--out", str(out)]
        + ["--jobs", "2"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]
    return {"done": done, "rows": {row["id"]: row for row in rows}}


@needs_sandbox
def test_a_correct_kernel_is_verified(batch: dict[str, Any]) -> None:
    row = batch["rows"]["ok"]
    assert row["scaled"]["verdict"] == "verified", row
    assert "full" not in row  # 4000 elements: the task is judged at its real shape at once
    assert len(row["scaled"]["trials"]) == 5  # KernelBench's five trials
    assert all(t["launches"] == 1 and t["max_abs_diff"] == 0 for t in row["scaled"]["trials"])
    assert row["label"] == "x"  # what the manifest carries rides along


@needs_sandbox
def test_a_wrong_value_is_wrong_in_the_first_trial(batch: dict[str, Any]) -> None:
    report = batch["rows"]["wrong"]["scaled"]
    assert report["verdict"] == "wrong", report
    trial = report["trials"][-1]
    assert len(report["trials"]) == 1 and not trial["pass"] and not trial["allclose"]
    assert trial["max_abs_diff"] > 0.1  # a tenth of the most negative input
    assert judge.describe(report).startswith("wrong in trial 1 (seed ")


@needs_sandbox
def test_a_load_without_its_mask_is_unsafe_at_the_answers_own_line(
    batch: dict[str, Any],
) -> None:
    report = batch["rows"]["unsafe"]["scaled"]
    assert report["verdict"] == "unsafe", report
    assert report["kind"] == "oob_read" and report["kernel"] == "relu_kernel"
    assert report["buffer"] == "x_ptr"
    lines = (FIXTURES / "answer_unsafe.py").read_text().splitlines()
    assert lines[report["line"] - 1].strip() == report["source"] == "x = tl.load(x_ptr + offs)"
    assert report["file"] == "answer_unsafe.py"
    assert "96 element(s) past the end" in report["message"]


@needs_sandbox
def test_pytorch_doing_the_work_is_no_kernel(batch: dict[str, Any]) -> None:
    report = batch["rows"]["no_kernel"]["scaled"]
    assert report["verdict"] == "no_kernel", report
    assert report["launches"] == 0 and report["torch_ops"] == {"relu": 5}


@needs_sandbox
def test_the_batch_prints_a_line_per_answer_and_the_counts(batch: dict[str, Any]) -> None:
    done = batch["done"]
    assert done.returncode == 0, done.stderr
    out = done.stdout.splitlines()
    assert out[0].startswith("ok: verified: 5 trial(s)")
    assert out[-1] == "scaled pass, 4 answer(s): verified 1, wrong 1, unsafe 1, no_kernel 1"


@needs_sandbox
def test_one_pair_prints_the_fault_and_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    code = judge.main([str(TASK), str(FIXTURES / "answer_unsafe.py")])
    assert code == 1
    line = capsys.readouterr().out.strip()
    assert line == (
        "unsafe: out-of-bounds read in kernel `relu_kernel` (`x_ptr`), "
        "answer_unsafe.py:16: x = tl.load(x_ptr + offs)  [real sizes]"
    )


@needs_sandbox
def test_one_pair_as_json_through_the_package_entry_point(tmp_path: Path) -> None:
    """From another directory, as an installed package is run: the sandboxed process imports
    the same `ttsem` the command was started from."""
    done = subprocess.run(
        [sys.executable, "-m", "ttsem", "judge", str(TASK), str(FIXTURES / "answer_ok.py")]
        + ["--json", "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=False,
    )
    assert done.returncode == 0, done.stderr
    row = json.loads(done.stdout)
    assert row["scaled"]["verdict"] == "verified" and row["rule"] == "kernelbench"
    assert (tmp_path / "answer_ok.scaled.json").exists()


def test_without_bwrap_the_judge_refuses_to_run_anything(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(judge.shutil, "which", lambda name: None)
    started: list[Any] = []
    monkeypatch.setattr(judge, "judge_one", lambda *a, **k: started.append(a))
    assert judge.main([str(TASK), str(FIXTURES / "answer_ok.py")]) == 2
    err = capsys.readouterr().err
    assert "bwrap is not on PATH" in err and "--no-sandbox" in err
    assert not started


def test_no_sandbox_warns_before_anything_runs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing is executed here: the pass is replaced, and only how it was called is checked."""
    calls: list[dict[str, Any]] = []

    def fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append({"args": args, **kwargs})
        return {"verdict": "verified", "trials": []}

    monkeypatch.setattr(judge, "judge_one", fake)
    assert judge.main([str(TASK), str(FIXTURES / "answer_ok.py"), "--no-sandbox"]) == 0
    assert "about to run unsandboxed" in capsys.readouterr().err
    assert [call["args"][-1] for call in calls] == [False]  # sandbox=False, one pass


def test_each_kind_of_verdict_reads_as_one_line() -> None:
    assert judge.describe({"verdict": "too_slow", "seconds": 600}) == (
        "too_slow: no verdict within 600 s"
    )
    error = {"verdict": "error", "stage": "load", "message": "NameError: x", "where": "a.py:3"}
    assert judge.describe(error) == "error (load): NameError: x at a.py:3"
    unjudged = {"verdict": "not_judged", "why": "unsupported", "message": "tt.foo"}
    assert judge.describe(unjudged) == "not_judged (unsupported): tt.foo"
    failed = {"index": 2, "config": "BLOCK: 64", "max_abs_diff": float("nan"), "ref_nan": False}
    config = {"verdict": "wrong", "trials": [{"pass": True}], "configs": {"failed": failed}}
    assert judge.describe(config) == (
        "wrong under autotune config 2 (BLOCK: 64): max abs diff nan "
        "(NaN: the output holds memory nobody wrote)"
    )
    shape = {"verdict": "wrong", "trials": [{"pass": False, "seed": 7, "shape": [[4], [2]]}]}
    assert (
        judge.describe(shape)
        == "wrong in trial 1 (seed 7): output shape [2], the reference's is [4]"
    )


def test_scaling_divides_every_size_by_one_power_of_two_and_keeps_the_lines() -> None:
    pytest.importorskip("torch")
    source = TASK.read_text().replace("batch_size = 4\n", "batch_size = 4096\n")
    values, note = judge_shapes.scaled(source)
    assert note["footprint_full"] == 4096 * 1000 and note["factor"] == 8
    assert values == {"batch_size": 512, "dim": 125}  # like 1000, not a multiple of 16
    assert note["footprint"] == 512 * 125 <= judge_shapes.BUDGET
    small = judge_shapes.rewrite(source, values)
    assert small.count("\n") == source.count("\n")
    assert "batch_size = 512\n" in small and "dim = 125\n" in small


@pytest.fixture(scope="module")
def edges(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The answers where the judge once blamed the wrong party, judged once as a manifest."""
    here = tmp_path_factory.mktemp("edges")
    manifest = here / "manifest.jsonl"
    pairs = {
        "nested_grid": "task_relu.py",
        "no_config_fits": "task_matmul.py",
        "config_asserts": "task_relu.py",
        "config_faults": "task_relu.py",
    }
    lines = [
        json.dumps(
            {
                "id": name,
                "task": str(FIXTURES / task),
                "answer": str(FIXTURES / f"answer_{name}.py"),
            }
        )
        for name, task in pairs.items()
    ]
    manifest.write_text("\n".join(lines) + "\n")
    out = here / "out"
    done = subprocess.run(
        [sys.executable, "-m", "ttsem", "judge", "--manifest", str(manifest), "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]
    return {row["id"]: row["scaled"] for row in rows}


@needs_sandbox
def test_a_grid_the_launcher_refuses_is_an_error_of_the_answer(edges: dict[str, Any]) -> None:
    report = edges["nested_grid"]
    assert report["verdict"] == "error" and "why" not in report, report
    assert report["message"] == "TypeError: 'tuple' object cannot be interpreted as an integer"
    assert report["where"] == "answer_nested_grid.py:27"  # the launch, not the check's own line


def test_the_grid_checks_are_the_launchers() -> None:
    """`JITFunction.run` indexes the grid, the C launcher parses three entries as C ints."""
    pytest.importorskip("triton")
    from ttsem import harness

    harness.check_launch_grid((1,))
    harness.check_launch_grid((2**31 - 1, 1, -(2**31), "not an entry the launcher reads"))
    with pytest.raises(TypeError):
        harness.check_launch_grid(4)
    with pytest.raises(IndexError):
        harness.check_launch_grid(())
    with pytest.raises(TypeError, match="'float' object cannot be interpreted as an integer"):
        harness.check_launch_grid((1, 2.0))
    with pytest.raises(OverflowError, match="^signed integer is greater than maximum$"):
        harness.check_launch_grid((2**31,))
    with pytest.raises(OverflowError, match="^signed integer is less than minimum$"):
        harness.check_launch_grid((1, 1, -(2**31) - 1))
    with pytest.raises(OverflowError, match="^Python int too large to convert to C long$"):
        harness.check_launch_grid((2**64,))


@needs_sandbox
def test_no_autotune_config_that_fits_the_gpu_is_an_error_of_the_answer(
    edges: dict[str, Any],
) -> None:
    """Every config is skipped, as on an H100 (the default GPU), and the launch with the one the
    autotuner falls back to fails there: the answer's error, not the tool's."""
    report = edges["no_config_fits"]
    assert report["verdict"] == "error" and report["why"] == "shared_memory", report
    assert report["shared"] == 262144 and report["limit"] == 232448
    assert report["where"] == "answer_no_config_fits.py:41"
    assert len(report["autotune"][0]["skipped"]) == 2
    assert judge.describe(report).startswith(
        "error (shared_memory): every autotune config of `matmul_kernel` was skipped, and the one "
        "it falls back to needs 262,144 B of shared memory, over the H100 (the default GPU)'s "
        "232,448 B (Triton 3.8.0's count)"
    )


@needs_sandbox
def test_a_config_that_does_not_compile_is_not_applicable_as_the_autotuner_skips_it(
    edges: dict[str, Any],
) -> None:
    report = edges["config_asserts"]
    assert report["verdict"] == "verified", report
    skipped = report["configs"]["not_applicable"]
    assert [(c["index"], c["reason"]) for c in skipped] == [(0, "compile")]
    assert "static_assert(BLOCK <= 1024)" in skipped[0]["detail"]
    assert "forcing" not in report["configs"]
    # what the trials ran with, not the config the sweep forced last
    assert report["autotune"][0]["ran"].startswith("BLOCK: 256,")


@needs_sandbox
def test_a_fault_under_a_config_the_sweep_forces_names_that_config(
    edges: dict[str, Any],
) -> None:
    report = edges["config_faults"]
    assert report["verdict"] == "unsafe", report
    assert all(t["pass"] for t in report["trials"])  # the first config is right
    assert report["kind"] == "oob_read" and report["kernel"] == "gather_relu_kernel"
    assert report["source"] == "x = tl.load(x_ptr + idx, mask=mask)"
    assert report["configs"]["forcing"] == {
        "index": 1,
        "config": report["autotune"][0]["ran"],
    }
    assert report["autotune"][0]["ran"].startswith("BLOCK: 128,")  # the config that faulted
    assert " under autotune config 1 (BLOCK: 128, " in judge.describe(report)


def test_the_configs_skipped_are_those_the_installed_autotuner_skips() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    from triton.runtime.autotuner import Autotuner

    from ttsem import _judge_runner as runner

    caught = re.search(r"except \(([^)]*)\) as", inspect.getsource(Autotuner._bench))
    assert caught is not None
    names = {name.strip() for name in caught.group(1).split(",")}
    assert names == {e.__name__ for e in runner.CONFIG_ERRORS}
