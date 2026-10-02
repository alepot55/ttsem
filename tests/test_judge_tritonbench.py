"""`judge --tritonbench`: TritonBench answers judged without a GPU, by value against the task's
own reference run, each file in its own sandboxed process.

The answers under `fixtures/tritonbench/` are the outcomes that matter: right, wrong by value,
right by value but reading past the end of its input, no kernel at all, and a file that raises;
one more looks for the reference's results and brings a test of its own, and must be judged as if
it did neither, and one tries to read its task's file and leaves a module and a cache entry
behind for the runs after it. The runs go through the judge's own `bwrap` sandbox like any
answer, so those tests skip where the sandbox cannot run; the verdict rules and the columns are
checked in this process on runs written by hand.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import Planted

from ttsem import judge
from ttsem import judge_tritonbench as tb

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "tritonbench"
TASK = FIXTURES / "task_relu.py"
ROOT = Path(__file__).resolve().parent.parent
ANSWERS = ("ok", "wrong", "unsafe", "no_kernel", "raises", "peeks")

needs_sandbox = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("triton") is None
    or judge.sandbox_problem() is not None,
    reason="needs torch, Triton and a working bwrap",
)


def judge_manifest(
    here: Path, names: tuple[str, ...], jobs: int
) -> subprocess.CompletedProcess[str]:
    """The answers `names` of `TASK` judged as a manifest into here/out; the manifest's paths
    are relative to its own directory."""
    manifest = here / "manifest.jsonl"
    lines = [
        json.dumps(
            {
                "id": name,
                "task": os.path.relpath(TASK, here),
                "answer": os.path.relpath(FIXTURES / f"answer_{name}.py", here),
                "model": "x",
            }
        )
        for name in names
    ]
    manifest.write_text("\n".join(lines) + "\n")
    return subprocess.run(
        [sys.executable, "-m", "ttsem", "judge", "--tritonbench", "--manifest", str(manifest)]
        + ["--out", str(here / "out"), "--jobs", str(jobs)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )


def rows_of(out: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]
    return {row["id"]: row for row in rows}


@pytest.fixture(scope="module")
def batch(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The answers judged once, as a manifest, two at a time."""
    here = tmp_path_factory.mktemp("tritonbench")
    done = judge_manifest(here, ANSWERS, jobs=2)
    assert done.returncode == 0, done.stderr
    return {"done": done, "out": here / "out", "rows": rows_of(here / "out")}


def record(batch: dict[str, Any], name: str) -> dict[str, Any]:
    return dict(batch["rows"][name][tb.RECORD])


@needs_sandbox
def test_a_correct_answer_is_verified(batch: dict[str, Any]) -> None:
    got = record(batch, "ok")
    assert (got["verdict"], got["class"], got["tolerance"]) == ("verified", "verified", "strict")
    assert got["launches"] == 2
    assert batch["rows"]["ok"]["model"] == "x"  # the manifest's labels are carried along


@needs_sandbox
def test_an_absolute_value_is_wrong_and_says_where(batch: dict[str, Any]) -> None:
    got = record(batch, "wrong")
    assert (got["verdict"], got["class"]) == ("wrong", "wrong_result"), got
    assert got["message"].startswith("test_case_1: ") and got["message"].endswith("values differ")


@needs_sandbox
def test_a_load_without_its_mask_is_unsafe_at_the_answers_own_line(batch: dict[str, Any]) -> None:
    got = record(batch, "unsafe")
    assert (got["verdict"], got["class"]) == ("unsafe", "oob_read"), got
    assert (got["file"], got["line"]) == ("answer_unsafe.py", 9)
    assert tb.describe(got) == (
        "unsafe: out-of-bounds read in kernel `relu_kernel` (`x_ptr`), answer_unsafe.py:9: "
        "x = tl.load(x_ptr + offs)"
    )


@needs_sandbox
def test_pytorch_doing_the_work_is_no_kernel(batch: dict[str, Any]) -> None:
    got = record(batch, "no_kernel")
    assert (got["verdict"], got["class"], got["launches"]) == ("no_kernel", "verified_no_kernel", 0)


@needs_sandbox
def test_an_answer_that_raises_is_an_error_at_its_line(batch: dict[str, Any]) -> None:
    got = record(batch, "raises")
    assert (got["verdict"], got["class"]) == ("error", "call_error"), got
    assert got["message"].startswith("CompilationError: ")
    assert "has no attribute 'relu'" in got["detail"]
    assert got["where"] == "answer_raises.py:17"  # the launch


@needs_sandbox
def test_the_references_are_out_of_sight_and_the_tasks_test_is_the_one_run(
    batch: dict[str, Any],
) -> None:
    got = record(batch, "peeks")
    assert got["verdict"] == "verified", got
    built = (batch["out"] / f"peeks.{tb.RECORD}.py").read_text()
    assert "a test of the answer's own" not in built and "result_gold = test_relu()" in built
    assert len(list((batch["out"] / "reference").glob("task_relu-*.json"))) == 2  # both poisons


@needs_sandbox
def test_the_batch_prints_a_line_per_answer_and_the_columns(batch: dict[str, Any]) -> None:
    text = batch["done"].stdout
    assert "unsafe: out-of-bounds read in kernel `relu_kernel`" in text
    assert "6 answer(s): verified 2, wrong 1, unsafe 1, no_kernel 1, error 1" in text
    assert (
        "columns: answers 6, verified 2, reduced 0, wrong 1, unsafe 1, "
        "not_compared 1, not_judged 0, ran 5"
    ) in text
    assert (
        "of the 4 that ran and could be judged, 2 (50%) return wrong values or are unsafe" in text
    )


@needs_sandbox
def test_an_answer_cannot_read_its_task_nor_reach_the_runs_after_it(tmp_path: Path) -> None:
    """`cheats` runs first: its task's file is out of its sight (so it stays wrong), and neither
    the module it leaves in OUT nor its cache entry reaches the reference's second run or the
    answer judged after it, which run on the references' own cache."""
    done = judge_manifest(tmp_path, ("cheats", "ok"), jobs=1)
    assert done.returncode == 0, done.stderr
    out, rows = tmp_path / "out", rows_of(tmp_path / "out")
    cheats = rows["cheats"][tb.RECORD]
    assert (cheats["verdict"], cheats["class"]) == ("wrong", "wrong_result"), cheats
    assert rows["ok"][tb.RECORD]["verdict"] == "verified"
    assert (out / "ttsem" / "__init__.py").exists() and (out / ".cache" / "planted").exists()
    assert not (out / "planted_ttsem_imported").exists()  # no later run imported it
    assert len(list((out / "reference").glob("task_relu-*.json"))) == 2  # the second ran after
    assert any((out / "reference" / ".cache").iterdir())
    assert not (out / "reference" / ".cache" / "planted").exists()


@needs_sandbox
def test_one_pair_prints_its_verdict_and_exits_as_it_says(tmp_path: Path) -> None:
    def run(name: str, *extra: str) -> subprocess.CompletedProcess[str]:
        answer = FIXTURES / f"answer_{name}.py"
        return subprocess.run(
            [sys.executable, "-m", "ttsem", "judge", "--tritonbench", str(TASK), str(answer)]
            + ["--out", str(tmp_path), *extra],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )

    done = run("ok", "--json")
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)[tb.RECORD]["verdict"] == "verified"
    done = run("wrong")
    assert done.returncode == 1
    assert done.stdout.startswith("wrong: the test's results differ from the reference's: ")


@needs_sandbox
def test_an_answer_sees_no_secret_of_the_caller(planted: Planted, tmp_path: Path) -> None:
    """An answer that copies whatever it can reach of a secret the caller holds, in its
    environment and in its home's config, into OUT, then is judged as any other."""
    answer = tmp_path / "answer_leak.py"
    answer.write_text(planted.leak() + (FIXTURES / "answer_ok.py").read_text())
    out = tmp_path / "out"
    out.mkdir()
    item = {"id": "leak", "task": str(TASK), "answer": str(answer)}
    got = tb.judge_item(item, out, 300, sandbox=True)[tb.RECORD]
    assert got["verdict"] == "verified", got
    planted.check(out)


def test_kernelbench_flags_are_refused_with_tritonbench(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        judge.main(["--tritonbench", str(TASK), str(FIXTURES / "answer_ok.py"), "--rule", "v3"])
    assert "--rule: KernelBench only, not with --tritonbench" in capsys.readouterr().err


def test_a_task_without_the_separator_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Taken whole as the test, the task would redefine the answer's functions after it and
    compare the reference with itself: one pair exits 2, a manifest line is not judged."""
    task = tmp_path / "task.py"
    task.write_text(TASK.read_text().replace(tb.SEP, "#"))
    answer = FIXTURES / "answer_wrong.py"
    argv = ["--tritonbench", str(task), str(answer), "--no-sandbox", "--out", str(tmp_path)]
    assert judge.main(argv) == 2
    assert "no TritonBench separator (a line of 146 '#')" in capsys.readouterr().err
    item = {"id": "a", "task": str(task), "answer": str(answer)}
    got = tb.judge_item(item, tmp_path, 10, sandbox=False)[tb.RECORD]
    assert (got["verdict"], got["class"]) == ("not_judged", "bad_task")
    assert not (tmp_path / "reference").exists()  # nothing ran


def results(
    tmp_path: Path, name: str, value: Any, launches: int = 1
) -> tuple[dict[str, Any], Path]:
    """A run report as the runner writes it, for a test that kept `{"case": value}`."""
    npz = tmp_path / f"{name}.npz"
    np.savez(npz, np.asarray(value, dtype=np.float32))
    run = {"run": "ok", "launches": launches, "results": {"dict": [["case", {"array": 0}]]}}
    return run, npz


CUDA_GENERATOR = "RuntimeError: Cannot get CUDA generator without ATen_cuda library. PyTorch"


@pytest.mark.parametrize(
    ("answer", "gold", "second", "random", "want"),
    [
        ({"run": "ok"}, {"run": "call_error"}, None, False, "not_judged ran_reference_unjudged"),
        ({"run": "ok"}, {"run": "ok", "results": {"list": [{"object": "Foo"}]}}, None, False,
         "not_judged ran_reference_unjudged"),
        ({"run": "ok"}, None, None, True, "not_judged ran_random_task"),
        ({"run": "ok"}, None, [9.0, 9.0], False, "not_judged ran_reference_unwritten"),
        ({"run": "ok"}, None, [1.0, 2.0], False, "verified verified"),
        ([1.0, 2.01], None, None, False, "verified verified_lenient"),
        ([1.0, 2.08], None, None, False, "verified close_reduced_precision"),
        (([1.0, 2.08], 0), None, None, False, "no_kernel verified_no_kernel"),
        (([1.0, 2.0], 0), None, None, False, "no_kernel verified_no_kernel"),
        ([1.0, 3.0], None, None, False, "wrong wrong_result"),
        ({"run": "oob_read"}, {"run": "oob_write"}, None, False,
         "not_judged fault_also_in_reference"),
        ({"run": "race"}, None, None, False, "unsafe race"),
        ({"run": "call_error", "message": "ModuleNotFoundError: No module named 'flash_attn'"},
         None, None, False, "not_judged environment"),
        ({"run": "call_error", "message": "ModuleNotFoundError: No module named 'triton.ops'"},
         None, None, False, "error call_error"),
        ({"run": "call_error", "message": CUDA_GENERATOR}, None, None, False,
         "not_judged environment"),
        ({"run": "call_error", "message": "NameError: name 'f' is not defined"},
         {"run": "call_error", "message": "NameError: name 'f' is not defined"}, None, False,
         "not_judged environment"),
        ({"run": "call_error", "message": "NameError: name 'f' is not defined"},
         {"run": "call_error", "message": "NameError: name 'g' is not defined"}, None, False,
         "error call_error"),
        ({"run": "crash", "message": "Aborted"}, None, None, False, "error crash"),
        ({"run": "timeout", "seconds": 300}, None, None, False, "too_slow timeout"),
        ({"run": "unsupported", "message": "tt.foo"}, None, None, False, "not_judged unsupported"),
    ],
)  # fmt: skip
def test_the_verdict_rules(
    tmp_path: Path, answer: Any, gold: Any, second: Any, random: bool, want: str
) -> None:
    """The reference kept `[1, 2]` unless said otherwise; an answer given as values ran and
    kept those (with one launch, or the number of launches given alongside)."""
    gold_run = results(tmp_path, "gold", [1.0, 2.0]) if gold is None else (gold, tmp_path / "x")
    if gold is not None and gold["run"] == "ok":
        gold_run = (gold, results(tmp_path, "gold", [1.0, 2.0])[1])
    again = results(tmp_path, "second", second if second is not None else [1.0, 2.0])
    if isinstance(answer, tuple):
        run, npz = results(tmp_path, "answer", answer[0], launches=answer[1])
    elif isinstance(answer, list):
        run, npz = results(tmp_path, "answer", answer)
    else:
        run, npz = {**answer}, tmp_path / "answer.npz"
        if run["run"] == "ok":
            run, npz = results(tmp_path, "answer", [1.0, 2.0])
    got = tb.decide(run, npz, gold_run, lambda: again, random)
    assert f"{got['verdict']} {got['class']}" == want, got


@pytest.mark.parametrize(
    ("module", "lacking"),
    [
        ("torch._inductor.triton_heuristics", True),  # moved in later torch
        ("flag_gems", True),
        ("vllm.model_executor", True),
        ("kernel_utils", False),  # a module of the answer's own invention
        ("models.mamba.ops", False),
        ("tanh_linear", False),
        ("triton.ops", False),  # Triton is here: what it lacks, the answer made up
    ],
)
def test_a_missing_module_is_this_machines_only_if_it_is_a_known_package(
    module: str, lacking: bool
) -> None:
    assert tb.environment(f"ModuleNotFoundError: No module named '{module}'") is lacking


@pytest.mark.parametrize(
    ("code", "random"),
    [
        ("def f(x):\n    return x + torch.randn(3)\n", True),
        ("def f(x):\n    return torch.nn.functional.dropout(x)\n", True),
        ("x = tl . rand(seed, offs)\n", True),
        ("torch.manual_seed(1234)  # the test's inputs move\n", True),
        ("def f(x):\n    # x = torch.randn(10, 10)\n    return x\n", False),
        ('def f(x):\n    """>>> f(torch.rand(3, 4))"""\n    return x\n', False),
        ("def f(x, seed):  # seed\n    return x\n", False),
        ('print(f"{torch.rand(1)}")\n', True),
        ('"""an unterminated docstring, torch.rand\n', True),  # does not tokenize: the text
    ],
)
def test_the_answers_randomness_is_read_off_its_code_only(code: str, random: bool) -> None:
    assert tb.draws_random("", code) is random


def test_the_tasks_randomness_is_read_off_its_text() -> None:
    assert tb.draws_random("# torch.manual_seed(0)\n", "")


def test_the_columns() -> None:
    """The columns, from the classes of made-up answers."""
    classes = {
        "call_error": 5, "verified": 2, "verified_lenient": 1, "wrong_result": 1, "race": 1,
        "ran_random_task": 1, "fault_also_in_reference": 1, "environment": 1,
    }  # fmt: skip
    records = [{"class": name} for name, n in classes.items() for _ in range(n)]
    assert tb.columns(records) == {
        "answers": 13, "verified": 3, "reduced": 0, "wrong": 1, "unsafe": 1,
        "not_compared": 2, "not_judged": 1, "ran": 7,
    }  # fmt: skip


def references(out: Path, values: list[float]) -> None:
    """Both reference runs of `TASK`, as if they had run and kept `{"case": values}`."""
    digest = hashlib.sha256(TASK.read_bytes()).hexdigest()[:12]
    for which in ("first", "second"):
        report = out / "reference" / f"{TASK.stem}-{digest}.{which}.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        run, npz = results(report.parent, report.stem, values)
        report.write_text(json.dumps(run))


@pytest.mark.parametrize(
    ("report", "message"),
    [
        ('{"run": "ok", "launches": 1, "resul', "unreadable report"),  # cut short by a kill
        ('["run", "ok"]', "unreadable report"),
        (json.dumps({"run": "ok", "launches": 1, "results": {"array": 7}}),
         "malformed run report: IndexError"),
        (json.dumps({"run": "oob_read", "fault_file": "a.tritonbench.py", "line": "x"}),
         "malformed run report: ValueError"),
        (json.dumps({"run": "verified"}), "malformed run report: ValueError"),
    ],
)  # fmt: skip
def test_a_report_that_cannot_be_read_is_the_answers_crash_for_good(
    tmp_path: Path, report: str, message: str
) -> None:
    """The answer's report is written in its own process, which it may write anything from, or
    cut short by a kill: nothing is run here (every report is on disk), and the record, which a
    rerun reads back, is the answer's crash, not an exception out of the batch."""
    references(tmp_path, [1.0, 2.0])
    np.savez(tmp_path / "a.tritonbench.run.npz", np.zeros(2, dtype=np.float32))
    (tmp_path / "a.tritonbench.run.json").write_text(report)
    item = {"id": "a", "task": str(TASK), "answer": str(FIXTURES / "answer_ok.py")}
    got = tb.judge_item(item, tmp_path, 10, sandbox=False)[tb.RECORD]
    assert (got["verdict"], got["class"]) == ("error", "crash"), got
    assert got["message"].startswith(message)
    assert tb.judge_item(item, tmp_path, 10, sandbox=False)[tb.RECORD] == got


def test_a_reference_report_that_cannot_be_read_leaves_the_answer_unjudged(tmp_path: Path) -> None:
    references(tmp_path, [1.0, 2.0])
    first = next((tmp_path / "reference").glob("*.first.json"))
    first.write_text('{"run": "o')
    np.savez(tmp_path / "a.tritonbench.run.npz", np.asarray([1.0, 2.0], dtype=np.float32))
    run = {"run": "ok", "launches": 1, "results": {"dict": [["case", {"array": 0}]]}}
    (tmp_path / "a.tritonbench.run.json").write_text(json.dumps(run))
    item = {"id": "a", "task": str(TASK), "answer": str(FIXTURES / "answer_ok.py")}
    got = tb.judge_item(item, tmp_path, 10, sandbox=False)[tb.RECORD]
    assert (got["verdict"], got["class"]) == ("not_judged", "ran_reference_unjudged"), got


def test_an_answer_does_not_see_the_tree_its_task_or_out_sits_in(tmp_path: Path) -> None:
    home = Path.home()
    bench = home / "bench-not-there" / "data" / "TritonBench_G_v1"
    out = home / "runs-not-there" / "out"
    got = tb.answer_sandbox(bench / "relu.py", out)
    assert got == [
        "--tmpfs", str(home / "bench-not-there"), "--tmpfs", str(home / "runs-not-there"),
        "--bind", str(out), str(out), "--tmpfs", str(out / "reference"),
    ]  # fmt: skip
    # a task in the home directory itself, OUT under /tmp (the sandbox's own /tmp hides it)
    got = tb.answer_sandbox(home / "relu.py", tmp_path)
    start = got.index("--ro-bind")
    assert got[start : start + 3] == ["--ro-bind", "/dev/null", str(home / "relu.py")]
    # a task inside this package's tree: the tree up to the package (OUT's own tree, where
    # TMPDIR is outside /tmp, is hidden next to it)
    if not ROOT.is_relative_to("/tmp"):
        got = tb.answer_sandbox(TASK, tmp_path)
        assert str(ROOT / "tests") in [got[i + 1] for i, arg in enumerate(got) if arg == "--tmpfs"]


def test_a_numpy_scalar_is_kept_as_a_value() -> None:
    pytest.importorskip("torch")
    from ttsem import _tritonbench_runner as runner

    arrays: list[np.ndarray] = []
    kept = runner.plain({"a": np.float64(0.5), "b": np.int32(3), "c": np.longdouble(1)}, arrays)
    assert kept == {"dict": [["a", 0.5], ["b", 3], ["c", {"array": 0}]]}
    assert runner.plain(object(), arrays) == {"object": "object"}
