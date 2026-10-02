"""The scripts that rerun the two benchmark audits (`audits/`), on miniature copies of the data.

`audits/` is not part of the package: the scripts are loaded from their files. Each one is checked
on a dataset made here in the benchmark's own layout (a TritonBench checkout with one G and one T
task, a KernelBench-v3 release with three rows, a makora parquet read through a stand-in for
`pyarrow`), and on judge records written by hand. The GPU control of TritonBench runs here on CPU
tensors, through the same sandbox, so it skips where torch, Triton or `bwrap` are missing.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import types
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from conftest import Planted

from ttsem import judge

AUDITS = Path(__file__).resolve().parent.parent / "audits"
SEP = "#" * 146


def load(relative: str) -> ModuleType:
    path = AUDITS / relative
    name = "audit_" + relative.replace("/", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- TritonBench ---------------------------------------------------------------------------------


def tritonbench(root: Path) -> Path:
    """A checkout in TritonBench's layout: one G task, one T task, one answer file for each."""
    g_ref = "import torch\n\ndef add(x, y):\n    return x + y\n"
    t_ref = "import torch\n\ndef relu(x):\n    return torch.relu(x)\n"
    g_test = "\ndef test_add():\n    return {'a': add(torch.ones(3), torch.ones(3))}\n\nresult_gold = test_add()\n"
    t_test = (
        "\ndef test_relu():\n    return {'a': relu(torch.ones(3))}\n\ntest_results = test_relu()\n"
    )
    data = root / "data"
    (data / "TritonBench_G_v1").mkdir(parents=True)
    (data / "TritonBench_T_v1").mkdir(parents=True)
    (data / "TritonBench_G_v1" / "add.py").write_text(g_ref + SEP + g_test)
    (data / "TritonBench_T_v1" / "relu.py").write_text(t_ref + SEP + t_test)
    stats_g = [
        {"file": "add.py", "output": g_ref, "comp_instru": "Add two.", "simp_instru": "Add."}
    ]
    (data / "TritonBench_G_v1.json").write_text(json.dumps(stats_g))
    stats_t = [{"file": "relu.py", "description": "Applies the rectifier elementwise."}]
    (data / "TritonBench_T_v1.jsonl").write_text(json.dumps(stats_t))  # one array, as theirs
    gen = root / "LLM_generated"
    (gen / "Bench_G_general").mkdir(parents=True)
    (gen / "Bench_T_general").mkdir(parents=True)
    g_answers = [
        {
            "label": g_ref,
            "predict": "```python\nimport torch\nX = 1\ndef add(x, y):\n    return x - y\n",
        },
        {
            "label": "nothing like it",
            "instruction": "no task says this",
            "predict": "def f(): pass",
        },
    ]
    (gen / "Bench_G_general" / "output_m.jsonl").write_text(
        "".join(json.dumps(a) + "\n" for a in g_answers)
    )
    prompt = (
        "Functional Description: Applies the rectifier elementwise.Wrapper Entry Information: x"
    )
    t_answers = [
        {
            "prompt": prompt,
            "predict": "text\n```python\nimport torch\ndef relu(x):\n    return x.clamp(min=0)\n",
        }
    ]
    (gen / "Bench_T_general" / "m.jsonl").write_text(
        "".join(json.dumps(a) + "\n" for a in t_answers)
    )
    return root


def test_tritonbench_prepare_and_manifest(tmp_path: Path) -> None:
    root = tritonbench(tmp_path / "TritonBench")
    prepare, manifest = load("tritonbench/prepare.py"), load("tritonbench/manifest.py")
    assert prepare.main([str(root), str(tmp_path / "pred")]) == 0
    g = (tmp_path / "pred" / "G" / "output_m" / "add.py").read_text()
    # G: the imports and the functions only, then the separator and the task's test
    assert g.split(SEP)[0] == "import torch\n\ndef add(x, y):\n    return x - y\n"
    assert g.split(SEP)[1].lstrip("\n").startswith("def test_add")
    t = (tmp_path / "pred" / "T" / "m" / "relu.py").read_text()
    assert t.split(SEP)[0] == "\nimport torch\ndef relu(x):\n    return x.clamp(min=0)\n\n"
    assert sorted(p.name for p in (tmp_path / "pred").rglob("*.py")) == ["add.py", "relu.py"]
    assert manifest.main([str(tmp_path / "pred"), str(root), str(tmp_path / "man")]) == 0
    lines = jsonl(tmp_path / "man" / "all.jsonl")
    assert [(i["id"], i["bench"], i["model"]) for i in lines] == [
        ("G.output_m.add", "G", "output_m"),
        ("T.m.relu", "T", "m"),
    ]
    assert lines[0]["task"] == str((root / "data" / "TritonBench_G_v1" / "add.py").resolve())
    assert jsonl(tmp_path / "man" / "G.jsonl") == lines[:1]


def record(name: str) -> dict[str, Any]:
    return {"verdict": "?", "class": name}


def test_tritonbench_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    table = load("tritonbench/table.py")
    rows = [
        ("G", "output_a", "verified"),
        ("G", "output_a", "wrong_result"),
        ("G", "output_a", "call_error"),
        ("G", "b", "race"),
        ("G", "b", "ran_random_task"),
        ("T", "c", "close_reduced_precision"),
        ("T", "c", "environment"),
    ]
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "".join(
            json.dumps({"id": f"{b}.{m}.{i}", "bench": b, "model": m, "tritonbench": record(c)})
            + "\n"
            for i, (b, m, c) in enumerate(rows)
        )
    )
    assert table.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "| a | 3 | 2 | 1 | 0 | 1 | 0 | 0 | 0 | 1 of 2 |" in out  # `output_` dropped
    assert "| b | 2 | 2 | 0 | 0 | 0 | 1 | 1 | 0 | 1 of 1 |" in out
    assert "G: of the 3 answers that ran and could be judged, 2 (67%)" in out
    total = "All: 7 answers; of the 4 that ran and could be judged, 2 (50%) return wrong values"
    assert f"{total} (1) or are unsafe (1)." in out


def test_tritonbench_agree(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    agree = load("tritonbench/agree.py")
    pairs = {
        "a": ("verified", "verified_lenient"),  # right, right
        "b": ("wrong_result", "wrong_result"),
        "c": ("call_error", "call_error"),
        "d": ("verified", "wrong_result"),  # a disagreement
        "e": ("race", "wrong_result"),  # a fault: the device has none, not counted
        "f": ("ran_random_task", "verified"),  # not comparable: not counted
    }
    ours, theirs = tmp_path / "judge.jsonl", tmp_path / "device.jsonl"
    ours.write_text(
        "".join(
            f'{{"id": "{k}", "tritonbench": {json.dumps(record(a))}}}\n'
            for k, (a, _) in pairs.items()
        )
    )
    theirs.write_text(
        "".join(
            f'{{"id": "{k}", "device": {json.dumps(record(b))}}}\n' for k, (_, b) in pairs.items()
        )
    )
    assert agree.main([str(ours), str(theirs), "--list"]) == 0
    out = capsys.readouterr().out
    assert "Both sides decide on 4 answers and agree on 3 (75.0%)." in out
    assert "d: ttsem verified, device wrong_result" in out
    assert "e: ttsem" not in out


needs_device_sandbox = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("triton") is None
    or judge.sandbox_problem() is not None,
    reason="needs torch, Triton and a working bwrap",
)


@needs_device_sandbox
def test_device_runner_writes_the_judge_format() -> None:
    import numpy as np
    import torch

    from ttsem import _tritonbench_runner

    runner = load("tritonbench/device_runner.py")
    value = {"a": [torch.ones(2, dtype=torch.float16), np.float32(1.5), (3, "x")], "b": object()}
    mine: list[np.ndarray] = []
    theirs: list[np.ndarray] = []
    assert runner.plain(value, mine) == _tritonbench_runner.plain(value, theirs)
    assert all(np.array_equal(a, b) and a.dtype == b.dtype for a, b in zip(mine, theirs))


@needs_device_sandbox
def test_device_control_runs_in_the_sandbox(tmp_path: Path, monkeypatch: Any) -> None:
    """The control's whole path on CPU tensors: the task and each answer in their own sandboxed
    process, the results compared by the judge's rules; no kernel is launched, so right values
    are `no_kernel` and other values `wrong_result`."""
    device = load("tritonbench/device.py")
    monkeypatch.setattr(device, "gpu_nodes", list)
    task = tmp_path / "data" / "relu.py"
    task.parent.mkdir()
    test = "\ndef test_relu():\n    return {'a': relu(torch.arange(-3.0, 3.0))}\n\nresult_gold = test_relu()\n"
    task.write_text("import torch\n\ndef relu(x):\n    return torch.relu(x)\n" + SEP + test)
    answers = {
        "right": "import torch\ndef relu(x):\n    return x.clamp(min=0)\n",
        "wrong": "import torch\ndef relu(x):\n    return x.abs()\n",
        "raises": "import torch\ndef relu(x):\n    raise ValueError('no')\n",
    }
    out = tmp_path / "out"
    out.mkdir()
    found = {}
    for name, code in answers.items():
        answer = tmp_path / f"answer_{name}.py"
        answer.write_text(code)
        item = {"id": name, "task": str(task), "answer": str(answer)}
        found[name] = device.judge_item(item, out, timeout=120)[device.RECORD]["class"]
    assert found == {"right": "verified_no_kernel", "wrong": "wrong_result", "raises": "call_error"}
    assert len(list((out / "reference").glob("relu-*.device.json"))) == 1  # one reference run


@needs_device_sandbox
def test_the_device_control_shows_an_answer_no_secret_of_the_caller(
    planted: Planted, tmp_path: Path, monkeypatch: Any
) -> None:
    """The control's sandbox, on CPU tensors: what the file copies of a secret the caller holds,
    in its environment and in its home's config, into OUT."""
    device = load("tritonbench/device.py")
    monkeypatch.setattr(device, "gpu_nodes", list)
    script = tmp_path / "leak.py"
    script.write_text(planted.leak() + "import torch\nresult_gold = {'a': torch.ones(2)}\n")
    out = tmp_path / "out"
    out.mkdir()
    got = device.run_file(script, out / "leak.device.json", out, 120, out / ".cache", [])
    assert got["run"] == "ok", got
    seen = planted.check(out)
    assert seen["environ"][device.SANDBOXED] == "1"  # set inside, for the runner


@needs_device_sandbox
@pytest.mark.parametrize("variable", [False, True], ids=["without_the_variable", "with_it"])
def test_the_device_runner_refuses_to_run_a_file_outside_the_sandbox(
    tmp_path: Path, variable: bool
) -> None:
    """Started by hand, without the variable only `device.py`'s sandbox sets, or with it but the
    host's network in sight: the file is not run."""
    if variable and {name for _, name in socket.if_nameindex()} <= {"lo"}:
        pytest.skip("no network interface but loopback here: the host looks like the sandbox")
    script = tmp_path / "answer.py"
    script.write_text("open('ran', 'w').close()\n")
    env = {key: value for key, value in os.environ.items() if key != "TTSEM_DEVICE_SANDBOX"}
    if variable:
        env["TTSEM_DEVICE_SANDBOX"] = "1"
    runner = AUDITS / "tritonbench" / "device_runner.py"
    done = subprocess.run(
        [sys.executable, "-P", str(runner), str(script), "out.json", "out.npz"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        check=False,
    )
    assert done.returncode == 2 and "only inside device.py's bwrap sandbox" in done.stderr
    assert not (tmp_path / "ran").exists() and not (tmp_path / "out.json").exists()


# --- KernelBench ---------------------------------------------------------------------------------


PROBLEM = """import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        return x * 2

def get_inputs():
    return [torch.randn(16)]

def get_init_inputs():
    return []
"""
CANDIDATE = """import triton
import triton.language as tl

class ModelNew(Model):
    pass
"""


def test_kernelbench_prepare_v3(tmp_path: Path) -> None:
    prepare = load("kernelbench/prepare.py")
    runs, problems = tmp_path / "runs", tmp_path / "problems"
    (runs / "solutions").mkdir(parents=True)
    (problems / "level1").mkdir(parents=True)
    (problems / "level1" / "1_Double.py").write_text(PROBLEM)
    (runs / "solutions" / "M 1_RTX3090_1_Double.py.txt").write_text("@triton.jit\ndef k(): pass\n")
    (runs / "solutions" / "M 2_H100_1_Double.py.txt").write_text("import torch  # CUDA only\n")
    head = "model,gpu,level,problem,correct,speedup,precision_used,solution_link\n"
    rows = [
        "M 1,RTX3090,1,1_Double.py,True,1.5,fp32,/data/solutions/M 1_RTX3090_1_Double.py.txt",
        "M 2,H100,1,1_Double.py,False,,fp32,/data/solutions/M 2_H100_1_Double.py.txt",
        "M 3,H100,1,1_Double.py,False,,fp32,",
    ]
    (runs / "results.csv").write_text(head + "\n".join(rows) + "\n")
    prepare.main(["v3", str(runs), str(problems), str(tmp_path / "out")])
    (item,) = jsonl(tmp_path / "out" / "manifest.jsonl")  # Triton only, solutions only
    assert item["id"] == "v3_M-1_RTX3090_L1_1_Double"
    assert (item["label"], item["rule"], item["precision"], item["gpu"]) == (
        "correct",
        "v3",
        "keep",
        "RTX3090",
    )
    assert Path(item["task"]) == (problems / "level1" / "1_Double.py").resolve()
    assert Path(item["answer"]).read_text() == "@triton.jit\ndef k(): pass\n"


def test_kernelbench_prepare_makora(tmp_path: Path, monkeypatch: Any) -> None:
    prepare = load("kernelbench/prepare.py")
    columns = {
        "x": [PROBLEM + "\n" + CANDIDATE] * 4 + ["no triton here"],
        "y": [0.1, float("nan"), None, 0.2, 0.3],
        "problem_id": ["1_2", "1_2", "1_2", "3_1", "1_2"],
        "holdout_group": ["h"] * 5,
    }
    table = types.SimpleNamespace(to_pydict=lambda: columns)
    parquet = types.ModuleType("pyarrow.parquet")
    parquet.read_table = lambda path: table  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyarrow", types.ModuleType("pyarrow"))
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)
    prepare.main(["makora", "test.parquet", str(tmp_path / "out"), "--per-problem", "1,5"])
    items = sorted(jsonl(tmp_path / "out" / "manifest.jsonl"), key=lambda i: i["row"])
    # level 3 is left out, and so is the row with no Triton import; one row labelled correct
    assert [(i["row"], i["label"]) for i in items] == [(0, "correct"), (1, "failed"), (2, "failed")]
    task = Path(items[0]["task"]).read_text()
    assert task == PROBLEM and "ModelNew" in Path(items[0]["answer"]).read_text()
    assert len({i["task"] for i in items}) == 1  # one task file per problem text
    skip = ["--skip", str(tmp_path / "out" / "manifest.jsonl")]
    prepare.main(["makora", "test.parquet", str(tmp_path / "again"), "--per-problem", "1,5", *skip])
    assert jsonl(tmp_path / "again" / "manifest.jsonl") == []


def test_kernelbench_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    table = load("kernelbench/table.py")
    out = tmp_path / "out"
    out.mkdir()
    reports = {
        "a": ("correct", {"verdict": "verified"}),
        "b": ("correct", {"verdict": "wrong", "trials": [{"max_abs_diff": 0.5}]}),
        "c": ("correct", {"verdict": "unsafe", "kind": "oob_read"}),
        "d": ("failed", {"verdict": "error", "stage": "forward"}),
        "e": ("failed", {"verdict": "wrong", "configs": {"failed": {"index": 2}}}),
    }
    lines = []
    for key, (label, report) in reports.items():
        (out / f"{key}.scaled.json").write_text(json.dumps(report))
        lines.append({"id": key, "label": label, "source": "s", "model": "m"})
    lines.append({"id": "a", "label": "failed", "source": "s", "model": "m"})  # twice: ambiguous
    lines.append({"id": "z", "label": "correct", "source": "s"})  # not judged yet: left out
    (out / "b.full.json").write_text(json.dumps({"verdict": "verified"}))
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("".join(json.dumps(line) + "\n" for line in lines))
    assert table.main([str(out), "--manifest", str(manifest), "--by", "model"]) == 0
    text = capsys.readouterr().out
    assert "## s: 5 answers judged" in text
    assert "| verified | 1 | 0 | 0 |" in text  # ambiguous, correct, failed
    assert "| wrong: under another autotune config | 0 | 0 | 1 |" in text
    assert "| error: forward | 0 | 0 | 1 |" in text
    assert "full-shape pass, 1 answers:" in text
    assert "labelled correct: the judge decides on 2, verifies 0 (0%), and judges 2 wrong" in text
    assert "  m: wrong 1, unsafe 1" in text
