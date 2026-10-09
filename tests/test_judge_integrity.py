"""What an answer can do to its own verdict, or to another's, from inside its run; and what the
sandbox around the run keeps from it.

The answers under `fixtures/judge/` and `fixtures/tritonbench/` named `swallows`, `forges` and
`prewrites` were written to cheat: one catches the fault the judge would raise into its code and
lets PyTorch compute the result, one rewrites at exit every report it can name as `verified`, one
writes a `verified` record for another answer of its batch before that answer is judged. Each got
`verified` from the judge before its runs sent faults out of band, ended with `os._exit`, wrote
outputs instead of verdicts and wrote only in directories of their own. They run through the
judge's own `bwrap` sandbox like any answer, so those tests skip where it cannot run; the rest
builds the sandbox's command or reads hand-made runs, and runs no answer.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pty
import select
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import Planted

from ttsem import _guard, judge
from ttsem import judge_tritonbench as tb
from ttsem.judge_io import (
    EXIT_STOPPED,
    Ran,
    load_npz,
    printable,
    read_untrusted,
    run_sandboxed,
)

HERE = Path(__file__).resolve().parent
KB = HERE / "fixtures" / "judge"
TB = HERE / "fixtures" / "tritonbench"
ROOT = HERE.parent

needs_bwrap = pytest.mark.skipif(
    judge.sandbox_problem() is not None, reason="needs a working bwrap"
)
needs_sandbox = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("triton") is None
    or judge.sandbox_problem() is not None,
    reason="needs torch, Triton and a working bwrap",
)


def batch(here: Path, lines: list[tuple[str, Path, Path]], *flags: str) -> dict[str, Any]:
    """A manifest judged in order, one answer at a time, through the package's entry point:
    the rows by id."""
    here.mkdir(parents=True, exist_ok=True)
    manifest = here / "manifest.jsonl"
    rows = [{"id": i, "task": str(task), "answer": str(answer)} for i, task, answer in lines]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    out = here / "out"
    done = subprocess.run(
        [sys.executable, "-m", "ttsem", "judge", *flags, "--manifest", str(manifest)]
        + ["--out", str(out), "--jobs", "1"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    found = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]
    return {row["id"]: row for row in found}


# --- the static tamper scan and the integrity snapshot -------------------------------------------


def test_the_tamper_scan_flags_an_answer_that_reaches_into_the_judge() -> None:
    """The forms the scan refuses, before an answer runs: importing the judge, the module table,
    the frame walk, `ctypes`, the garbage collector, rewriting an object's code, `/proc/self/fd`,
    and the os calls that leave the process or write a foreign fd."""
    forbidden = judge.FORBIDDEN_MODULES
    for source in (
        "import ttsem.memory as M",
        "from ttsem import sanitize",
        "import sys\nx = sys.modules",
        "import ctypes",
        "import os\nos._exit(0)",
        "import os\nos.write(3, b'x')",
        "f.__code__ = g.__code__",
        "setattr(obj, '__code__', c)",
        "import gc\ngc.get_referrers(x)",
        "import inspect\ninspect.currentframe()",
        "frame = e.__traceback__.tb_frame.f_back.f_globals",
        "open('/proc/self/fd/4')",
        "import sys\nfd = sys.argv[sys.argv.index('--events') + 1]",
    ):
        assert _guard.tamper_scan(source, forbidden) is not None, source


def test_the_tamper_scan_passes_ordinary_kernels_and_pytorch() -> None:
    """Narrow enough that a real Triton answer passes: imports, torch and triton calls, os.environ
    and os.path, reading an attribute that is not being rewritten."""
    forbidden = judge.FORBIDDEN_MODULES
    for name in ("ok", "wrong", "unsafe", "no_kernel", "swallows", "forges", "prewrites"):
        source = (KB / f"answer_{name}.py").read_text()
        assert _guard.tamper_scan(source, forbidden) is None, name
    ordinary = (
        "import os, math\n"
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n"
        "p = os.environ.get('X')\n"
        "d = os.path.dirname(__file__)\n"
        "@triton.jit\n"
        "def k(x, n, B: tl.constexpr):\n"
        "    o = tl.program_id(0) * B + tl.arange(0, B)\n"
        "    tl.store(x + o, tl.load(x + o, mask=o < n), mask=o < n)\n"
    )
    assert _guard.tamper_scan(ordinary, forbidden) is None


# The round-3 evasions: every way an answer was seen (or imagined) reaching the checker without
# naming it. Each must be refused by the scan, before it runs, whatever rule fires.
ROUND_THREE = {
    "getattr_chain": "import torch\ng = getattr\nm = g(g(torch, 'jit'), 'script')\n",
    "builtins_import_attr": "import builtins\nM = builtins.__import__('ttsem.memory')\n",
    "dunder_builtins_subscript": "imp = __builtins__['__imp' + 'ort__']\n",
    "vars_sys_modules": "import sys\nm = vars(sys)['modules']\n",
    "importlib_computed": "import importlib\nM = importlib.import_module('tt' + 'sem')\n",
    "subclasses_walk": (
        "for c in type('x', (), {}).__class__.__base__.__subclasses__():\n    pass\n"
    ),
    "func_globals": "def f():\n    pass\nns = f.__globals__\n",
    "tb_frame": (
        "try:\n    raise ValueError\nexcept ValueError as e:\n"
        "    g = e.__traceback__.tb_frame.f_globals\n"
    ),
    "numpy_rebind": "import numpy as np\nnp.searchsorted = lambda *a, **k: 0\n",
    "numpy_setattr_computed": "import numpy as np\nsetattr(np, 'search' + 'sorted', None)\n",
    "os_write_split": "import os\nw = getattr(os, 'wri' + 'te')\n",
    "base64_exec": "import base64\nexec(base64.b64decode('cGFzcw==').decode())\n",
    "marshal_code": "import marshal\nexec(marshal.loads(b''))\n",
    "types_functiontype": "import types\nF = types.FunctionType\n",
    "utf7_coding": "# -*- coding: utf-7 -*-\nimport torch\n",
    "ctypes": "import ctypes\nctypes.memmove(0, 0, 0)\n",
    "gc_objects": "import gc\nxs = gc.get_objects()\n",
    "inspect_stack": "import inspect\nf = inspect.stack()\n",
    "proc_mem": "data = open('/proc/self/mem', 'rb')\n",
    "threading": "import threading\nthreading.Thread(target=len).start()\n",
    "setattr_code_computed": "def f():\n    pass\nsetattr(f, '__co' + 'de__', None)\n",
    "signal": "import signal\nsignal.signal(signal.SIGTERM, None)\n",
}


@pytest.mark.parametrize("name", sorted(ROUND_THREE))
def test_the_scan_refuses_the_round_three_evasions(name: str) -> None:
    """The run-time name tricks of round 2 and the fresh variants of round 3: each is refused by
    the scan, before the answer runs."""
    assert _guard.tamper_scan(ROUND_THREE[name], judge.FORBIDDEN_MODULES) is not None, name


def test_the_integrity_snapshot_catches_a_swapped_or_recoded_callable() -> None:
    """A watched attribute rebound to another object, or the same function given another function's
    code, is a breach; the untouched snapshot is not."""
    holder: dict[str, Any] = {}

    def original() -> int:
        return 1

    def replacement() -> int:
        return 2

    holder["f"] = original
    guard = _guard.Integrity()
    guard.watch("f", lambda: holder["f"])
    check = guard.arm()
    assert check() is None and guard.breach() is None
    holder["f"] = replacement
    assert guard.breach() == "f"
    # the same object, with its code object swapped
    holder["f"] = original
    guard.arm()
    assert guard.breach() is None
    original.__code__ = replacement.__code__
    assert guard.breach() == "f"


def test_the_package_snapshot_catches_a_whole_module_and_a_library_and_the_environment() -> None:
    """`checker_integrity` over the ttsem package catches a rebind of any function of any of its
    modules (not only a hand-picked list: round 2 broke that by patching `Memory._read`, which
    was not watched), a rebind of a numpy function the checker calls, a module swapped in
    `sys.modules`, and a thread started after the snapshot. The check is a closure that holds
    everything it reads, so rebinding this module's or the class's own names does not disarm it."""
    import threading

    import numpy as np

    from ttsem import memory as memory_mod

    root = Path(_guard.__file__).resolve().parent
    guard = _guard.checker_integrity(roots=[root])
    check = guard.arm()
    assert check() is None

    original = memory_mod.Memory._read
    memory_mod.Memory._read = lambda self, addrs, dtype: None  # not on any hand-picked list
    assert check() == "ttsem.memory.Memory._read"
    memory_mod.Memory._read = original
    assert check() is None

    saved = np.searchsorted
    np.searchsorted = lambda *a, **k: 0  # type: ignore[assignment]
    assert check() == "numpy.searchsorted"
    np.searchsorted = saved  # type: ignore[assignment]
    assert check() is None

    stolen = sys.modules["ttsem.memory"]
    sys.modules["ttsem.memory"] = object()  # type: ignore[assignment]
    assert check() == "sys.modules['ttsem.memory']"
    sys.modules["ttsem.memory"] = stolen
    assert check() is None

    done = threading.Event()
    thread = threading.Thread(target=done.wait)
    thread.start()
    try:
        assert check() == "a thread started after the snapshot"
    finally:
        done.set()
        thread.join()


def test_the_check_is_a_closure_the_guard_module_cannot_be_disarmed() -> None:
    """The check holds everything it reads, builtins included: rebinding this module's own
    `_code_of` neither crashes it nor blinds it. And `_code_of` is itself a watched function of
    the package, so rebinding it is a breach the held check reports on its own."""
    root = Path(_guard.__file__).resolve().parent
    guard = _guard.checker_integrity(roots=[root])
    check = guard.arm()
    assert check() is None
    saved = _guard._code_of
    _guard._code_of = lambda obj: None  # type: ignore[assignment]
    try:
        assert check() == "ttsem._guard._code_of"  # the held check still runs, and catches this
    finally:
        _guard._code_of = saved
    assert check() is None


def test_the_channel_exit_is_bound_at_creation() -> None:
    """The channel's way out is a closure built when it is created; the class carries no `stop`,
    `finish` or `end` attribute to rebind, so an answer cannot reach the exit of the instance in
    use by assigning to `Channel.stop` (the round-2 mute)."""
    from ttsem import judge_io

    for name in ("stop", "finish", "end", "send", "launch", "unchecked"):
        assert not hasattr(judge_io.Channel, name), name
    channel = judge_io.Channel(None)  # no fd: sends nothing
    for name in ("stop", "finish", "end", "send", "launch", "unchecked"):
        assert callable(getattr(channel, name)), name


def test_the_scan_allows_a_relative_import_and_ordinary_dunders() -> None:
    """The scan is narrow: a relative import (an answer with no package: it fails when it runs),
    `super().__init__()`, `self.__class__.__name__`, `torch.__version__`, a `getattr` with a
    string constant, a settable config flag."""
    forbidden = judge.FORBIDDEN_MODULES
    for source in (
        "from . import helper\n",
        "import torch\nx = torch.__version__\n",
        "class M(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n",
        "import torch\nname = self.__class__.__name__\n",
        "import torch\nf = getattr(torch, 'relu')\n",
        "import torch\ntorch.backends.cuda.matmul.allow_tf32 = True\n",
    ):
        assert _guard.tamper_scan(source, forbidden) is None, source


def test_a_ttsem_importing_answer_is_not_judged_before_it_runs(tmp_path: Path) -> None:
    """The KernelBench judge refuses an answer that imports the checker and patches its memory
    resolver: `not_judged`, from the scan in the parent, so the answer never runs (no sandbox
    needed for what this shows)."""
    got = judge.judge_one(
        KB / "task_relu.py", KB / "answer_tampers.py", tmp_path / "a.json", 300, sandbox=False
    )
    assert (got["verdict"], got["why"]) == ("not_judged", "tampered"), got
    assert "imports the judge's own module `ttsem`" in got["message"]


def _work_with_account(tmp_path: Path, account: dict[str, Any]) -> Path:
    work = tmp_path / "work"
    work.mkdir()
    (work / "run.json").write_text(json.dumps(account))
    return work


def test_a_report_without_the_runners_own_done_is_not_judged(tmp_path: Path) -> None:
    """A run that wrote an account but whose token-bearing `done` never arrived (the answer left
    by `os._exit`, having forged the report) is `not_judged` (`forged`), not read as finished.
    A forged `done` without the token does not count."""
    account = {"trials": [{"seed": 1, "launches": 1, "output": {"type": "x"}}], "launches": 1}
    work = _work_with_account(tmp_path, account)
    head = {"answer": "a.py", "scale": "auto", "launches": 0, "settings": judge.DEFAULTS}
    reference = {"trials": [], "sweep": {}}
    forged = Ran(0, "", [{"event": "hello", "token": "T"}, {"event": "done"}])
    got = judge.verdict_of(head, reference, forged, work, "kernelbench", 60)
    assert (got["verdict"], got["why"]) == ("not_judged", "forged"), got


def test_a_forged_launch_count_is_not_judged(tmp_path: Path) -> None:
    """When the account claims more launches than the judge counted over its own channel, the run
    is `not_judged` (`forged`): a pure-torch answer cannot forge the count into `no_kernel`'s
    opposite."""
    account = {"trials": [{"seed": 1, "launches": 7, "output": {"type": "x"}}], "launches": 35}
    work = _work_with_account(tmp_path, account)
    head = {"answer": "a.py", "scale": "auto", "launches": 0, "settings": judge.DEFAULTS}
    reference = {"trials": [], "sweep": {}}
    ran = Ran(
        0,
        "",
        [{"event": "hello", "token": "T"}, {"event": "done", "token": "T", "launches": 0}],
    )
    got = judge.verdict_of(head, reference, ran, work, "kernelbench", 60)
    assert (got["verdict"], got["why"]) == ("not_judged", "forged"), got
    assert "35" in got["message"] and "0" in got["message"]


# --- the race check's budget ---------------------------------------------------------------------


def test_the_channel_counts_the_launches_not_race_checked() -> None:
    """Each launch too large for the race check is counted by the channel itself and sent at once;
    the judge reads the count off the runner's own events only, not off a line the answer wrote
    to the same pipe."""
    from ttsem import judge_io

    read_fd, write_fd = os.pipe()
    try:
        channel = judge_io.Channel(write_fd)
        channel.launch(1)
        channel.unchecked()
        channel.unchecked()
        os.write(write_fd, b'\n{"event": "unchecked", "n": 0}\n{"event": "done"}\n')
        os.close(write_fd)
        events = judge_io.events_of(judge_io.drain(read_fd))
    finally:
        os.close(read_fd)
    assert Ran(0, "", events).confirmed_unchecked() == 2


def _kb_run(
    tmp_path: Path, got: list[float], claimed: int | None, last: dict[str, Any]
) -> dict[str, Any]:
    """The report the judge writes for a hand-made run: one trial whose output is `got` against a
    reference of [1, 2], an account claiming `claimed` launches not race-checked (no claim when
    None), and `last` as the runner's last event (its `done` or `stop`)."""
    torch = pytest.importorskip("torch")
    work = tmp_path / "work"
    work.mkdir()
    np.savez(
        work / "run.npz", allow_pickle=False, trial0=np.asarray(got, np.float32).view(np.uint8)
    )
    output = {"dtype": "float32", "shape": [len(got)]}
    account: dict[str, Any] = {
        "trials": [{"seed": 1, "launches": 1, "output": output}],
        "launches": 1,
        "tuners": 0,
    }
    if claimed is not None:
        account["races_unchecked"] = claimed
    (work / "run.json").write_text(json.dumps(account))
    reference = {
        "trials": [{"seed": 1, "precision": "fp32", "want": torch.tensor([1.0, 2.0])}],
        "sweep": {},
    }
    head = {"answer": "a.py", "scale": "auto", "launches": 0, "settings": judge.DEFAULTS}
    code = EXIT_STOPPED if last["event"] == "stop" else 0
    ran = Ran(code, "", [{"event": "hello", "token": "T"}, {"token": "T", **last}])
    return judge.verdict_of(head, reference, ran, work, "kernelbench", 60)


@pytest.mark.parametrize(
    ("got", "claimed", "sent", "want"),
    [
        ([1.0, 2.0], 0, 0, "verified"),
        ([1.0, 2.0], 2, 2, "not_judged races_unchecked"),
        ([1.0, 2.0], 2, None, "not_judged races_unchecked"),  # a runner that sends no count
        ([1.0, 3.0], 2, 2, "wrong"),
        ([1.0, 2.0], 0, 2, "not_judged forged"),  # an account that lowers the count
    ],
)
def test_a_kernelbench_run_with_launches_not_race_checked_is_never_verified(
    tmp_path: Path, got: list[float], claimed: int, sent: int | None, want: str
) -> None:
    """Round 4: a launch over `ttsem.memory.ACCESS_BUDGET` accesses is run and compared but not
    checked for races, and a racing kernel padded past the budget was `verified`. Values that
    match from such a run are `not_judged` (`races_unchecked`), with the count and the budget in
    the message; values that differ are still `wrong`. The count is the channel's: an account
    that says fewer is `forged`."""
    last: dict[str, Any] = {"event": "done", "launches": 1}
    if sent is not None:
        last["races_unchecked"] = sent
    report = _kb_run(tmp_path, got, claimed, last)
    assert " ".join([report["verdict"], *([report["why"]] if "why" in report else [])]) == want
    line = judge.describe(report)
    if want == "not_judged races_unchecked":
        assert report["races_unchecked"] == 2
        assert line.startswith("not_judged (races_unchecked): 2 launch(es) too large"), line
        assert "4,000,000" in line
    if want == "wrong":
        assert line.endswith("; 2 launch(es) too large to be checked for races"), line


def test_a_kernelbench_fault_is_unsafe_whatever_the_count(tmp_path: Path) -> None:
    stop = {"event": "stop", "kind": "fault", "fault": "race", "launches": 1, "races_unchecked": 3}
    report = _kb_run(tmp_path, [1.0, 2.0], 3, stop)
    assert (report["verdict"], report["kind"], report["races_unchecked"]) == ("unsafe", "race", 3)


def test_a_tritonbench_report_that_lowers_the_count_is_tampered(tmp_path: Path) -> None:
    """The TritonBench judge reads the count the same way: the channel's, and a report that says
    fewer is `tampered`; from a runner that sends none, the report's own."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "run.json").write_text(json.dumps({"run": "ok", "launches": 1, "races_unchecked": 0}))
    hello = {"event": "hello", "token": "T"}
    done = {"event": "done", "token": "T", "launches": 1, "races_unchecked": 1}
    got = tb.run_report(Ran(0, "", [hello, done]), work, Path("a.py"), 60)
    assert got["run"] == "tampered" and "1 over its own channel" in got["message"], got
    (work / "run.json").write_text(json.dumps({"run": "ok", "launches": 1, "races_unchecked": 2}))
    older = {"event": "done", "token": "T", "launches": 1}
    got = tb.run_report(Ran(0, "", [hello, older]), work, Path("a.py"), 60)
    assert (got["run"], got["races_unchecked"]) == ("ok", 2), got


@needs_sandbox
@pytest.mark.parametrize(("rewrites", "want"), [(0, ("unsafe", "race")), (1100, None)])
def test_a_race_padded_past_the_budget_is_not_verified(
    tmp_path: Path, rewrites: int, want: tuple[str, str] | None
) -> None:
    """The round-4 break, end to end: program 0 stores what every program reads, which the order
    ttsem runs them in makes right. Unpadded, the judge finds the race; padded with rewrites past
    the race check's budget, the values match and the judge says it did not look (it said
    `verified` before)."""
    answer = KB / "answer_race_unchecked.py"
    if rewrites != 1100:
        source = answer.read_text().replace("REWRITES = 1100\n", f"REWRITES = {rewrites}\n")
        answer = tmp_path / "answer_race.py"
        answer.write_text(source)
    got = judge.judge_one(KB / "task_relu.py", answer, tmp_path / "a.json", 300)
    if want is not None:
        assert (got["verdict"], got["kind"]) == want, got
        return
    assert (got["verdict"], got["why"]) == ("not_judged", "races_unchecked"), got
    assert got["races_unchecked"] == 5 and all(t["pass"] for t in got["trials"])  # 5 trials


@needs_sandbox
def test_a_tritonbench_race_padded_past_the_budget_is_not_verified(tmp_path: Path) -> None:
    item = {
        "id": "a",
        "task": str(TB / "task_relu.py"),
        "answer": str(TB / "answer_race_unchecked.py"),
    }
    got = tb.judge_item(item, tmp_path, 300, sandbox=True)[tb.RECORD]
    assert (got["verdict"], got["class"], got["races_unchecked"]) == (
        "not_judged",
        "races_unchecked",
        1,
    ), got


# --- the forged answers -------------------------------------------------------------------------


@needs_sandbox
def test_a_fault_the_answer_catches_is_still_unsafe(tmp_path: Path) -> None:
    """KernelBench: the fault reaches the judge before the answer's `except` does, and the run
    ends there."""
    got = judge.judge_one(KB / "task_relu.py", KB / "answer_swallows.py", tmp_path / "a.json", 300)
    assert (got["verdict"], got["kind"], got["buffer"]) == ("unsafe", "oob_read", "x_ptr"), got
    assert (got["file"], got["source"]) == ("answer_swallows.py", "x = tl.load(x_ptr + offs)")


@needs_sandbox
def test_a_fault_a_tritonbench_answer_catches_is_still_unsafe(tmp_path: Path) -> None:
    item = {"id": "a", "task": str(TB / "task_relu.py"), "answer": str(TB / "answer_swallows.py")}
    got = tb.judge_item(item, tmp_path, 300, sandbox=True)[tb.RECORD]
    assert (got["verdict"], got["class"]) == ("unsafe", "oob_read"), got
    assert (got["file"], got["line"]) == ("answer_swallows.py", 14)


@needs_sandbox
@pytest.mark.parametrize("answer", ["answer_patch_restore.py", "answer_thread_flip.py"])
def test_a_runtime_patch_of_the_checker_is_not_judged(tmp_path: Path, answer: str) -> None:
    """Two answers that patch a watched function of the checker at run time, with the static scan
    off in the parent (the runtime integrity check alone): one restores the patch around the
    launch, one flips it from a background thread between launches. Each is caught (`tampered`),
    never `verified`. With the scan on (the default) they are refused before they run."""
    import unittest.mock as mock

    with mock.patch.object(judge, "tamper_reason", lambda answer: None):
        got = judge.judge_one(KB / "task_relu.py", KB / answer, tmp_path / "a.json", 300)
    assert (got["verdict"], got["why"]) == ("not_judged", "tampered"), got
    assert "changed the judge's own code" in got["message"]


@needs_sandbox
def test_an_answer_that_rewrites_reports_at_exit_is_judged_on_its_outputs(tmp_path: Path) -> None:
    """Its exit handler never runs (the run leaves by `os._exit`), and no file of its run says
    the verdict anyway: x + 1 is wrong."""
    task, answer = KB / "task_relu.py", KB / "answer_forges.py"
    got = judge.judge_one(task, answer, tmp_path / "a.json", 300)
    assert got["verdict"] == "wrong", got
    assert got["trials"][0]["max_abs_diff"] > 0.5
    account = json.loads((tmp_path / judge.WORK / "a" / "run.json").read_text())
    assert account.get("verdict") is None and account["trials"]  # the runner's, not the forge's


@needs_sandbox
def test_no_answer_writes_the_record_of_another_in_its_batch(tmp_path: Path) -> None:
    """`prewrites` runs first and writes a `verified` record for `victim` (a wrong answer)
    wherever it can reach; `victim` is judged all the same, and nothing was written in OUT."""
    task = KB / "task_relu.py"
    lines = [
        ("prewrites", task, KB / "answer_prewrites.py"),
        ("victim", task, KB / "answer_wrong.py"),
    ]
    rows = batch(tmp_path, lines)
    assert rows["prewrites"]["scaled"]["verdict"] == "verified"
    assert rows["victim"]["scaled"]["verdict"] == "wrong", rows["victim"]
    out = tmp_path / "out"
    assert json.loads((out / "victim.scaled.json").read_text())["trials"][0]["seed"] is not None
    assert not (out / judge.WORK / "victim.scaled.json").exists()
    assert (out / judge.WORK / "prewrites.scaled" / "victim.scaled.json").exists()  # its own


@needs_sandbox
def test_no_tritonbench_answer_writes_the_record_of_another_in_its_batch(tmp_path: Path) -> None:
    task = TB / "task_relu.py"
    lines = [
        ("prewrites", task, TB / "answer_prewrites.py"),
        ("victim", task, TB / "answer_wrong.py"),
    ]
    rows = batch(tmp_path, lines, "--tritonbench")
    assert rows["prewrites"][tb.RECORD]["verdict"] == "verified"
    victim = rows["victim"][tb.RECORD]
    assert (victim["verdict"], victim["class"]) == ("wrong", "wrong_result"), victim


@needs_sandbox
def test_strict_keeps_every_verdict(tmp_path: Path) -> None:
    """With --strict the runs see only what they need of the filesystem, and judge the same."""
    task = KB / "task_relu.py"
    names = ("ok", "wrong", "unsafe", "no_kernel")
    rows = batch(tmp_path / "kb", [(n, task, KB / f"answer_{n}.py") for n in names], "--strict")
    assert {n: rows[n]["scaled"]["verdict"] for n in names} == {
        "ok": "verified", "wrong": "wrong", "unsafe": "unsafe", "no_kernel": "no_kernel",
    }  # fmt: skip
    task = TB / "task_relu.py"
    names = ("ok", "unsafe")
    rows = batch(
        tmp_path / "tb",
        [(n, task, TB / f"answer_{n}.py") for n in names],
        "--tritonbench",
        "--strict",
    )
    assert {n: rows[n][tb.RECORD]["verdict"] for n in names} == {
        "ok": "verified",
        "unsafe": "unsafe",
    }


# --- the sandbox --------------------------------------------------------------------------------

UNHIDE = r"""
import ctypes, json, os, sys
libc = ctypes.CDLL(None, use_errno=True)
status = open("/proc/self/status").read().splitlines()
caps = {l.split(":")[0]: l.split()[1] for l in status if l.startswith("CapEff")}
# first try to take the hiding away, as a run with CAP_SYS_ADMIN could (MNT_DETACH)
tries = {path: libc.umount2(path.encode(), 2) for path in sys.argv[2:]}
try:
    secret = open(sys.argv[1]).read()
except OSError as exc:
    secret = "unreadable: " + type(exc).__name__
print(json.dumps({"caps": caps, "umount": tries, "secret": secret}))
"""


@needs_bwrap
def test_a_run_has_no_capability_and_cannot_take_the_hiding_away(
    planted: Planted, tmp_path: Path
) -> None:
    """Run by root (as in a cloud session) bwrap keeps its capabilities unless told to drop them,
    and with CAP_SYS_ADMIN a run unmounts the tmpfs over a hidden directory and reads what is
    under it. The run tries that first, on every directory the sandbox hides, then reads the
    secret planted in the caller's home."""
    out = tmp_path / "out"
    out.mkdir()
    cmd = judge.sandbox_prefix(out, out / ".cache", [], hide=[tmp_path / "elsewhere"])
    hidden = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--tmpfs"]
    done = subprocess.run(
        [*cmd, sys.executable, "-c", UNHIDE, str(planted.file), *hidden],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=judge.run_env(out / ".cache"),
        check=False,
    )
    assert done.returncode == 0, done.stderr
    seen = json.loads(done.stdout)
    # the effective set is empty, so no CAP_SYS_ADMIN: the unmount below is what it forbids
    assert seen["caps"] == {"CapEff": "0000000000000000"}, seen
    assert all(code != 0 for code in seen["umount"].values()), seen["umount"]
    assert seen["secret"].startswith("unreadable: ")
    assert planted.value not in done.stdout


INJECT = r"""
import fcntl, os, termios
done = []
for name, fd in (("tty", None), ("stdin", 0)):
    try:
        fd = os.open("/dev/tty", os.O_RDWR) if fd is None else fd
        for c in b"INJECTED\n":
            fcntl.ioctl(fd, termios.TIOCSTI, bytes([c]))
        done.append(name)
    except OSError as exc:
        pass
print("pushed:", done)
"""


def in_a_terminal(command: str) -> str:
    """`command` run by a shell in a fresh terminal, which then reads one line from it: what
    it read (a run that pushes a line into the terminal's input queue is read here)."""
    pid, fd = pty.fork()
    if pid == 0:  # the shell
        script = f"{command} >/dev/null 2>&1; IFS= read -r -t 3 line; echo READ=[$line]"
        os.execvp("bash", ["bash", "--norc", "--noprofile", "-c", script])
    seen = b""
    end = time.time() + 60
    while time.time() < end and b"READ=[" not in seen.split(b"\n")[-1] + seen[-80:]:
        ready, _, _ = select.select([fd], [], [], 0.5)
        if ready:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            seen += chunk
        if b"READ=[" in seen and seen.rstrip().endswith(b"]"):
            break
    os.waitpid(pid, 0)
    os.close(fd)
    text = seen.decode(errors="replace")
    return text[text.rindex("READ=[") + 6 :].split("]")[0] if "READ=[" in text else ""


@needs_bwrap
@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_run_cannot_push_input_into_the_callers_terminal(tmp_path: Path) -> None:
    """TIOCSTI on its controlling terminal, or on a terminal it was handed as stdin, puts bytes
    in the input queue of whatever reads that terminal next: the shell the judge was started
    from. Out of the sandbox, the same code does it (else the kernel forbids it, and there is
    nothing to show); in the sandbox (a session of its own, no capability), it cannot."""
    child = shlex.join([sys.executable, "-c", INJECT])
    if in_a_terminal(child) != "INJECTED":
        pytest.skip("this kernel lets no process push input into a terminal (TIOCSTI is off)")
    out = tmp_path / "out"
    out.mkdir()
    cmd = judge.sandbox_prefix(out, out / ".cache", [], judge.run_env(out / ".cache"))
    assert in_a_terminal(shlex.join([*cmd, sys.executable, "-c", INJECT])) == ""


def test_every_run_starts_with_stdin_closed_and_its_events_reach_the_judge(tmp_path: Path) -> None:
    """`run_sandboxed`, on a child of this test's own (no sandbox needed for what it shows): its
    stdin reads nothing, and what it sends on the pipe comes back as events."""
    child = (
        "import json, os, sys; fd = int(sys.argv[-1]); data = sys.stdin.read()\n"
        "os.write(fd, b'garbage without a newline')\n"
        "os.write(fd, b'\\n' + json.dumps({'event': 'note', 'stdin': data}).encode() + b'\\n')\n"
    )
    ran = run_sandboxed([sys.executable, "-c", child], os.environ, tmp_path, 60)
    assert ran.code == 0 and ran.events == [{"event": "note", "stdin": ""}]


CHANNEL_CHILD = r"""
import atexit, os, sys
from ttsem.judge_io import Channel
fd = int(os.environ["EVENTS_FD"])
channel = Channel(fd)
atexit.register(lambda: os.write(fd, b'\n{"event": "atexit"}\n'))
if sys.argv[1] == "stop":
    channel.stop({"kind": "fault", "run": "oob_read"})
channel.finish(0)
raise SystemExit("unreachable")
"""


@pytest.mark.parametrize(
    ("mode", "code", "last_event"),
    [("stop", EXIT_STOPPED, "stop"), ("done", 0, "done")],
)
def test_the_channel_sends_then_leaves_without_the_answers_exit_handlers(
    tmp_path: Path, mode: str, code: int, last_event: str
) -> None:
    """`Channel.stop` and `.finish` send their event and leave by `os._exit`, so the atexit
    handler a run registered (a forge) never runs, and its `{"event": "atexit"}` never reaches
    the judge after the last word. The channel announces its token in a first `hello`; the child
    imports this checkout's `ttsem`."""
    read_fd, write_fd = os.pipe()
    env = {**os.environ, "PYTHONPATH": str(ROOT), "EVENTS_FD": str(write_fd)}
    done = subprocess.run(
        [sys.executable, "-c", CHANNEL_CHILD, mode],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=env,
        pass_fds=(write_fd,),
        check=False,
    )
    os.close(write_fd)
    os.set_blocking(read_fd, False)
    try:
        sent = os.read(read_fd, 65536)
    finally:
        os.close(read_fd)
    assert done.returncode == code, done.stderr
    events = [json.loads(line) for line in sent.split(b"\n") if line.strip()]
    assert [e["event"] for e in events] == ["hello", last_event]  # no atexit event after it
    token = events[0]["token"]
    assert all(e.get("token") == token for e in events)  # every event the runner's, by its token


def test_read_untrusted_refuses_a_link_a_fifo_and_too_large(tmp_path: Path) -> None:
    """A run writes in its own directory; what the judge reads back there is read as untrusted:
    a regular file of bounded size, never followed through a link."""
    plain = tmp_path / "plain"
    plain.write_bytes(b"ok")
    assert read_untrusted(plain) == b"ok"
    assert read_untrusted(plain, limit=1) is None  # over the limit
    link = tmp_path / "link"
    link.symlink_to("/etc/passwd")
    assert read_untrusted(link) is None  # a link is not followed
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    assert read_untrusted(fifo) is None  # not a regular file
    assert read_untrusted(tmp_path / "missing") is None


def test_load_npz_refuses_a_compressed_member_and_a_bad_header(tmp_path: Path) -> None:
    """A run's arrays are read with `allow_pickle=False` and only when stored as `np.savez`
    stores them: a compressed member, or one whose header does not read, is refused."""
    good = tmp_path / "good.npz"
    np.savez(good, np.arange(4, dtype=np.float32), np.ones(2))
    data = load_npz(good)
    assert set(data) == {"arr_0", "arr_1"} and data["arr_0"].dtype == np.float32
    packed = tmp_path / "packed.npz"
    np.savez_compressed(packed, np.arange(4))
    with pytest.raises(ValueError, match="stored"):
        load_npz(packed)
    with zipfile.ZipFile(tmp_path / "bad.npz", "w") as archive:
        archive.writestr("arr_0.npy", b"not a real npy header")
    with pytest.raises(ValueError):
        load_npz(tmp_path / "bad.npz")


def test_printable_escapes_control_characters() -> None:
    """What a run says (an exception's message, a kernel's source line) is the answer's to
    choose; a terminal takes some control sequences as commands, so the judge escapes them."""
    assert printable("ok line") == "ok line"
    assert "\x1b" not in printable("a\x1b]0;x\x07b")
    rtl = chr(0x202E)  # a right-to-left override, built so the source holds no control char
    assert printable("a" + rtl + "b") == "a\\u202eb"
    assert printable("tab\tkept?") == "tab\\tkept?"
