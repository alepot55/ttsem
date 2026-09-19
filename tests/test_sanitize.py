"""`sanitize`: a Triton program run without a GPU, every launch executed by the semantics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

from ttsem import sanitize  # noqa: E402

AGENTS = Path(__file__).resolve().parent / "fixtures" / "agents"


def test_a_correct_kernel_runs_and_its_result_reaches_the_program(tmp_path: Path) -> None:
    out = tmp_path / "out.npy"
    report = sanitize.run_program(AGENTS / "add_ok.py", argv=["--out", str(out)])
    assert report.verdict == "ok", report
    assert report.launches == 1
    np.testing.assert_array_equal(np.load(out), np.arange(1000, dtype=np.float32) + 1)


def test_a_store_without_its_mask_is_an_out_of_bounds_write_with_a_name_and_a_size() -> None:
    report = sanitize.run_program(AGENTS / "add_no_store_mask.py")
    assert report.verdict == "oob_write", report
    fault = report.fault
    assert fault is not None and fault.buffer == "out_ptr"
    assert fault.elements_past_end == 24  # four blocks of 256 over 1000 elements
    assert "tt.store" in fault.op
    assert "out_ptr" in report.message_for_agent and "24" in report.message_for_agent
    # the agent is told the line of its own source, not the IR
    assert fault.source_line == 18 and fault.source_file.endswith("add_no_store_mask.py")
    assert "add_no_store_mask.py:18" in report.message_for_agent
    assert "tl.store(out_ptr + offs, x + y)" in report.message_for_agent


def test_a_load_without_its_mask_is_an_out_of_bounds_read() -> None:
    report = sanitize.run_program(AGENTS / "add_no_load_mask.py")
    assert report.verdict == "oob_read", report
    assert report.fault is not None and report.fault.buffer == "x_ptr"
    assert "tt.load" in report.fault.op


def test_a_size_that_fills_its_blocks_hides_the_bug_from_any_test() -> None:
    """n = 1024 is what a quick test uses: nothing is out of bounds, the kernel 'works'."""
    report = sanitize.run_program(AGENTS / "add_no_store_mask.py", argv=["--n", "1024"])
    assert report.verdict == "ok", report


def test_a_script_written_for_a_gpu_runs_unmodified_without_one(capsys) -> None:
    report = sanitize.run_script(AGENTS / "cuda_script.py")
    assert report.verdict == "ok", report
    assert report.launches == 1
    assert "SCRIPT_OK 999000.0" in capsys.readouterr().out


def test_a_pytest_suite_written_for_a_gpu_fails_where_the_kernel_is_wrong(capsys) -> None:
    code = sanitize.run_pytest([str(AGENTS / "suite_no_mask.py"), "-q", "-p", "no:cacheprovider"])
    out = capsys.readouterr().out
    assert code != 0
    assert "1 failed, 1 passed" in out  # n = 1024 hides it, n = 1000 does not
    assert "24 element(s) past the end" in out and "suite_no_mask.py:15" in out


def test_memory_from_torch_empty_is_poisoned_so_reading_it_shows(capsys) -> None:
    """`torch.empty` returns zero pages often enough, on a GPU too, that a kernel accumulating
    into it passes its tests. Here such memory holds NaN (a loud pattern for integers), so the
    same program fails every time."""
    with pytest.raises(AssertionError, match="depends on what the memory held"):
        sanitize.run_script(AGENTS / "accumulate_into_empty.py")
    assert "ALLNAN True" in capsys.readouterr().out  # every element, not whatever was there


def test_poisoned_memory_does_not_disturb_a_program_that_writes_before_it_reads(capsys) -> None:
    report = sanitize.run_script(AGENTS / "cuda_script.py")  # its `out` is a `torch.empty`
    assert report.verdict == "ok" and "SCRIPT_OK 999000.0" in capsys.readouterr().out
