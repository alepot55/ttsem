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


def test_the_source_line_is_found_through_named_and_aliased_locations(tmp_path: Path) -> None:
    src = tmp_path / "k.py"
    src.write_text("first\ntl.store(p, x)\n")
    plain = f'"tt.store"(%p, %x) : () -> () loc("{src}":2:4)'
    named = f'%v = "tt.load"(%p) : () -> f32 loc("v"("{src}":2:4))'
    module = f'#loc3 = loc("{src}":2:4)\n#loc7 = loc("v"(#loc3))\n'
    for op_text, ir_text in ((plain, ""), (named, ""), ("%v = tt.load %p loc(#loc7)", module)):
        assert sanitize._source_of(op_text, ir_text) == (str(src), 2, "tl.store(p, x)")


def test_a_program_that_asks_whether_its_tensors_are_on_the_gpu_is_told_yes() -> None:
    import torch

    with sanitize.session():
        x = torch.zeros(4, device="cuda")
        assert x.is_cuda and x.device.type == "cuda"
        y = torch.ones(4, device=x.device) + x.to(x.device)  # a device read off a tensor works
        assert y.tolist() == [1.0] * 4 and torch.empty_like(x).shape == (4,)
        with torch.cuda.device(0):
            torch.cuda.set_device(0)
    assert not torch.zeros(1).is_cuda and torch.zeros(1).device.type == "cpu"


def test_a_launch_grid_that_is_a_bare_int_fails_as_it_does_on_a_gpu(tmp_path: Path) -> None:
    script = tmp_path / "int_grid.py"
    script.write_text(
        "import torch, triton, triton.language as tl\n"
        "@triton.jit\n"
        "def k(x, n, B: tl.constexpr):\n"
        "    o = tl.arange(0, B)\n"
        "    tl.store(x + o, o.to(tl.float32), mask=o < n)\n"
        "x = torch.zeros(4, device='cuda')\n"
        "k[1](x, 4, B=4)\n"
    )
    with pytest.raises(TypeError, match="has no len"):
        sanitize.run_script(script)


def test_a_second_poison_pattern_shows_memory_returned_without_being_written(monkeypatch) -> None:
    import torch

    def never_written() -> list[float]:
        with sanitize.session():
            return torch.empty(2, device="cuda").tolist()

    first = never_written()
    monkeypatch.setenv("TTSEM_POISON", "second")
    second = never_written()
    assert first != second and second == [-12345.0, -12345.0]


def test_code_in_inductors_style_finds_its_grid_helper_and_its_allocator() -> None:
    import torch

    undo = sanitize.inductor_names()
    try:
        from torch._inductor.runtime.triton_heuristics import grid

        assert grid(1000)({"XBLOCK": 256}) == (4, 1, 1)
        assert grid(6, 1000)({"XBLOCK": 256, "YBLOCK": 4}) == (4, 2, 1)  # the last one is x
        with sanitize.session():
            buf = torch._C._dynamo.guards._empty_strided_cuda((2, 3), (3, 1), torch.float32)
            assert buf.shape == (2, 3) and buf.stride() == (3, 1) and bool(buf.isnan().all())
    finally:
        undo()


def test_a_kernel_warmed_up_and_launched_through_its_handle_is_judged(tmp_path: Path) -> None:
    # the idiom of the official softmax tutorial: warmup, read the register count, kernel[grid](...)
    script = tmp_path / "warm.py"
    script.write_text(
        "import torch, triton, triton.language as tl\n"
        "@triton.jit\n"
        "def double(out, x, n, B: tl.constexpr, num_stages: tl.constexpr):\n"
        "    o = tl.program_id(0) * B + tl.arange(0, B)\n"
        "    tl.store(out + o, tl.load(x + o, mask=o < n) * 2, mask=o < n)\n"
        "x = torch.arange(10, device='cuda', dtype=torch.float32)\n"
        "out = torch.empty_like(x)\n"
        "k = double.warmup(out, x, 10, B=16, num_stages=2, num_warps=4, grid=(1,))\n"
        "k._init_handles()\n"
        "programs = max(1, min(64 // k.n_regs, 1))\n"
        "assert k.metadata.shared > 0\n"
        "k[(programs, 1, 1)](out, x, 10, 16, 2)\n"
        "assert out.tolist() == [2.0 * i for i in range(10)]\n"
    )
    report = sanitize.run_script(script)
    assert (report.verdict, report.launches) == ("ok", 1)


def test_the_output_code_of_torch_compile_runs_here_and_its_race_is_seen() -> None:
    """Two files TorchInductor 2.9.1 wrote on a GPU (fixtures/inductor). In one,
    `x /= x.sum(0, keepdim=True); return x * 2.0` is fused into a kernel that reads the rows it is
    overwriting (on the device: 8.4 M of 12.6 M elements wrong at 3 x 2048 x 2048); the other is
    the out-of-place form, which is right."""
    import runpy

    pytest.importorskip("torch._inductor.runtime.triton_heuristics")
    import torch._inductor.utils as inductor_utils

    fixtures = Path(__file__).resolve().parent / "fixtures" / "inductor"
    once = inductor_utils.print_performance
    inductor_utils.print_performance = lambda fn, *a, **k: fn()  # type: ignore[assignment]
    undo = sanitize.inductor_names()
    try:
        with sanitize.session() as state:
            safe = runpy.run_path(str(fixtures / "safe_share_output_code.py"), run_name="code")
            safe["benchmark_compiled_module"](times=1, repeat=1)
            assert state.launches == 1
        with sanitize.session(), pytest.raises(sanitize.KernelFault) as stop:
            racy = runpy.run_path(str(fixtures / "channel_share_output_code.py"), run_name="code")
            racy["benchmark_compiled_module"](times=1, repeat=1)
        assert stop.value.fault.kind == "race"
        assert "triton_poi_fused_copy__div_mul_sum_0" in stop.value.fault.kernel
    finally:
        undo()
        inductor_utils.print_performance = once  # type: ignore[assignment]
