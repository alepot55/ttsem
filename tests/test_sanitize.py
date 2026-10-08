"""`sanitize`: a Triton program run without a GPU, every launch executed by the semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    assert "too large to be checked for races" not in out  # no oversized launch here


def test_a_pytest_suite_reports_launches_too_large_for_the_race_check(capsys) -> None:
    """A launch over the race-check budget (over 4,000,000 accesses) is run and its values
    checked, so the test passes, but the session says how many such launches there were, in the
    terminal summary, rather than dropping the count without a word (item 7)."""
    code = sanitize.run_pytest(
        [str(AGENTS / "suite_big_launch.py"), "-q", "-p", "no:cacheprovider"]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "1 passed" in out
    assert "1 launch(es) too large to be checked for races" in out
    assert "4,000,000" in out


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


def test_memory_from_torch_empty_that_requires_grad_is_poisoned_too() -> None:
    """`torch.empty_like(x, requires_grad=True)` is a leaf that requires grad, which an in-place
    fill refuses; the poison is not an operation of the program, so autograd does not see it.
    TritonBench-T's `ones_like` and `logspace` answers allocate their output this way."""
    import torch

    with sanitize.session():
        x = torch.zeros(4, device="cuda")
        out = torch.empty_like(x, requires_grad=True)
    assert out.is_leaf and out.requires_grad
    assert torch.isnan(out.detach()).all()


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

    before = pretend_state()
    with sanitize.session():
        x = torch.zeros(4, device="cuda")
        assert x.is_cuda and x.device.type == "cuda"
        y = torch.ones(4, device=x.device) + x.to(x.device)  # a device read off a tensor works
        assert y.tolist() == [1.0] * 4 and torch.empty_like(x).shape == (4,)
        with torch.cuda.device(0):
            torch.cuda.set_device(0)
    assert pretend_state() == before


def pretend_state() -> tuple[Any, ...]:
    """What a tensor says of its device, and what `torch.Tensor` holds for it: outside any
    session (False, "cpu", None, None); under `python -m ttsem.sanitize pytest`, the session
    around the test's."""
    import torch

    x = torch.zeros(1)
    return (
        x.is_cuda,
        x.device.type,
        *(torch.Tensor.__dict__.get(k) for k in ("is_cuda", "device")),
    )


def test_a_session_inside_another_gives_the_outer_one_its_pretend_device_back() -> None:
    """As under `python -m ttsem.sanitize pytest` on a suite whose tests open a session: the
    inner session's exit leaves the outer one as it was, and the outer one's exit leaves torch
    as it was."""
    import torch

    before = pretend_state()
    with sanitize.session():
        with sanitize.session():
            assert torch.zeros(1, device="cuda").is_cuda
        x = torch.zeros(1, device="cuda")
        assert x.is_cuda and x.device.type == "cuda"
    assert pretend_state() == before


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


def test_the_matmul_tutorial_on_a_matrix_smaller_than_its_block_is_not_a_conflict(
    tmp_path: Path,
) -> None:
    """The tutorial wraps rows with `% M`, so with M = 48 and BLOCK_M = 64 two lanes load the
    same row and store the same product to one address, which is benign on the device (both
    lanes compute the same bits). The semantics used to read it as a conflicting store, because
    numpy's `matmul` summed the two copies of the row in different orders (fixed 24 Sep
    2026)."""
    script = tmp_path / "tutorial.py"
    script.write_text(
        "import torch, triton, triton.language as tl\n"
        "@triton.jit\n"
        "def mm(a, b, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):\n"
        "    rm = (tl.program_id(0) * BM + tl.arange(0, BM)) % M\n"
        "    rn = (tl.program_id(1) * BN + tl.arange(0, BN)) % N\n"
        "    rk = tl.arange(0, BK)\n"
        "    acc = tl.zeros((BM, BN), dtype=tl.float32)\n"
        "    for k in range(0, tl.cdiv(K, BK)):\n"
        "        x = tl.load(a + rm[:, None] * K + (k * BK + rk)[None, :])\n"
        "        y = tl.load(b + (k * BK + rk)[:, None] * N + rn[None, :])\n"
        "        acc += tl.dot(x, y)\n"
        "    tl.store(c + rm[:, None] * N + rn[None, :], acc, mask=rm[:, None] < M)\n"
        "torch.manual_seed(0)\n"
        "a = torch.randn(48, 64, device='cuda')\n"
        "b = torch.randn(64, 64, device='cuda')\n"
        "c = torch.empty(48, 64, device='cuda')\n"
        "mm[(1, 1)](a, b, c, 48, 64, 64, BM=64, BN=64, BK=32)\n"
        "assert torch.allclose(c, a @ b, atol=5e-2, rtol=1e-2)\n"
    )
    report = sanitize.run_script(script)
    assert (report.verdict, report.launches) == ("ok", 1), report


TUNED = """import triton, triton.language as tl
@triton.autotune(
    configs=[
        triton.Config({"BM": 128, "BK": 64}, num_stages=4),  # 131072 bytes on sm_90
        triton.Config({"BM": 16, "BK": 16}, num_stages=1),  # 1024 bytes
    ],
    key=["M"],
)
@triton.jit
def mm(a, b, c, M, BM: tl.constexpr, BK: tl.constexpr):
    r = tl.arange(0, BM) % M
    acc = tl.zeros((BM, BM), dtype=tl.float32)
    for k in range(0, M, BK):
        rk = k + tl.arange(0, BK)
        x = tl.load(a + r[:, None] * M + rk[None, :], mask=rk[None, :] < M, other=0.0)
        y = tl.load(b + rk[:, None] * M + r[None, :], mask=rk[:, None] < M, other=0.0)
        acc += tl.dot(x, y)
    tl.store(c + r[:, None] * M + r[None, :], acc)
"""


def test_a_config_needing_more_shared_memory_than_the_gpu_has_is_skipped_as_on_the_gpu(
    tmp_path: Path,
) -> None:
    """On a device, a launch whose kernel needs more shared memory than the device has raises
    `OutOfResources`, and an autotuner skips that configuration. With `shared_limit` the
    session does the same, by Triton's own count for that GPU: the configuration never runs,
    so nothing it would compute (or fault on) is judged."""
    import importlib.util

    import torch
    from triton.runtime.errors import OutOfResources

    source = tmp_path / "tuned.py"
    source.write_text(TUNED)
    spec = importlib.util.spec_from_file_location("tuned", source)
    assert spec is not None and spec.loader is not None
    tuned = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tuned)
    with sanitize.session() as state:
        torch.manual_seed(0)
        a = torch.randn(16, 16, device="cuda", dtype=torch.float16)
        b = torch.randn(16, 16, device="cuda", dtype=torch.float16)
        c = torch.empty(16, 16, device="cuda")
        state.shared_limit = (90, 101376)  # an sm_90 compile, an Ada-sized limit
        tuned.mm[(1,)](a, b, c, 16)
        assert tuned.mm.best_config.kwargs["BM"] == 16
        assert state.launches == 2  # the small config, benchmarked once and run once
        torch.testing.assert_close(c, a.float() @ b.float(), atol=1e-2, rtol=1e-2)
        big = {"BM": 128, "BK": 64, "num_stages": 4}
        with pytest.raises(OutOfResources, match="Required: 131072"):
            tuned.mm.fn.run(a, b, c, 16, grid=(1,), warmup=False, **big)
        state.shared_limit = None
        tuned.mm.fn.run(a, b, c, 16, grid=(1,), warmup=False, **big)  # no limit: it runs
        assert state.launches == 3


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


def test_a_kernel_the_user_wrote_runs_from_the_output_code_with_its_constexprs() -> None:
    """What torch 2.14 wrote for `y = x.clone(); reverse_kernel[grid](y, x, n, BLOCK=1024)` at 4096
    elements: the clone is gone and the kernel gets `x` twice. BLOCK is among the kernel's
    constants, not in a config, and Inductor's interpret path passes only the config. On the
    device this size comes out right, 2 M elements do not (1.7 M wrong on an RTX 4070)."""
    import runpy

    pytest.importorskip("torch._inductor.runtime.triton_heuristics")
    import torch._inductor.utils as inductor_utils

    fixtures = Path(__file__).resolve().parent / "fixtures" / "inductor"
    once = inductor_utils.print_performance
    inductor_utils.print_performance = lambda fn, *a, **k: fn()  # type: ignore[assignment]
    undo = sanitize.inductor_names()
    try:
        with sanitize.session(), pytest.raises(sanitize.KernelFault) as stop:
            code = runpy.run_path(
                str(fixtures / "user_kernel_clone_removed_output_code.py"), run_name="code"
            )
            code["benchmark_compiled_module"](code["get_args"](), times=1, repeat=1)
        assert stop.value.fault.kind == "race"
        assert "reverse_kernel" in stop.value.fault.kernel
    finally:
        undo()
        inductor_utils.print_performance = once  # type: ignore[assignment]


def test_the_scratch_buffer_of_a_decoupled_lookback_scan_is_not_a_race() -> None:
    """Inductor's split scan passes each block's partial to the next through a scratch buffer:
    a plain store, then a flag exchanged with release order, which the next block spins on with
    acquire order before its plain load. Run one after the other, the two plain accesses look
    like a race; the spin makes the order hold in every schedule."""
    import runpy

    pytest.importorskip("torch._inductor.runtime.triton_heuristics")
    import torch
    import torch._inductor.utils as inductor_utils

    fixtures = Path(__file__).resolve().parent / "fixtures" / "inductor"
    once = inductor_utils.print_performance
    inductor_utils.print_performance = lambda fn, *a, **k: fn()  # type: ignore[assignment]
    undo = sanitize.inductor_names()
    try:
        with sanitize.session() as state:
            code = runpy.run_path(str(fixtures / "split_scan_output_code.py"), run_name="code")
            (x,) = code["get_args"]()
            want = (x > 0.5).to(torch.int64).cumsum(0) * 2
            (got,) = code["call"]([x])
            assert state.launches == 1  # the scan and the multiply are one kernel
        assert torch.equal(got.cpu(), want.cpu())
    finally:
        undo()
        inductor_utils.print_performance = once  # type: ignore[assignment]


def test_a_kernel_that_writes_back_rows_it_did_not_change_does_not_race_with_their_readers() -> (
    None
):
    """What a compiler emits for `x[::2] += x[1::2]`: every instance stores its whole block of
    `x`, the odd rows with the values they had, while other instances load those rows."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def even_rows_plus_odd(x_ptr, n, cols, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < n
        row = i // cols
        own = tl.load(x_ptr + i, mask)
        below = tl.load(x_ptr + i + cols, mask & (row % 2 == 0), other=0.0)
        tl.store(x_ptr + i, tl.where(row % 2 == 0, own + below, own), mask)

    @triton.jit
    def every_row_plus_the_next(x_ptr, n, cols, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < n
        own = tl.load(x_ptr + i, mask)
        below = tl.load(x_ptr + i + cols, mask & (i + cols < n), other=0.0)
        tl.store(x_ptr + i, own + below, mask)

    with sanitize.session():
        x = torch.arange(64, dtype=torch.float32, device="cuda").reshape(8, 8) + 1.0
        want = x.clone()
        want[::2] += want[1::2].clone()
        even_rows_plus_odd[(8,)](x, 64, 8, BLOCK=8)
        assert torch.equal(x, want)
        with pytest.raises(sanitize.KernelFault) as stop:
            every_row_plus_the_next[(8,)](x, 64, 8, BLOCK=8)
        assert stop.value.fault.kind == "race"


def test_a_race_whose_first_store_leaves_the_bytes_it_found_is_reported() -> None:
    """Instance 0 stores 0.0 where 0.0 already is (a silent store), instance 1 stores 1.0 there:
    what the element holds in the end depends on the order, and the fault names both stores.
    TritonBench's `kv_cache_filling` reference writes its scales this way."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def pid_to_first(out_ptr):
        tl.store(out_ptr, tl.program_id(0).to(tl.float32))

    with sanitize.session():
        out = torch.zeros(4, dtype=torch.float32, device="cuda")
        with pytest.raises(sanitize.KernelFault) as stop:
            pid_to_first[(2,)](out)
    assert stop.value.fault.kind == "race"
    text = sanitize.message(stop.value.fault)
    # here one serial order already gives a wrong value, so the message must not claim that an
    # output comparison never sees a race
    assert "an output comparison may not see it" in text and "cannot see" not in text
    assert "(0, 0, 0) stores to it, (1, 0, 0) stores to it" in text


def test_a_recorded_race_lets_the_program_go_on_and_another_order_witnesses_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With `races="record"` the race is kept and the launch completes; the element then holds
    what the last instance stored, and `TTSEM_GRID_ORDER=reverse` makes that the other one: the
    result depends on the schedule, seen in the values without a GPU."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def pid_to_first(out_ptr):
        tl.store(out_ptr, tl.program_id(0).to(tl.float32))

    got = {}
    for order in ("", "reverse"):
        monkeypatch.setenv("TTSEM_GRID_ORDER", order)
        with sanitize.session(races="record") as state:
            out = torch.zeros(4, dtype=torch.float32, device="cuda")
            pid_to_first[(2,)](out)
        assert [fault.kind for fault in state.races] == ["race"]
        got[order] = float(out[0])
    assert got == {"": 1.0, "reverse": 0.0}
    monkeypatch.setenv("TTSEM_GRID_ORDER", "")
    with sanitize.session(races="off") as state:
        pid_to_first[(2,)](torch.zeros(4, dtype=torch.float32, device="cuda"))
    assert state.races == []


def test_a_block_cap_shrinks_the_block_of_a_direct_launch_whose_grid_reads_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`TTSEM_BLOCK_CAP` does for a launch written as `kernel[grid](..., XBLOCK=64)` (the style of
    older Inductor output, KernelBook's) what `inductor_names(block=...)` does for the autotuner:
    the block shrinks and the grid, which reads XBLOCK from the meta, grows to cover the same
    elements. A grid given as numbers is left alone, since a smaller block would leave elements
    out."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def stamp(out_ptr, n, XBLOCK: tl.constexpr):
        offs = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
        tl.store(out_ptr + offs, tl.program_id(0).to(tl.float32), mask=offs < n)

    def grid(meta):  # type: ignore[no-untyped-def]
        return (triton.cdiv(64, meta["XBLOCK"]),)

    got = {}
    for cap, launch_grid in (("", grid), ("16", grid), ("16", (1,))):
        monkeypatch.setenv("TTSEM_BLOCK_CAP", cap)
        with sanitize.session(races="off"):
            out = torch.full((64,), -1.0, device="cuda")
            stamp[launch_grid](out, 64, XBLOCK=64)
            got[(cap, callable(launch_grid))] = out.cpu().clone()
    assert torch.equal(got[("", True)], torch.zeros(64))
    assert torch.equal(got[("16", True)], torch.arange(4).repeat_interleave(16).float())
    assert torch.equal(got[("16", False)], torch.zeros(64))

def test_source_line_falls_back_to_the_kernels_own_file_when_the_loc_path_is_gone(
    tmp_path: Path,
) -> None:
    """When the `loc` baked into the IR points at a path this checkout does not have (a Triton
    cache filled from another checkout), the line's text cannot be read from it; the kernel's own
    file has the same line numbers, so it is read there instead (bug B)."""
    kernel = tmp_path / "here" / "k.py"
    kernel.parent.mkdir()
    kernel.write_text("import triton\n# a pad line\nx = tl.load(ptr)  # faulting line 3\n")
    gone = "/a/checkout/that/is/not/here/k.py"
    op = f'tt.load %x loc("{gone}":3:9)'
    # the loc path is unreadable: fall back to the kernel's own file at the same line
    path, line, text = sanitize._source_of(op, "", str(kernel))
    assert (line, text, path) == (3, "x = tl.load(ptr)  # faulting line 3", str(kernel))
    # without the kernel file there is nothing to read: the old behaviour, an empty source line
    path0, line0, text0 = sanitize._source_of(op, "", "")
    assert (line0, text0, path0) == (3, "", gone)


def test_a_fault_from_a_cache_filled_elsewhere_still_names_the_line(tmp_path: Path) -> None:
    """End to end: compile a no-mask kernel in one directory, then delete it and resolve the same
    op against a copy in another directory. The loc's own path is gone, but the copy's line
    numbers match, so the fault still carries its source line."""
    first = tmp_path / "checkoutA"
    first.mkdir()
    (first / "k.py").write_text((AGENTS / "add_no_load_mask.py").read_text())
    report = sanitize.run_program(first / "k.py")
    assert report.verdict == "oob_read" and report.fault is not None
    op, line, had = report.fault.op, report.fault.source_line, report.fault.source_text
    assert had  # with the file present, the line was read
    second = tmp_path / "checkoutB"
    second.mkdir()
    (second / "k.py").write_text((AGENTS / "add_no_load_mask.py").read_text())
    import shutil

    shutil.rmtree(first)  # the loc's own path no longer exists
    # resolve the recorded op: nothing without a fallback, the copy's line with it
    _, line_gone, text_gone = sanitize._source_of(op, "", "")
    _, line_back, text_back = sanitize._source_of(op, "", str(second / "k.py"))
    if text_gone == "":  # the loc held an absolute path that is now gone
        assert line_back == line and text_back == had
    else:  # a relative or still-present path: the line was there all along
        assert text_back == had
