"""Updates lost to interleaving: what the race rules see and no serial order of the instances does.

Each kernel reads, modifies and writes one shared address without an atomic. Run one instance after
another, in any order, every instance sees the previous one's store and the result is right; on a
device two instances can both load the old value and one update is lost. Another order of the
instances (`TTSEM_GRID_ORDER`) only permutes whole instances, so it cannot show it; the read-write
rule of `pidraces` reports it from one execution.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

from ttsem import sanitize  # noqa: E402

ORDERS = ("", "reverse", "shuffle:1", "shuffle:2")


def _kernels():  # type: ignore[no-untyped-def]
    import triton
    import triton.language as tl

    @triton.jit
    def count(hist_ptr, x_ptr):
        pid = tl.program_id(0)
        b = tl.load(x_ptr + pid)
        tl.store(hist_ptr + b, tl.load(hist_ptr + b) + 1)

    @triton.jit
    def accumulate(out_ptr, x_ptr, N: tl.constexpr):
        pid = tl.program_id(0)
        cols = tl.arange(0, N)
        row = tl.load(x_ptr + pid * N + cols)
        tl.store(out_ptr + cols, tl.load(out_ptr + cols) + row)

    @triton.jit
    def running_max(out_ptr, x_ptr):
        pid = tl.program_id(0)
        tl.store(out_ptr, tl.maximum(tl.load(out_ptr), tl.load(x_ptr + pid)))

    return count, accumulate, running_max


def _cases():  # type: ignore[no-untyped-def]
    """name -> (launch, a fresh output, the right result); the inputs are made inside the session,
    which is what maps `device="cuda"` to the CPU."""
    import torch

    count, accumulate, running_max = _kernels()
    rows = torch.arange(32, dtype=torch.float32).reshape(8, 4)

    def bins():  # type: ignore[no-untyped-def]
        return torch.tensor([0, 1, 0, 0, 1, 0, 1, 0], dtype=torch.int32, device="cuda")

    def values():  # type: ignore[no-untyped-def]
        return torch.tensor([3.0, 9.0, 1.0, 7.0, 2.0, 8.0], device="cuda")

    return {
        "histogram": (
            lambda out: count[(8,)](out, bins()),
            lambda: torch.zeros(2, dtype=torch.int32, device="cuda"),
            torch.tensor([5, 3], dtype=torch.int32),
        ),
        "row sum": (
            lambda out: accumulate[(8,)](out, rows.to("cuda"), N=4),
            lambda: torch.zeros(4, dtype=torch.float32, device="cuda"),
            rows.sum(0),
        ),
        "running max": (
            lambda out: running_max[(6,)](out, values()),
            lambda: torch.zeros(1, dtype=torch.float32, device="cuda"),
            torch.tensor([9.0]),
        ),
    }


@pytest.mark.parametrize("name", ["histogram", "row sum", "running max"])
def test_a_lost_update_is_a_race_that_every_serial_order_gets_right(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    launch, fresh, want = _cases()[name]
    with sanitize.session(), pytest.raises(sanitize.KernelFault) as stop:
        launch(fresh())
    assert stop.value.fault.kind == "race"
    for order in ORDERS:
        monkeypatch.setenv("TTSEM_GRID_ORDER", order)
        with sanitize.session(races="off"):
            out = fresh()
            launch(out)
            got = out.cpu().clone()
        assert torch.equal(got, want), (name, order)


def test_a_silent_store_next_to_atomic_updates_is_a_race_in_every_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Instance 0 stores 0 into an output that holds 0, the others add into it atomically. Run
    first, the store is silent; run last, it wipes the sum. Whether a store is silent depends on
    the order that was executed, so the rule for a store next to an atomic update counts it."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def reset_or_add(out_ptr):
        pid = tl.program_id(0)
        if pid == 0:
            tl.store(out_ptr, 0.0)
        else:
            tl.atomic_add(out_ptr, 1.0)

    for order in ("", "reverse"):
        monkeypatch.setenv("TTSEM_GRID_ORDER", order)
        with sanitize.session(), pytest.raises(sanitize.KernelFault) as stop:
            reset_or_add[(4,)](torch.zeros(1, dtype=torch.float32, device="cuda"))
        assert stop.value.fault.kind == "race", order
