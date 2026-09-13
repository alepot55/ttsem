"""The mbarrier phase model and the cooperative scheduler of the partitions."""

from __future__ import annotations

import pytest

from ttsem.interp import Interp, Scheduler
from ttsem.ir_types import Module
from ttsem.memory import Memory
from ttsem.values import Barrier, Unsupported


def test_a_fresh_barrier_passes_parity_one_and_blocks_parity_zero() -> None:
    bar = Barrier(count=1, pending=1)
    assert bar.passes(1) and not bar.passes(0)
    bar.expect(2048)  # the arrive of `mbarrier.arrive.expect_tx`: bytes still outstanding
    assert not bar.passes(0)
    bar.complete_tx(2048)
    assert bar.passes(0) and not bar.passes(1)
    bar.arrive()  # the next phase, two arrivals would be needed with count=2
    assert bar.passes(1) and bar.completed == 2


def test_a_barrier_with_two_arrivals_per_phase_needs_both() -> None:
    bar = Barrier(count=2, pending=2)
    bar.arrive()
    assert not bar.passes(0)
    bar.arrive()
    assert bar.passes(0)


def _interp() -> Interp:
    return Interp(Module(funcs={}, attrs={}), Memory())


def test_the_scheduler_hands_the_turn_over_until_the_wait_holds() -> None:
    interp = _interp()
    sched = Scheduler(interp)
    interp.scheduler = sched
    flag = {"ready": False}
    order: list[str] = []

    def producer() -> None:
        order.append("producer runs")
        flag["ready"] = True
        interp.progress()

    sched.start([("producer", producer)], [])
    interp.block_until(lambda: flag["ready"], "consumer")  # the default partition waits first
    order.append("consumer resumes")
    sched.join()
    sched.abort()
    assert order == ["producer runs", "consumer resumes"]


def test_a_wait_nobody_can_satisfy_is_reported_as_a_deadlock() -> None:
    interp = _interp()
    sched = Scheduler(interp)
    interp.scheduler = sched
    sched.start(
        [("worker", lambda: interp.block_until(lambda: False, "wait_barrier parity 0"))], []
    )
    with pytest.raises(Unsupported, match="deadlock"):
        sched.join()
    sched.abort()


def test_a_failing_partition_surfaces_its_own_error() -> None:
    interp = _interp()
    sched = Scheduler(interp)
    interp.scheduler = sched

    def worker() -> None:
        raise Unsupported("tt.gather", "not modelled")

    sched.start([("worker", worker)], [])
    with pytest.raises(Unsupported, match="tt.gather"):
        sched.join()
    sched.abort()


def test_outside_a_warp_specialize_a_failing_wait_is_unsupported() -> None:
    interp = _interp()
    with pytest.raises(Unsupported, match="nothing before it completes"):
        interp.block_until(lambda: False, "wait_barrier parity 0")
    interp.block_until(lambda: True, "wait_barrier parity 1")  # a satisfied wait is fine
