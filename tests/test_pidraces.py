"""Races between program instances, from a hand-made access log."""

from __future__ import annotations

import numpy as np

from ttsem import pidraces
from ttsem.memory import AccessSpans, Memory


def _e(agent: int, kind: str, addrs: list[int], raw: list[int] | None = None) -> pidraces.Entry:
    written = None if raw is None else np.array(raw, dtype=np.uint8).reshape(len(addrs), 1)
    return ((agent, 0, 0), kind, f"op of {agent}", np.array(addrs, dtype=np.int64), 1, written)


def test_disjoint_instances_do_not_race() -> None:
    log = [
        _e(0, "r", [0, 1]),
        _e(0, "w", [8, 9], [1, 2]),
        _e(1, "r", [2, 3]),
        _e(1, "w", [10, 11], [3, 4]),
    ]
    assert pidraces.find_race(log) is None


def test_two_instances_storing_different_bytes_to_one_address_race() -> None:
    race = pidraces.find_race([_e(0, "w", [8], [1]), _e(1, "w", [8], [2])])
    assert race is not None and race.kind == "write-write" and race.address == 8


def test_the_same_bytes_from_two_instances_are_not_a_race() -> None:
    assert pidraces.find_race([_e(0, "w", [8], [7]), _e(1, "w", [8], [7])]) is None


def test_instances_that_each_store_the_same_sequence_leave_the_same_bytes() -> None:
    # every instance writes 1 then 2: whatever the schedule, the last store is somebody's 2
    same = [_e(0, "w", [8], [1]), _e(0, "w", [8], [2]), _e(1, "w", [8], [1]), _e(1, "w", [8], [2])]
    assert pidraces.find_race(same) is None
    differ = [
        _e(0, "w", [8], [2]),
        _e(0, "w", [8], [1]),
        _e(1, "w", [8], [1]),
        _e(1, "w", [8], [2]),
    ]
    race = pidraces.find_race(differ)
    assert race is not None and race.kind == "write-write"


def test_a_load_of_what_another_instance_stores_is_a_race() -> None:
    race = pidraces.find_race(
        [_e(0, "r", [8]), _e(0, "w", [8], [1]), _e(1, "r", [8]), _e(1, "w", [8], [1])]
    )
    assert race is not None and race.kind == "read-write"
    assert race.first[0] != race.second[0]


def test_an_instance_reading_back_its_own_store_is_not_a_race() -> None:
    assert pidraces.find_race([_e(0, "w", [8], [1]), _e(0, "r", [8]), _e(1, "w", [9], [1])]) is None


def test_atomic_updates_of_one_address_are_not_a_race_but_a_plain_store_next_to_them_is() -> None:
    assert pidraces.find_race([_e(0, "a", [8]), _e(1, "a", [8]), _e(2, "a", [8])]) is None
    race = pidraces.find_race([_e(0, "a", [8]), _e(1, "w", [8], [0])])
    assert race is not None and race.kind == "write-atomic"


def test_elements_of_one_size_are_compared_whole_not_byte_by_byte() -> None:
    four = np.array([[1, 0, 0, 0]], dtype=np.uint8)
    other = np.array([[2, 0, 0, 0]], dtype=np.uint8)
    a = ((0, 0, 0), "w", "store a", np.array([64], dtype=np.int64), 4, four)
    b = ((1, 0, 0), "w", "store b", np.array([64], dtype=np.int64), 4, other)
    race = pidraces.find_race([a, b])
    assert race is not None and race.kind == "write-write" and race.address == 64
    assert race.shared_bytes == 4
    same = ((1, 0, 0), "w", "store b", np.array([64], dtype=np.int64), 4, four)
    assert pidraces.find_race([a, same]) is None


def test_the_access_log_stops_at_its_budget_and_says_so() -> None:
    memory = Memory()
    memory.register(4096, np.zeros(64, dtype=np.int32))
    memory.access_log = []
    memory.access_budget = 40
    addrs = 4096 + 4 * np.arange(32, dtype=np.int64)
    memory.load(addrs, None, None, np.dtype(np.int32))
    assert memory.access_log is not None and len(memory.access_log) == 1
    memory.load(addrs, None, None, np.dtype(np.int32))  # 64 elements logged: over the budget
    assert memory.access_log is None and memory.access_overflow


def test_the_range_check_does_not_hide_a_race_inside_interleaved_ranges() -> None:
    """Two instances whose address ranges interleave (strided outputs) are looked at closely,
    and a shared element among them is still found; without one, still no race."""
    a = _e(0, "w", [0, 8, 16], [1, 1, 1])
    b = _e(1, "w", [4, 12, 16], [2, 2, 2])
    race = pidraces.find_race([a, b])
    assert race is not None and race.address == 16
    c = _e(1, "w", [4, 12, 20], [2, 2, 2])
    assert pidraces.find_race([a, c]) is None


def test_a_load_by_one_instance_inside_what_another_stores_is_found_past_the_range_check() -> None:
    log = [_e(0, "w", [100, 101, 102], [1, 1, 1]), _e(1, "w", [200], [1]), _e(1, "r", [101])]
    race = pidraces.find_race(log)
    assert race is not None and race.kind == "read-write" and race.address == 101
    quiet = [_e(0, "w", [100, 101, 102], [1, 1, 1]), _e(1, "w", [200], [1]), _e(1, "r", [50, 300])]
    assert pidraces.find_race(quiet) is None


def test_a_store_of_the_bytes_already_there_is_no_conflict_for_a_load() -> None:
    # instance 1 writes back what address 8 already held (kind `s`): instance 0 loads the same
    # bytes before and after it
    assert pidraces.find_race([_e(0, "r", [8]), _e(1, "s", [8], [5]), _e(1, "w", [9], [1])]) is None
    race = pidraces.find_race([_e(0, "r", [8]), _e(1, "w", [8], [6])])
    assert race is not None and race.kind == "read-write"


def test_a_silent_store_still_counts_among_the_last_stores_of_its_address() -> None:
    # instance 0 rewrites the 5 that was there, instance 1 leaves a 6: the order decides
    race = pidraces.find_race([_e(0, "s", [8], [5]), _e(1, "w", [8], [6])])
    assert race is not None and race.kind == "write-write"


def _spans_of(log: list[pidraces.Entry]) -> AccessSpans:
    spans = AccessSpans()
    for agent, kind, _op, addrs, itemsize, _raw in log:
        spans.add(agent, kind != "r", int(addrs.min()), int(addrs.max()) + itemsize - 1)
    return spans


def _may_race(spans: AccessSpans) -> bool:
    return pidraces.spans_may_race(spans.lo, spans.hi, spans.who, spans.write)


def test_spans_rule_out_a_race_only_where_no_two_instances_ranges_meet() -> None:
    """Past the log's budget only each access's range is left: disjoint ranges per instance (all
    of them reading the same input nobody writes) rule a race out; a range of one instance that
    meets another's write does not, even with no shared byte (interleaved rows)."""
    disjoint = [_e(0, "r", [0, 1]), _e(0, "w", [8, 9], [1, 2]), _e(1, "r", [0, 1])]
    disjoint.append(_e(1, "w", [10, 11], [3, 4]))
    assert not _may_race(_spans_of(disjoint))
    interleaved = [_e(0, "w", [0, 8, 16], [1, 1, 1]), _e(1, "w", [4, 12, 20], [2, 2, 2])]
    assert _may_race(_spans_of(interleaved)) and pidraces.find_race(interleaved) is None
    assert _may_race(_spans_of([_e(0, "w", [8], [1]), _e(1, "r", [8])]))


def test_the_spans_of_one_instance_merge_where_they_touch() -> None:
    spans = AccessSpans()
    for start in range(0, 4096, 64):  # a loop that writes one row after another, then reads
        spans.add((0, 0, 0), True, start, start + 63)
    spans.add((0, 0, 0), False, 0, 15)
    spans.add((1, 0, 0), True, 4096, 4159)
    assert (list(spans.lo), list(spans.hi), list(spans.who)) == (
        [0, 0, 4096],
        [4095, 15, 4159],
        [0, 0, 1],
    )


def test_spans_never_rule_out_a_race_the_exact_check_finds() -> None:
    """Soundness, over random logs of few instances on few addresses: whenever the spans say no
    race is possible, the exact check finds none."""
    rng = np.random.default_rng(7)
    ruled_out = 0
    for _ in range(3000):
        log = []
        for _ in range(int(rng.integers(1, 7))):
            agent = int(rng.integers(0, 3))
            kind = str(rng.choice(["r", "w", "a", "s"]))
            start = int(rng.integers(0, 40))
            addrs = sorted({start + int(d) for d in rng.integers(0, 6, size=3)})
            raw = None if kind in "ra" else [int(v) for v in rng.integers(0, 2, size=len(addrs))]
            log.append(_e(agent, kind, addrs, raw))
        log.sort(key=lambda entry: entry[0])  # instances run one after the other
        if not _may_race(_spans_of(log)):
            ruled_out += 1
            assert pidraces.find_race(log) is None, log
    assert ruled_out > 100  # the property was exercised


def test_past_the_budget_the_memory_keeps_the_spans() -> None:
    memory = Memory()
    memory.register(4096, np.zeros(64, dtype=np.int32))
    memory.trace()
    memory.access_budget = 40
    for pid in range(2):  # two instances, each loading its own half twice
        memory.agent = (pid, 0, 0)
        addrs = 4096 + 128 * pid + 4 * np.arange(32, dtype=np.int64)
        memory.load(addrs, None, None, np.dtype(np.int32))
        memory.load(addrs, None, None, np.dtype(np.int32))
    spans = memory.access_spans
    assert memory.access_log is None and memory.access_overflow and spans is not None
    assert (list(spans.lo), list(spans.hi), list(spans.write)) == (
        [4096, 4224],
        [4223, 4351],
        [0, 0],
    )
    memory.agent = (2, 0, 0)
    memory.store(np.array([4100], dtype=np.int64), np.array([1], dtype=np.int32), None)
    assert _may_race(spans)  # a write into what instance 0 reads
