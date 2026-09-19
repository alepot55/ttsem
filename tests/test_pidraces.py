"""Races between program instances, from a hand-made access log."""

from __future__ import annotations

import numpy as np
from ttsem import pidraces


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
    from ttsem.memory import Memory

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
