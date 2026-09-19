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
