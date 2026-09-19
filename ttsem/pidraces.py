"""Races between the program instances of one launch, from the accesses the semantics made.

A launch is a grid of program instances that run in any order and at the same time. The
level-1 semantics runs them one after the other, which is *one* of the schedules: a kernel whose
instances touch the same bytes gets the answer of that schedule, and a test that compares
outputs passes. With `Memory.access_log` on, every load, store and atomic update is kept with
the instance that made it, and this module looks for the pairs that make the result depend on
the schedule:

- two instances store different bytes to the same address;
- one instance loads an address another one stores to or updates atomically;
- one instance stores to an address another one updates atomically.

Two atomic updates of one address are not a race (that is what they are for), nor are two loads.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

Entry = tuple[Any, str, str, np.ndarray, int, "np.ndarray | None"]


@dataclasses.dataclass
class PidRace:
    kind: str  # "write-write" | "read-write" | "write-atomic"
    address: int  # first byte the two accesses share
    first: tuple[Any, str, str]  # (program instance, access kind r|w|a, op text)
    second: tuple[Any, str, str]
    shared_bytes: int  # how many byte addresses are in a conflict of this kind


def _expand(log: list[Entry]) -> tuple[np.ndarray, ...]:
    """Every access as byte rows: address, instance index, kind code, entry index, byte written."""
    agents: dict[Any, int] = {}
    addr, who, kind, entry, val = [], [], [], [], []
    codes = {"r": 0, "w": 1, "a": 2}
    for index, (agent, k, _op, addrs, itemsize, raw) in enumerate(log):
        ident = agents.setdefault(agent, len(agents))
        span = (addrs[:, None] + np.arange(itemsize, dtype=np.int64)[None, :]).reshape(-1)
        addr.append(span)
        who.append(np.full(span.size, ident, dtype=np.int64))
        kind.append(np.full(span.size, codes[k], dtype=np.int8))
        entry.append(np.full(span.size, index, dtype=np.int64))
        val.append(
            raw.reshape(-1).astype(np.int16) if raw is not None else np.zeros(span.size, np.int16)
        )
    if not addr:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty.astype(np.int8), empty, empty.astype(np.int16)
    return tuple(np.concatenate(x) for x in (addr, who, kind, entry, val))


def _spread(keys: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(unique keys, min of `values` per key, max of `values` per key)."""
    order = np.argsort(keys, kind="stable")
    k, v = keys[order], values[order]
    starts = np.flatnonzero(np.concatenate(([True], k[1:] != k[:-1])))
    return k[starts], np.minimum.reduceat(v, starts), np.maximum.reduceat(v, starts)


def find_race(log: list[Entry] | None) -> PidRace | None:
    if not log:
        return None
    addr, who, kind, entry, val = _expand(log)
    if addr.size == 0 or np.unique(who).size < 2:
        return None
    found: list[tuple[str, np.ndarray, int, int]] = []  # (kind, conflict addrs, code a, code b)

    w = kind == 1
    if w.any():
        keys, lo, hi = _spread(addr[w], who[w])
        _, vlo, vhi = _spread(addr[w], val[w])
        clash = keys[(lo != hi) & (vlo != vhi)]
        if clash.size:
            found.append(("write-write", clash, 1, 1))

    def cross(a_mask: np.ndarray, b_mask: np.ndarray) -> np.ndarray:
        """Addresses touched under `a_mask` by one instance and under `b_mask` by another."""
        if not a_mask.any() or not b_mask.any():
            return np.zeros(0, dtype=np.int64)
        ka, alo, ahi = _spread(addr[a_mask], who[a_mask])
        kb, blo, bhi = _spread(addr[b_mask], who[b_mask])
        common, ia, ib = np.intersect1d(ka, kb, assume_unique=True, return_indices=True)
        same = (alo[ia] == ahi[ia]) & (blo[ib] == bhi[ib]) & (alo[ia] == blo[ib])
        return common[~same]

    reads, atomics = kind == 0, kind == 2
    rw = cross(reads, w | atomics)
    if rw.size:
        found.append(("read-write", rw, 0, -1))
    wa = cross(w, atomics)
    if wa.size:
        found.append(("write-atomic", wa, 1, 2))
    if not found:
        return None

    name, clash, code_a, code_b = found[0]
    at = int(clash.min())
    here = addr == at
    first_rows = np.flatnonzero(here & (kind == code_a))
    other_kinds = (kind != 0) if code_b == -1 else (kind == code_b)
    a = int(first_rows[0])
    second_rows = np.flatnonzero(here & other_kinds & (who != who[a]))
    if second_rows.size == 0:  # the first row was not one of the clashing instances
        for cand in first_rows[1:]:
            second_rows = np.flatnonzero(here & other_kinds & (who != who[cand]))
            if second_rows.size:
                a = int(cand)
                break
    b = int(second_rows[0])
    ea, eb = log[int(entry[a])], log[int(entry[b])]
    return PidRace(name, at, (ea[0], ea[1], ea[2]), (eb[0], eb[1], eb[2]), int(clash.size))
