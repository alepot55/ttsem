"""The plugin's own recompile must not leak warnings into the test whose launch it validates."""

from __future__ import annotations

import collections
import io
import json
import types
import warnings

import pytest

from ttsem import pytest_ttsem


def test_validate_keeps_its_own_warnings_out_of_the_test(monkeypatch) -> None:
    def noisy(*_args, **_kwargs):
        warnings.warn("ttsem recompile noise", DeprecationWarning, stacklevel=1)
        raise RuntimeError("stop here")

    monkeypatch.setattr(pytest_ttsem.harness, "ir_for_launch", noisy)
    monkeypatch.setattr(pytest_ttsem, "_target", lambda: None)
    log = io.StringIO()
    for key, value in dict(
        stage="ttgir",
        triton_opt=None,
        dump=None,
        below=None,
        log=log,
        counts=collections.Counter(),
        nodeid="t.py::test_x",
        launch_idx=0,
        max_launches=10**9,
    ).items():
        monkeypatch.setitem(pytest_ttsem._state, key, value)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pytest_ttsem._validate(types.SimpleNamespace(fn_name="k"))
    assert caught == []
    assert '"verdict": "error"' in log.getvalue()
    assert "stop here" in log.getvalue()


def test_an_instrumented_launch_is_unsupported_not_compared(monkeypatch) -> None:
    def must_not_run(*_args, **_kwargs):
        raise AssertionError("the recompile must not run under an instrumentation mode")

    monkeypatch.setattr(pytest_ttsem.harness, "ir_for_launch", must_not_run)
    monkeypatch.setattr(pytest_ttsem, "_target", lambda: None)
    monkeypatch.setattr(pytest_ttsem, "_instrumentation_mode", lambda: "fpsan")
    log = io.StringIO()
    counts: collections.Counter[str] = collections.Counter()
    for key, value in dict(
        stage="ttgir",
        triton_opt=None,
        dump=None,
        below=None,
        log=log,
        counts=counts,
        nodeid="t.py::test_fpsan",
        launch_idx=0,
        max_launches=10**9,
    ).items():
        monkeypatch.setitem(pytest_ttsem._state, key, value)
    pytest_ttsem._validate(types.SimpleNamespace(fn_name="k"))
    line = json.loads(log.getvalue())
    assert line["verdict"] == "unsupported"
    assert line["unsupported"] == ["instrumentation mode fpsan"]
    assert counts["unsupported"] == 1 and pytest_ttsem._state["launch_idx"] == 1


def test_the_instrumentation_mode_reads_the_knob(monkeypatch) -> None:
    knobs = pytest.importorskip("triton.knobs")  # the no-wheel CI has no triton

    monkeypatch.setattr(knobs.compilation, "instrumentation_mode", "consan")
    assert pytest_ttsem._instrumentation_mode() == "consan"
    monkeypatch.setattr(knobs.compilation, "instrumentation_mode", "")
    assert pytest_ttsem._instrumentation_mode() == ""
