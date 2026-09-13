"""The plugin's own recompile must not leak warnings into the test whose launch it validates."""

from __future__ import annotations

import collections
import io
import types
import warnings

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
