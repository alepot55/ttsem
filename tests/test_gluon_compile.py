"""The plugin's recompile of a Gluon kernel goes through Gluon's own frontend."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from ttsem import harness

HERE = Path(__file__).resolve().parent


def test_a_gluon_kernel_recompiles_through_gluon_ast_source() -> None:
    pytest.importorskip("triton.experimental.gluon")
    spec = importlib.util.spec_from_file_location("gluon_copy", HERE / "fixtures" / "gluon_copy.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn, args, kwargs = module.launch_args()
    assert harness._source_class(fn).__name__ == "GluonASTSource"
    harness._install_fake_driver(90)
    asm = harness._compile_asm(fn, args, kwargs, harness.GPUTarget("cuda", 90, 32))
    assert "ttgir" in asm and "#ttg.blocked" in asm["ttgir"] and "ttir" not in asm
