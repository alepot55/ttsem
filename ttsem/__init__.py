"""ttsem: an executable semantics of Triton's TTIR and TTGIR, and a per-pass validator.

The package has three layers, each usable on its own:

* **Level 1, values.** :func:`ttsem.mlir.parse` reads a generic-form MLIR module and
  :class:`ttsem.interp.Interp` executes it over ``numpy`` arrays. No GPU, no Triton install,
  no ``triton-opt``: the module text is all that is needed.
* **Level 2, layouts.** :class:`ttsem.interp2.LayoutInterp` and
  :func:`ttsem.interp2.check_module` add the linear-layout model, so a TTGIR module can be
  checked for layout violations without running it.
* **Level 3, concurrency.** :func:`ttsem.races.detect` replays one execution and reports
  shared-memory races between the warps of a CTA.

On top of those, :func:`ttsem.validate.validate_stages` takes the modules a compilation
produced before every pass and names the first pass whose output the semantics reads
differently from the device, and :func:`ttsem.minimize.reduce` shrinks that module to the
smallest one on which the pass still changes the meaning.

Everything above level 3 needs a Triton install, because a launch has to be recorded and
compiled to get the stages in the first place; those names are imported lazily, so importing
``ttsem`` with nothing but ``numpy`` installed works and gives you the interpreter.

The whole-program entry point keeps its module name rather than being re-exported here, so
that ``from ttsem import validate`` is unambiguously the submodule: call
``ttsem.validate.validate(program, record)`` for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ttsem.interp import Interp
from ttsem.interp2 import LayoutInterp, check_module, launch_config, module_attrs
from ttsem.ir_types import Block, Module, Op, Region, Type
from ttsem.memory import Memory, MemoryFault
from ttsem.mlir import NotGeneric, ParseError, parse, to_generic, unparse
from ttsem.races import detect
from ttsem.values import Poison, Unsupported, Value

__version__ = "0.1.0"

# The names below pull in `triton`, so they are resolved on first use rather than at import.
_LAZY = {
    "LaunchRecord": ("ttsem.harness", "LaunchRecord"),
    "capture_launches": ("ttsem.harness", "capture_launches"),
    "compare": ("ttsem.harness", "compare"),
    "dump_for_launch": ("ttsem.harness", "dump_for_launch"),
    "execute": ("ttsem.harness", "execute"),
    "ir_for_launch": ("ttsem.harness", "ir_for_launch"),
    "run_launch": ("ttsem.harness", "run_launch"),
    "PassResult": ("ttsem.validate", "PassResult"),
    "Report": ("ttsem.validate", "Report"),
    "capture_dump": ("ttsem.validate", "capture_dump"),
    "culprit_of": ("ttsem.validate", "culprit_of"),
    "split_dump": ("ttsem.validate", "split_dump"),
    "validate_stages": ("ttsem.validate", "validate_stages"),
    "reduce": ("ttsem.minimize", "reduce"),
}

if TYPE_CHECKING:  # so that type checkers and editors see the same API
    from ttsem.harness import (
        LaunchRecord,
        capture_launches,
        compare,
        dump_for_launch,
        execute,
        ir_for_launch,
        run_launch,
    )
    from ttsem.minimize import reduce
    from ttsem.validate import (
        PassResult,
        Report,
        capture_dump,
        culprit_of,
        split_dump,
        validate_stages,
    )


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "Block",
    "Interp",
    "LaunchRecord",
    "LayoutInterp",
    "Memory",
    "MemoryFault",
    "Module",
    "NotGeneric",
    "Op",
    "ParseError",
    "PassResult",
    "Poison",
    "Region",
    "Report",
    "Type",
    "Unsupported",
    "Value",
    "__version__",
    "capture_dump",
    "capture_launches",
    "check_module",
    "compare",
    "culprit_of",
    "detect",
    "dump_for_launch",
    "execute",
    "ir_for_launch",
    "launch_config",
    "module_attrs",
    "parse",
    "reduce",
    "run_launch",
    "split_dump",
    "to_generic",
    "unparse",
    "validate_stages",
]
