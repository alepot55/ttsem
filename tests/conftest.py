"""Builders for hand-made IR, so the ops can be tested without going through the parser.

Everything here constructs the dataclasses of `ttsem.ir_types` directly. When a test says
`op("arith.addi", ["i32", "i32"], "i32")` it is writing the same thing the generic form would
print, minus the syntax.

At the end, `planted`: a secret where a sandbox that lets the caller's environment or home
directory through would show it to a model-written answer.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ttsem.interp import Interp
from ttsem.ir_types import Block, Module, Op, Region, Type
from ttsem.memory import Memory

FLOAT_WIDTH = {"f16": 16, "bf16": 16, "f32": 32, "f64": 64}


def ty(spelling: str) -> Type:
    """A scalar type from its printed spelling: `i32`, `f32`, `bf16`, `f8E4M3FN`, `index`."""
    if spelling == "index":
        return Type("index", width=64, name=spelling)
    if spelling.startswith("i"):
        return Type("int", width=int(spelling[1:]), name=spelling)
    width = FLOAT_WIDTH.get(spelling, 8)
    return Type("float", width=width, name=spelling)


def tensor(shape: tuple[int, ...], elem: str | Type) -> Type:
    inner = ty(elem) if isinstance(elem, str) else elem
    return Type("tensor", shape=tuple(shape), elem=inner, name="tensor")


def ptr(elem: str | Type) -> Type:
    inner = ty(elem) if isinstance(elem, str) else elem
    return Type("ptr", elem=inner, width=64, name="ptr")


def memdesc(shape: tuple[int, ...], elem: str | Type) -> Type:
    inner = ty(elem) if isinstance(elem, str) else elem
    return Type("memdesc", shape=tuple(shape), elem=inner, name="memdesc")


def tensordesc(shape: tuple[int, ...], elem: str | Type) -> Type:
    inner = ty(elem) if isinstance(elem, str) else elem
    return Type("tensordesc", shape=tuple(shape), elem=inner, name="tensordesc")


def op(
    name: str,
    operand_types: list[Type] | None = None,
    result_types: list[Type] | Type | None = None,
    *,
    attrs: dict[str, object] | None = None,
    regions: list[Region] | None = None,
    successors: list[str] | None = None,
    results: list[str] | None = None,
    operands: list[str] | None = None,
) -> Op:
    """An op whose operands default to `%a0, %a1, ...` and results to `%r0, %r1, ...`."""
    rts = [result_types] if isinstance(result_types, Type) else list(result_types or [])
    ots = list(operand_types or [])
    return Op(
        name=name,
        results=results if results is not None else [f"%r{i}" for i in range(len(rts))],
        result_types=rts,
        operands=operands if operands is not None else [f"%a{i}" for i in range(len(ots))],
        operand_types=ots,
        attrs=dict(attrs or {}),
        regions=list(regions or []),
        successors=list(successors or []),
    )


def block(args: list[tuple[str, Type]], ops: list[Op], label: str | None = None) -> Block:
    return Block(label=label, args=args, ops=ops)


def region(*blocks: Block) -> Region:
    return Region(blocks=list(blocks))


def func(name: str, params: list[tuple[str, Type]], ops: list[Op]) -> Module:
    entry = Op(name="tt.func", attrs={"sym_name": name}, regions=[region(block(params, ops))])
    return Module(funcs={name: entry}, ops=[entry])


def interp(module: Module | None = None, memory: Memory | None = None, **kw: object) -> Interp:
    grid = kw.get("num_programs", (1, 1, 1))
    assert isinstance(grid, tuple)
    return Interp(module or Module(), memory or Memory(), grid)


def evaluate(
    target: Op,
    args: list[object] | None = None,
    memory: Memory | None = None,
    **kw: object,
) -> list[object]:
    """Evaluate one op on the given operand values and return its results."""
    machine = interp(memory=memory, **kw)
    pid = kw.get("program_id", (0, 0, 0))
    assert isinstance(pid, tuple)
    machine.program_id = pid
    machine.scopes = [dict(zip(target.operands, args or [], strict=True))]
    machine.eval_op(target)
    return [machine.value(n) for n in target.results]


def one(target: Op, args: list[object] | None = None, **kw: object) -> np.ndarray:
    result = evaluate(target, args, **kw)[0]
    assert isinstance(result, np.ndarray)
    return result


def buffer(memory: Memory, base: int, array: np.ndarray) -> np.ndarray:
    memory.register(base, array)
    return array


# --- a secret where a careless sandbox would show it ---------------------------------------------

LEAK = """
import json as _json
import os as _os
import pathlib as _pathlib
import pwd as _pwd


def _read(path):
    try:
        return _pathlib.Path(path).read_text()
    except OSError as exc:
        return "unreadable: " + type(exc).__name__


# what a model-written answer could copy into its report, written where the run may write (its
# working directory, OUT): the whole environment, a file of the caller's home directory, whether
# the real home's config directory is in sight, and the host's sockets under /run
_real = _pwd.getpwuid(_os.getuid()).pw_dir
_seen = {
    "environ": dict(_os.environ),
    "secret": _read(%r),
    "home": _os.path.expanduser("~"),
    "real_config": _os.path.exists(_os.path.join(_real, ".config")),
    "run": sorted(_os.listdir("/run")) if _os.path.isdir("/run") else [],
}
_pathlib.Path("leak.json").write_text(_json.dumps(_seen))
"""


@dataclass
class Planted:
    value: str
    home: Path
    file: Path

    def leak(self) -> str:
        """Python an answer runs at import: it copies what it can see of the secret into
        OUT/leak.json."""
        return LEAK % str(self.file)

    def check(self, out: Path) -> dict[str, Any]:
        """What the answer saw, after asserting it saw none of the secret: the variable is not in
        its environment, the file is out of its reach, HOME is not the caller's, the caller's real
        config directory is out of sight, and /run is empty."""
        seen: dict[str, Any] = json.loads((out / "leak.json").read_text())
        # the names only: a failure must not print the values of the caller's environment
        names = sorted(seen["environ"])
        assert "TTSEM_TEST_SECRET" not in names, names
        assert str(seen["secret"]).startswith("unreadable: "), seen["secret"]
        leaked = self.value in json.dumps(seen)
        assert not leaked, "the secret's value is in what the answer saw"
        assert seen["home"] != str(self.home)
        assert seen["real_config"] is False
        assert seen["run"] == [], seen["run"]
        return seen


@pytest.fixture
def planted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Planted:
    """A secret in the judge's own environment (`TTSEM_TEST_SECRET`) and in a file of its home
    directory (a fake one: HOME points at it), where a token would sit."""
    value = "ttsem-test-" + secrets.token_hex(8)
    home = tmp_path / "home"
    file = home / ".config" / "ttsem-test" / "token"
    file.parent.mkdir(parents=True)
    file.write_text(value)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TTSEM_TEST_SECRET", value)
    return Planted(value, home, file)
