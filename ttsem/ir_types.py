"""The parsed-IR dataclasses of `DESIGN.md`, in one place so that the parser and the
interpreter agree on them without either importing the other.

`mlir.py` builds these; `interp.py` and `ops.py` only read them. Nothing here interprets a
layout: `Type.encoding` keeps the raw text and level 1 never looks at it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Type:
    """A parsed MLIR type.

    `kind` is one of "tensor", "ptr", "int", "float", "index", "memdesc", "tensordesc",
    "other". `name` is the printed spelling of the scalar type ("bf16", "f8E4M3FN", "i1"),
    which is what the numeric conventions are keyed on.
    """

    kind: str
    shape: tuple[int, ...] | None = None
    elem: Type | None = None
    width: int | None = None
    name: str = ""
    encoding: str | None = None

    @property
    def scalar(self) -> Type:
        """The element type of a tensor or memdesc, or the type itself."""
        if self.kind in ("tensor", "vector", "memdesc", "tensordesc") and self.elem is not None:
            return self.elem.scalar
        return self


@dataclass(frozen=True)
class FloatBits:
    """A float literal MLIR printed in hexadecimal (`0x7FC00000`, `0x7FC0`, `0xFF800000`).

    MLIR falls back to hex whenever the shortest decimal would not round-trip, which is every
    NaN and every infinity, so this is a raw bit pattern of the element type and not a number:
    keeping the bits means a NaN payload and the sign of an infinity survive the parse.
    """

    bits: int
    width: int  # of the element type the literal was printed for


@dataclass
class DenseAttr:
    """A `dense<...>` constant: `value` is a scalar when splat, else a flat list.

    A scalar or a list entry is an `int`, a `float`, a `bool`, or a `FloatBits` when MLIR
    printed the element in hex.
    """

    value: object
    elem_type: Type | None = None
    shape: tuple[int, ...] | None = None


@dataclass
class Op:
    name: str
    results: list[str] = field(default_factory=list)
    result_types: list[Type] = field(default_factory=list)
    operands: list[str] = field(default_factory=list)
    operand_types: list[Type] = field(default_factory=list)
    attrs: dict[str, object] = field(default_factory=dict)
    regions: list[Region] = field(default_factory=list)
    successors: list[str] = field(default_factory=list)
    loc: str | None = None
    text: str | None = None


@dataclass
class Block:
    label: str | None = None
    args: list[tuple[str, Type]] = field(default_factory=list)
    ops: list[Op] = field(default_factory=list)


@dataclass
class Region:
    blocks: list[Block] = field(default_factory=list)


@dataclass
class Module:
    funcs: dict[str, Op] = field(default_factory=dict)  # every `*.func` op, by symbol
    ops: list[Op] = field(default_factory=list)
    attrs: dict[str, object] = field(default_factory=dict)  # the builtin.module's attributes
