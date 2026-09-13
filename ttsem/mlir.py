"""A parser for MLIR's generic textual form, as printed by
`triton-opt --mlir-print-op-generic --mlir-print-local-scope`.

The generic form is regular: every op is
`%r = "dialect.op"(%a, %b) <{attr = v}> ({region}) {attr = v} : (T, T) -> (T) loc(...)`.
This module is a real tokenizer plus a recursive-descent parser over that grammar, not a
regex-over-lines scanner: brackets nest and are matched by tracking depth over tokens, and a
handful of constructs (dialect layout attributes, `dense<...>` payloads, function-type
attribute values) are captured as raw source text and interpreted separately, or not at all.

The dataclasses (`Type`, `DenseAttr`, `Op`, `Block`, `Region`, `Module`) live in `ir_types.py`
so that the interpreter and this parser agree on them without either importing the other; this
module imports and re-exports them.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import NamedTuple

from ttsem.ir_types import Block, DenseAttr, FloatBits, Module, Op, Region, Type

__all__ = [
    "Type",
    "DenseAttr",
    "FloatBits",
    "Op",
    "Block",
    "Region",
    "Module",
    "ParseError",
    "NotGeneric",
    "parse",
    "unparse",
    "to_generic",
    "tokenize",
]


class ParseError(Exception):
    """A syntax error in the MLIR generic text being parsed."""


class NotGeneric(Exception):
    """`to_generic` was given non-generic text and no `triton_opt` to convert it."""


# --------------------------------------------------------------------------- tokenizer


class Token(NamedTuple):
    kind: str
    value: str
    start: int
    end: int


_TOKEN_SPEC: list[tuple[str, str]] = [
    ("STRING", r'"(?:\\.|[^"\\])*"'),
    ("ARROW", r"->"),
    ("PERCENT", r"%[A-Za-z0-9_$.]+(?::[0-9]+|#[0-9]+)?"),
    ("CARETID", r"\^[A-Za-z0-9_$.]*"),
    (
        "NUMBER",
        r"0[xX][0-9a-fA-F]+"
        r"|-?[0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?"
        r"|-?[0-9]+[eE][+-]?[0-9]+"
        r"|-?[0-9]+",
    ),
    ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("LANGLE", r"<"),
    ("RANGLE", r">"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("LBRACKET", r"\["),
    ("RBRACKET", r"\]"),
    ("COLON", r":"),
    ("COMMA", r","),
    ("EQUALS", r"="),
    ("DOT", r"\."),
    ("HASH", r"#"),
    ("BANG", r"!"),
    ("AT", r"@"),
    ("PLUS", r"\+"),  # `#ttg.padded_shared<[16:+4] ...>`: an interval with a padding
    ("COMMENT", r"//[^\n]*"),
    ("NEWLINE", r"\n"),
    ("SKIP", r"[ \t\r]+"),
]
_MASTER_RE = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in _TOKEN_SPEC))
_IGNORED_KINDS = {"COMMENT", "NEWLINE", "SKIP"}
_OPENERS = {"LPAREN", "LBRACE", "LBRACKET", "LANGLE"}
_CLOSERS = {"RPAREN", "RBRACE", "RBRACKET", "RANGLE"}


def tokenize(text: str) -> list[Token]:
    """Split `text` into tokens, ending with a sentinel `EOF` token."""
    tokens: list[Token] = []
    pos = 0
    n = len(text)
    while pos < n:
        m = _MASTER_RE.match(text, pos)
        if m is None:
            raise ParseError(f"unexpected character {text[pos]!r} at offset {pos}")
        kind = m.lastgroup
        assert kind is not None
        if kind not in _IGNORED_KINDS:
            tokens.append(Token(kind, m.group(), m.start(), m.end()))
        pos = m.end()
    tokens.append(Token("EOF", "", n, n))
    return tokens


def _unescape_string(raw: str) -> str:
    inner = raw[1:-1]
    out: list[str] = []
    i = 0
    mapping = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
    while i < len(inner):
        c = inner[i]
        if c == "\\" and i + 1 < len(inner):
            nxt = inner[i + 1]
            if nxt in mapping:
                out.append(mapping[nxt])
                i += 2
                continue
            if i + 2 < len(inner) and all(
                h in "0123456789abcdefABCDEF" for h in inner[i + 1 : i + 3]
            ):
                out.append(chr(int(inner[i + 1 : i + 3], 16)))  # MLIR's \XX byte escape
                i += 3
                continue
            out.append(nxt)
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _escape_string(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _parse_number_literal(raw: str) -> int | float:
    if raw[:2].lower() == "0x":
        return int(raw, 16)
    if "." in raw or "e" in raw.lower():
        return float(raw)
    return int(raw)


def _find_top_comma(text: str) -> int | None:
    """The index of the first top-level (bracket- and string-depth zero) comma in `text`."""
    depth = 0
    in_str = False
    i = 0
    while i < len(text):
        c = text[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
        elif c in "<{[(":
            depth += 1
        elif c in ">}])":
            depth -= 1
        elif c == "," and depth == 0:
            return i
        i += 1
    return None


def _split_top_all(text: str) -> list[str]:
    """Every top-level comma-separated part of `text`, trimmed."""
    parts: list[str] = []
    depth = 0
    in_str = False
    start = 0
    i = 0
    while i < len(text):
        c = text[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
        elif c in "<{[(":
            depth += 1
        elif c in ">}])":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
        i += 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _elem_width(elem_type: Type) -> int:
    """The storage width in bits of the type a hex literal was printed against.

    `parse_type` fills `width` for every scalar type it builds (8 for the fp8 kinds), so the
    fallback only covers a hand-made `Type` that left it out.
    """
    return int(elem_type.width or 32)


def _parse_scalar_literal(text: str, elem_type: Type) -> object:
    text = text.strip()
    if text in ("true", "false"):
        return text == "true"
    if text[:2].lower() == "0x":
        # MLIR prints a float in hex whenever the shortest decimal would not round-trip, so
        # every NaN and every infinity arrives here: the digits are a bit pattern, not a value.
        if elem_type.kind == "float":
            return FloatBits(int(text, 16), _elem_width(elem_type))
        return int(text, 16)
    if elem_type.kind == "float":
        return float(text)
    return int(text)


def _parse_dense_value_text(text: str, elem_type: Type) -> object:
    text = text.strip()
    if text.startswith("["):
        return [_parse_dense_value_text(p, elem_type) for p in _split_top_all(text[1:-1])]
    return _parse_scalar_literal(text, elem_type)


def _parse_dense_hex_blob(text: str, elem_type: Type, count: int) -> object:
    """`dense<"0x...">`: the raw little-endian bytes of every element, as MLIR writes them.

    A blob the size of one element is a splat, which is how MLIR shortens a large constant.
    """
    digits = text.strip().strip('"')[2:]
    width = max(_elem_width(elem_type), 8)
    nibbles = width // 4
    words = [digits[i : i + nibbles] for i in range(0, len(digits), nibbles)]
    values = [int.from_bytes(bytes.fromhex(w), "little") for w in words]
    out: list[object] = (
        [FloatBits(v, width) for v in values] if elem_type.kind == "float" else list(values)
    )
    if len(out) == 1 and count != 1:
        return out[0]  # a one-element blob is MLIR's shorthand for a splat
    return out


# --------------------------------------------------------------------------- parser

_FLOAT_WIDTHS = {"f16": 16, "bf16": 16, "f32": 32, "f64": 64}
_INT_RE = re.compile(r"[su]?i([0-9]+)$")  # i32, and the si8/ui8 of tensor descriptors
_F8_RE = re.compile(r"f8E[0-9A-Za-z]+$")
_SHAPE_DIM_RE = re.compile(r"^(\d+)x")
_SHAPE_DYN_RE = re.compile(r"^\?x")


class Parser:
    """A recursive-descent parser positioned over a fixed token list."""

    def __init__(self, tokens: list[Token], text: str) -> None:
        self.tokens = tokens
        self.text = text
        self.pos = 0
        self.aliases: dict[str, object] = {}

    # -- low-level token helpers -------------------------------------------------

    def cur(self) -> Token:
        return self.tokens[self.pos]

    def check(self, kind: str) -> bool:
        return self.tokens[self.pos].kind == kind

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return tok

    def expect_kind(self, kind: str) -> Token:
        tok = self.cur()
        if tok.kind != kind:
            raise ParseError(f"expected {kind}, got {tok.kind} {tok.value!r} at {tok.start}")
        return self.advance()

    def _peek(self, offset: int) -> Token:
        idx = min(self.pos + offset, len(self.tokens) - 1)
        return self.tokens[idx]

    def _span_from(self, start_idx: int) -> str:
        end_idx = self.pos - 1
        return self.text[self.tokens[start_idx].start : self.tokens[end_idx].end]

    def _read_dotted_ident(self) -> str:
        parts = [self.expect_kind("IDENT").value]
        while self.check("DOT"):
            self.advance()
            parts.append(self.expect_kind("IDENT").value)
        return ".".join(parts)

    def _read_group_raw(self) -> str:
        """Consume a bracketed group starting at the current opening token; return the
        source text strictly between the brackets, stripped."""
        self.advance()  # the opener
        inner_start_idx = self.pos
        depth = 1
        while depth > 0:
            kind = self.cur().kind
            if kind == "EOF":
                raise ParseError("unbalanced brackets: unexpected end of input")
            if kind in _OPENERS:
                depth += 1
            elif kind in _CLOSERS:
                depth -= 1
            self.advance()
        closer_idx = self.pos - 1
        if inner_start_idx == closer_idx:
            return ""
        return self.text[self.tokens[inner_start_idx].start : self.tokens[closer_idx].start].strip()

    # -- aliases -------------------------------------------------------------------

    def parse_leading_aliases(self) -> None:
        """`#name = ...` definitions at the top of the file, substituted at use sites."""
        while (
            self.check("HASH") and self._peek(1).kind == "IDENT" and self._peek(2).kind == "EQUALS"
        ):
            self.advance()  # HASH
            name = self.expect_kind("IDENT").value
            self.expect_kind("EQUALS")
            self.aliases[name] = self.parse_attr_value()

    # -- types -----------------------------------------------------------------

    def parse_type(self) -> Type:
        tok = self.cur()
        if tok.kind == "BANG":
            return self._parse_bang_type()
        if tok.kind == "IDENT":
            return self._parse_ident_type()
        if tok.kind == "LPAREN":
            return self._parse_function_type()
        raise ParseError(f"unexpected token starting a type: {tok.kind} {tok.value!r}")

    def _parse_bang_type(self) -> Type:
        start_idx = self.pos
        self.advance()  # BANG
        chain = self._read_dotted_ident()
        if not self.check("LANGLE"):
            return Type(kind="other", name=self._span_from(start_idx))
        inner = self._read_group_raw()
        full_raw = self._span_from(start_idx)
        if chain == "tt.ptr":
            return self._make_ptr_type(inner, full_raw)
        if chain == "ttg.memdesc":
            return self._make_shaped_type("memdesc", inner, full_raw)
        if chain == "tt.tensordesc":
            return self._make_shaped_type("tensordesc", inner, full_raw)
        return Type(kind="other", name=full_raw)

    def _make_ptr_type(self, inner: str, full_raw: str) -> Type:
        idx = _find_top_comma(inner)
        elem_text = inner[:idx] if idx is not None else inner
        addrspace = inner[idx + 1 :].strip() if idx is not None else None
        elem = self._parse_type_from_text(elem_text.strip())
        encoding = self._substitute_aliases_in_raw(addrspace) if addrspace else None
        return Type(kind="ptr", elem=elem, width=64, name=full_raw, encoding=encoding)

    def _make_shaped_type(self, kind: str, inner: str, full_raw: str) -> Type:
        idx = _find_top_comma(inner)
        shape_elem_text = inner[:idx] if idx is not None else inner
        raw_encoding = inner[idx + 1 :].strip() if idx is not None else None
        dims, elem = self._parse_shape_elem_text(shape_elem_text.strip())
        encoding = self._substitute_aliases_in_raw(raw_encoding) if raw_encoding else None
        return Type(kind=kind, shape=dims, elem=elem, name=full_raw, encoding=encoding)

    def _substitute_aliases_in_raw(self, text: str) -> str:
        """A type's encoding is captured as raw text, bypassing `parse_attr_value`, so a bare
        `#name` alias reference within it (e.g. `tensor<4xi32, #blocked>`) is substituted here
        instead. Only whole comma-separated parts that are exactly an alias name qualify."""
        parts = _split_top_all(text)
        out = []
        for part in parts:
            m = re.fullmatch(r"#([A-Za-z_][A-Za-z0-9_]*)", part)
            value = self.aliases.get(m.group(1)) if m else None
            out.append(value if isinstance(value, str) else part)
        return ", ".join(out)

    def _parse_ident_type(self) -> Type:
        word = self.cur().value
        if word in ("tensor", "vector"):
            start_idx = self.pos
            self.advance()
            inner = self._read_group_raw()
            return self._make_shaped_type(word, inner, self._span_from(start_idx))
        m = _INT_RE.match(word)
        if m:
            self.advance()
            return Type(kind="int", width=int(m.group(1)), name=word)
        if word == "index":
            self.advance()
            return Type(kind="index", name="index")
        if word in _FLOAT_WIDTHS or _F8_RE.match(word):
            self.advance()
            return Type(kind="float", width=_FLOAT_WIDTHS.get(word, 8), name=word)
        start_idx = self.pos
        self.advance()
        if self.check("LANGLE"):
            self._read_group_raw()
        return Type(kind="other", name=self._span_from(start_idx))

    def _parse_function_type(self) -> Type:
        start_idx = self.pos
        self._read_group_raw()  # the operand-types group; content is not needed
        self.expect_kind("ARROW")
        if self.check("LPAREN"):
            self._read_group_raw()
        else:
            self.parse_type()
        return Type(kind="other", name=self._span_from(start_idx))

    def _parse_type_from_text(self, text: str) -> Type:
        return Parser(tokenize(text), text).parse_type()

    def _parse_shape_elem_text(self, text: str) -> tuple[tuple[int, ...], Type]:
        dims: list[int] = []
        rest = text.strip()
        while True:
            m = _SHAPE_DIM_RE.match(rest)
            if m:
                dims.append(int(m.group(1)))
                rest = rest[m.end() :]
                continue
            m2 = _SHAPE_DYN_RE.match(rest)
            if m2:
                dims.append(-1)
                rest = rest[m2.end() :]
                continue
            break
        elem = self._parse_type_from_text(rest)
        return tuple(dims), elem

    def parse_signature(self) -> tuple[list[Type], list[Type]]:
        operand_types = self._parse_type_list_paren()
        self.expect_kind("ARROW")
        if self.check("LPAREN"):
            result_types = self._parse_type_list_paren()
        else:
            result_types = [self.parse_type()]
        return operand_types, result_types

    def _parse_type_list_paren(self) -> list[Type]:
        self.expect_kind("LPAREN")
        types: list[Type] = []
        while not self.check("RPAREN"):
            types.append(self.parse_type())
            if self.check("COMMA"):
                self.advance()
                continue
            break
        self.expect_kind("RPAREN")
        return types

    # -- attribute values --------------------------------------------------------

    def parse_attr_value(self) -> object:
        tok = self.cur()
        if tok.kind == "STRING":
            self.advance()
            return _unescape_string(tok.value)
        if tok.kind == "AT":
            return self._parse_symbol_ref()
        if tok.kind == "HASH":
            return self._parse_hash_value()
        if tok.kind == "NUMBER":
            return self._parse_number_value()
        if tok.kind == "IDENT":
            return self._parse_ident_value()
        if tok.kind in ("BANG", "LPAREN"):
            return self.parse_type()
        if tok.kind == "LBRACKET":
            return self._parse_list_value()
        if tok.kind == "LBRACE":
            return self._parse_dict_value()
        raise ParseError(f"unexpected token in attribute value: {tok.kind} {tok.value!r}")

    def _parse_symbol_ref(self) -> str:
        self.advance()  # AT
        if self.check("STRING"):
            return "@" + _unescape_string(self.advance().value)
        return "@" + self._read_dotted_ident()

    def _parse_hash_value(self) -> object:
        """`#dialect.name<...>` becomes its raw text; a bare `#name` is an alias reference."""
        start_idx = self.pos
        self.advance()  # HASH
        first = self.expect_kind("IDENT").value
        if not self.check("DOT"):
            if self.check("LANGLE"):  # `#nvvm<shfl_kind bfly>`: an enum of a dialect, raw
                self._read_group_raw()
                return self._span_from(start_idx)
            return self.aliases.get(first, f"#{first}")
        while self.check("DOT"):
            self.advance()
            self.expect_kind("IDENT")
        if self.check("LANGLE"):
            self._read_group_raw()
        return self._span_from(start_idx)

    def _parse_number_value(self) -> object:
        tok = self.advance()
        value = _parse_number_literal(tok.value)
        annotation: Type | None = None
        if self.check("COLON"):
            self.advance()
            annotation = self.parse_type()  # only a hex float needs it; otherwise discarded
        if annotation is not None and annotation.kind == "float" and tok.value[:2].lower() == "0x":
            return FloatBits(int(tok.value, 16), _elem_width(annotation))
        return value

    def _parse_ident_value(self) -> object:
        word = self.cur().value
        if word in ("true", "false"):
            self.advance()
            return word == "true"
        if word == "dense":
            return self.parse_dense_attr()
        if word == "array":
            return self.parse_array_attr()
        return self.parse_type()

    def _parse_list_value(self) -> list[object]:
        self.expect_kind("LBRACKET")
        items: list[object] = []
        while not self.check("RBRACKET"):
            items.append(self.parse_attr_value())
            if self.check("COMMA"):
                self.advance()
                continue
            break
        self.expect_kind("RBRACKET")
        return items

    def _parse_dict_value(self) -> dict[str, object]:
        self.expect_kind("LBRACE")
        d = self.parse_attr_dict_body()
        self.expect_kind("RBRACE")
        return d

    def parse_array_attr(self) -> list[object]:
        self.expect_kind("IDENT")  # 'array'
        self.expect_kind("LANGLE")
        elem_type = self.parse_type()
        values: list[int | float] = []
        if self.check("COLON"):
            self.advance()
            while not self.check("RANGLE"):
                values.append(_parse_number_literal(self.expect_kind("NUMBER").value))
                if self.check("COMMA"):
                    self.advance()
                    continue
                break
        self.expect_kind("RANGLE")
        if elem_type.kind == "float":
            return [float(v) for v in values]
        return [int(v) for v in values]

    def parse_dense_attr(self) -> DenseAttr:
        self.expect_kind("IDENT")  # 'dense'
        raw_value = self._read_group_raw()
        self.expect_kind("COLON")
        tensor_type = self.parse_type()
        elem_type = tensor_type.elem if tensor_type.elem is not None else tensor_type
        shape = tensor_type.shape if tensor_type.shape is not None else ()
        count = 1
        for dim in shape:
            count *= dim
        if raw_value.lstrip().startswith('"'):
            value = _parse_dense_hex_blob(raw_value, elem_type, count)
        else:
            value = _parse_dense_value_text(raw_value, elem_type)
        return DenseAttr(value=value, elem_type=elem_type, shape=shape)

    def _parse_dict_key(self) -> str:
        if self.check("STRING"):
            return _unescape_string(self.advance().value)
        return self._read_dotted_ident()

    def parse_attr_dict_body(self) -> dict[str, object]:
        result: dict[str, object] = {}
        while not self.check("RBRACE"):
            key = self._parse_dict_key()
            if self.check("EQUALS"):
                self.advance()
                result[key] = self.parse_attr_value()
            else:
                result[key] = True
            if self.check("COMMA"):
                self.advance()
                continue
            break
        return result

    # -- ops, blocks, regions ---------------------------------------------------

    def parse_op(self) -> Op:
        start_idx = self.pos
        results = self._parse_result_list() if self.check("PERCENT") else []
        if results:
            self.expect_kind("EQUALS")
        name = _unescape_string(self.expect_kind("STRING").value)
        operands = self._parse_operand_list()
        successors = self._parse_successor_list() if self.check("LBRACKET") else []
        attrs = self._parse_inherent_attrs() if self.check("LANGLE") else {}
        regions = self._parse_region_list() if self.check("LPAREN") else []
        if self.check("LBRACE"):
            attrs.update(self._parse_discardable_attrs())
        operand_types: list[Type] = []
        result_types: list[Type] = []
        if self.check("COLON"):
            self.advance()
            operand_types, result_types = self.parse_signature()
        loc = self._parse_loc()
        return Op(
            name=name,
            results=results,
            result_types=result_types,
            operands=operands,
            operand_types=operand_types,
            attrs=attrs,
            regions=regions,
            successors=successors,
            loc=loc,
            text=self._span_from(start_idx),
        )

    def _parse_result_list(self) -> list[str]:
        names = [self.expect_kind("PERCENT").value]
        while self.check("COMMA"):
            self.advance()
            names.append(self.expect_kind("PERCENT").value)
        results: list[str] = []
        for n in names:
            if ":" in n:
                base, count = n.split(":")
                results.extend(f"{base}#{i}" for i in range(int(count)))
            else:
                results.append(n)
        return results

    def _parse_operand_list(self) -> list[str]:
        self.expect_kind("LPAREN")
        operands: list[str] = []
        while not self.check("RPAREN"):
            operands.append(self.expect_kind("PERCENT").value)
            if self.check("COMMA"):
                self.advance()
                continue
            break
        self.expect_kind("RPAREN")
        return operands

    def _parse_successor_list(self) -> list[str]:
        """`[^bb1, ^bb2]`: plain target labels. Destination values travel in the op's own
        operand list (sliced by the op's semantics), not in the successor list itself."""
        self.expect_kind("LBRACKET")
        labels: list[str] = []
        while not self.check("RBRACKET"):
            labels.append(self.expect_kind("CARETID").value)
            if self.check("LPAREN"):
                self._read_group_raw()  # tolerate (and discard) a typed operand list
            if self.check("COMMA"):
                self.advance()
                continue
            break
        self.expect_kind("RBRACKET")
        return labels

    def _parse_inherent_attrs(self) -> dict[str, object]:
        self.advance()  # LANGLE
        self.expect_kind("LBRACE")
        attrs = self.parse_attr_dict_body()
        self.expect_kind("RBRACE")
        self.expect_kind("RANGLE")
        return attrs

    def _parse_discardable_attrs(self) -> dict[str, object]:
        self.advance()  # LBRACE
        attrs = self.parse_attr_dict_body()
        self.expect_kind("RBRACE")
        return attrs

    def _parse_region_list(self) -> list[Region]:
        self.advance()  # LPAREN
        regions = [self.parse_region()]
        while self.check("COMMA"):
            self.advance()
            regions.append(self.parse_region())
        self.expect_kind("RPAREN")
        return regions

    def _parse_loc(self) -> str | None:
        if self.check("IDENT") and self.cur().value == "loc" and self._peek(1).kind == "LPAREN":
            start_idx = self.pos
            self.advance()
            self._read_group_raw()
            return self._span_from(start_idx)
        return None

    def parse_region(self) -> Region:
        self.expect_kind("LBRACE")
        blocks = []
        while not self.check("RBRACE"):
            blocks.append(self._parse_block())
        self.expect_kind("RBRACE")
        return Region(blocks=blocks)

    def _parse_block(self) -> Block:
        label = None
        args: list[tuple[str, Type]] = []
        if self.check("CARETID"):
            label = self.advance().value
            if self.check("LPAREN"):
                args = self._parse_block_args()
            self.expect_kind("COLON")
        ops = []
        while not (self.check("RBRACE") or self.check("CARETID")):
            ops.append(self.parse_op())
        return Block(label=label, args=args, ops=ops)

    def _parse_block_args(self) -> list[tuple[str, Type]]:
        self.expect_kind("LPAREN")
        args: list[tuple[str, Type]] = []
        while not self.check("RPAREN"):
            name = self.expect_kind("PERCENT").value
            self.expect_kind("COLON")
            args.append((name, self.parse_type()))
            self._parse_loc()  # Triton's printer attaches a location to every block argument
            if self.check("COMMA"):
                self.advance()
                continue
            break
        self.expect_kind("RPAREN")
        return args


def parse(text: str) -> Module:
    """Parse MLIR generic text into a `Module`."""
    tokens = tokenize(text)
    parser = Parser(tokens, text)
    parser.parse_leading_aliases()
    module_op = parser.parse_op()
    if not parser.check("EOF"):
        tok = parser.cur()
        raise ParseError(f"trailing input after the top-level op: {tok.kind} at {tok.start}")
    if module_op.name != "builtin.module":
        raise ParseError(f"expected a top-level builtin.module op, got {module_op.name!r}")
    body_ops = module_op.regions[0].blocks[0].ops if module_op.regions else []
    funcs: dict[str, Op] = {}

    def collect(ops: list[Op]) -> None:  # functions of this module and of any nested module
        for op in ops:
            if op.name.endswith(".func") and "sym_name" in op.attrs:
                funcs.setdefault(str(op.attrs["sym_name"]), op)
            elif op.name.endswith(".module") and op.regions and op.regions[0].blocks:
                collect(op.regions[0].blocks[0].ops)  # builtin.module, gpu.module

    collect(body_ops)
    return Module(funcs=funcs, ops=body_ops, attrs=dict(module_op.attrs))


# --------------------------------------------------------------------------- unparser


def unparse(module: Module) -> str:
    """Print `module` back to generic form. Formatting is not meant to match `triton-opt`
    byte for byte, only to be valid generic text that reparses to an equal structure."""
    body = "\n".join(f"  {_unparse_op(op, '  ')}" for op in module.ops)
    return '"builtin.module"() ({\n' + body + "\n}) : () -> ()"


def _unparse_op(op: Op, indent: str) -> str:
    prefix = f"{_unparse_results_lhs(op.results)} = " if op.results else ""
    line = f'{prefix}"{op.name}"({", ".join(op.operands)})'
    if op.successors:
        line += "[" + ", ".join(op.successors) + "]"
    if op.attrs:
        line += f" <{{{_unparse_dict_body(op.attrs)}}}>"
    if op.regions:
        line += " (" + ", ".join(_unparse_region(r, indent + "  ") for r in op.regions) + ")"
    line += " : " + _unparse_signature(op)
    if op.loc:
        line += " " + op.loc
    return line


def _unparse_signature(op: Op) -> str:
    operand_part = "(" + ", ".join(_unparse_type(t) for t in op.operand_types) + ")"
    result_part = "(" + ", ".join(_unparse_type(t) for t in op.result_types) + ")"
    return f"{operand_part} -> {result_part}"


def _unparse_results_lhs(results: list[str]) -> str:
    if len(results) == 1:
        return results[0]
    base = results[0].split("#")[0]
    return f"{base}:{len(results)}"


def _unparse_region(r: Region, indent: str) -> str:
    lines = [_unparse_block(b, indent + "  ") for b in r.blocks]
    return "{\n" + "\n".join(lines) + "\n" + indent + "}"


def _unparse_block(b: Block, indent: str) -> str:
    lines: list[str] = []
    if b.label is not None:
        args_str = ", ".join(f"{name}: {_unparse_type(ty)}" for name, ty in b.args)
        lines.append(f"{indent}{b.label}({args_str}):")
    lines.extend(f"{indent}{_unparse_op(op, indent)}" for op in b.ops)
    return "\n".join(lines)


_BARE_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _unparse_key(k: str) -> str:
    if _BARE_KEY_RE.match(k):
        return k
    return '"' + _escape_string(k) + '"'


def _unparse_dict_body(d: dict[str, object]) -> str:
    return ", ".join(f"{_unparse_key(k)} = {_unparse_attr_value(v)}" for k, v in d.items())


def _format_float(v: float) -> str:
    text = repr(float(v))
    if "." not in text and "e" not in text and "inf" not in text and "nan" not in text:
        text += ".0"
    return text


# A hex float literal is reprinted against a float type of the same width; which 16-bit kind
# it was does not survive, and does not have to: only the width is read back.
_FLOAT_NAME_BY_WIDTH = {8: "f8E4M3FN", 16: "f16", 32: "f32", 64: "f64"}


def _format_float_bits(v: FloatBits) -> str:
    return f"0x{v.bits:0{max(v.width, 8) // 4}X}"


def _unparse_attr_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, FloatBits):
        return f"{_format_float_bits(v)} : {_FLOAT_NAME_BY_WIDTH.get(v.width, 'f32')}"
    if isinstance(v, int):
        return f"{v} : i64"
    if isinstance(v, float):
        return f"{_format_float(v)} : f64"
    if isinstance(v, str):
        if v.startswith("#") or v.startswith("@"):
            return v
        return '"' + _escape_string(v) + '"'
    if isinstance(v, Type):
        return _unparse_type(v)
    if isinstance(v, DenseAttr):
        return _unparse_dense(v)
    if isinstance(v, list):
        return "[" + ", ".join(_unparse_attr_value(e) for e in v) + "]"
    if isinstance(v, dict):
        return "{" + _unparse_dict_body(v) + "}"
    raise TypeError(f"cannot unparse attribute value of type {type(v)!r}")


def _format_dense_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, FloatBits):
        return _format_float_bits(v)
    if isinstance(v, list):
        return "[" + ", ".join(_format_dense_value(e) for e in v) + "]"
    if isinstance(v, float):
        return _format_float(v)
    return str(int(v))  # type: ignore[call-overload]


def _unparse_dense(d: DenseAttr) -> str:
    elem_type = d.elem_type if d.elem_type is not None else Type(kind="int", width=32, name="i32")
    shape = d.shape if d.shape is not None else ()
    type_str = _unparse_type(Type(kind="tensor", shape=shape, elem=elem_type))
    if isinstance(d.value, str) and d.value.lower().startswith("0x"):
        inner = d.value
    else:
        inner = _format_dense_value(d.value)
    return f"dense<{inner}> : {type_str}"


_SHAPED_PREFIX = {"tensor": "tensor", "memdesc": "!ttg.memdesc", "tensordesc": "!tt.tensordesc"}


def _unparse_type(t: Type) -> str:
    if t.kind in ("int", "float"):
        return t.name
    if t.kind == "index":
        return "index"
    if t.kind == "ptr":
        inner = _unparse_type(t.elem) if t.elem is not None else "i8"
        return f"!tt.ptr<{inner}, {t.encoding}>" if t.encoding else f"!tt.ptr<{inner}>"
    if t.kind in _SHAPED_PREFIX:
        return _unparse_shaped_type(t)
    return t.name


def _unparse_shaped_type(t: Type) -> str:
    shape = t.shape or ()
    shape_str = "x".join(str(d) if d >= 0 else "?" for d in shape)
    elem_str = _unparse_type(t.elem) if t.elem is not None else "i8"
    body = f"{shape_str}x{elem_str}" if shape_str else elem_str
    prefix = _SHAPED_PREFIX[t.kind]
    return f"{prefix}<{body}, {t.encoding}>" if t.encoding else f"{prefix}<{body}>"


# --------------------------------------------------------------------------- to_generic


def to_generic(text: str, triton_opt: str | None = None) -> str:
    """Convert pretty-printed MLIR to generic form via `triton_opt`, or pass generic text
    through unchanged. Raises `NotGeneric` if `text` is not already generic and no
    `triton_opt` binary was given to convert it."""
    if triton_opt:
        return _run_triton_opt(text, triton_opt)
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if '"builtin.module"()' in stripped:
            return text
        raise NotGeneric("input is not in MLIR generic form and no triton_opt was given")
    raise NotGeneric("empty input")


def _run_triton_opt(text: str, triton_opt: str, extra: tuple[str, ...] = ()) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        result = subprocess.run(
            [triton_opt, *extra, "--mlir-print-op-generic", "--mlir-print-local-scope", path],
            capture_output=True,
            text=True,
        )
    finally:
        Path(path).unlink(missing_ok=True)
    if result.returncode != 0:
        # the first diagnostic names the reason (a printer that does not round-trip, a
        # verifier constraint the printed op violates); the caller classifies on it
        lines = [ln for ln in result.stderr.splitlines() if ln.strip()]
        first = lines[0].replace(path, "<stage>") if lines else f"exit {result.returncode}"
        raise ParseError(f"triton-opt cannot re-read the module: {first}")
    return result.stdout


from ttsem.defuse import Def, ancestors, definer, defs, parents, users  # noqa: E402,F401

_GENERIC_FLAGS = (b"ttsem", b"--mlir-print-op-generic", b"--mlir-print-local-scope")
_REGISTER_ASM_PRINTER_CL_OPTIONS = "_ZN4mlir27registerAsmPrinterCLOptionsEv"
_generic_printing_enabled = [False]


def enable_generic_printing() -> None:
    """Switch the Triton wheel's MLIR printer to the generic form, once per process.

    The wheel ships no `triton-opt`, but its `libtriton.so` is MLIR and exports
    `mlir::registerAsmPrinterCLOptions()` and `LLVMParseCommandLineOptions`; registering the
    printer options and parsing `--mlir-print-op-generic --mlir-print-local-scope` into them
    makes every later `str(module)`, `compiled.asm[...]` and `MLIR_ENABLE_DUMP` line generic,
    with locations. Process-wide and irreversible, so it is for subprocesses that exist to
    produce IR (the dump runner), never for a pytest process whose tests read the pretty form.
    """
    if _generic_printing_enabled[0]:
        return
    import ctypes
    import pathlib as _pathlib

    import triton

    root = _pathlib.Path(triton.__file__).parent / "_C"
    candidates = sorted(root.glob("libtriton*.so"))
    if not candidates:
        raise RuntimeError(f"no libtriton*.so under {root}")
    lib = ctypes.CDLL(str(candidates[0]), mode=ctypes.RTLD_GLOBAL)
    try:
        register = getattr(lib, _REGISTER_ASM_PRINTER_CL_OPTIONS)
    except AttributeError as exc:
        raise RuntimeError(
            f"{candidates[0]} does not export mlir::registerAsmPrinterCLOptions(); use triton-opt"
        ) from exc
    register.restype = None
    register()
    argv = (ctypes.c_char_p * len(_GENERIC_FLAGS))(*_GENERIC_FLAGS)
    lib.LLVMParseCommandLineOptions.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_char_p,
    ]
    lib.LLVMParseCommandLineOptions.restype = None
    lib.LLVMParseCommandLineOptions(len(_GENERIC_FLAGS), argv, b"ttsem")
    _generic_printing_enabled[0] = True
