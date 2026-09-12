"""The parser on multi-block regions, non-tt functions, and module attributes."""

from __future__ import annotations

from ttsem import mlir
from ttsem.defuse import definer, defs

CFG = """
"builtin.module"() ({
  "llvm.func"() <{sym_name = "k", function_type = !llvm.func<void (i32)>}> ({
  ^bb0(%arg0: i32):
    %0 = "llvm.mlir.constant"() <{value = 1 : i32}> : () -> i32
    "llvm.br"(%0)[^bb2] : (i32) -> ()
  ^bb1:
    %1 = "llvm.add"(%2, %0) : (i32, i32) -> i32
    "llvm.return"() : () -> ()
  ^bb2(%2: i32):
    "llvm.br"()[^bb1] : () -> ()
  }) : () -> ()
  "tt.func"() <{sym_name = "t", function_type = () -> ()}> ({
    "tt.return"() : () -> ()
  }) : () -> ()
}) {"ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} : () -> ()
"""


def test_uses_across_blocks_of_one_region_resolve() -> None:
    module = mlir.parse(CFG)
    table = defs(module)
    fn = module.funcs["k"]
    blocks = fn.regions[0].blocks
    add = blocks[1].ops[0]
    assert add.name == "llvm.add"
    d2 = definer(table, add, "%2")
    d0 = definer(table, add, "%0")
    assert d2 is not None and d2.block is blocks[2] and d2.op is None
    assert d0 is not None and d0.op is blocks[0].ops[0]


def test_every_func_kind_is_a_function_and_module_attrs_are_kept() -> None:
    module = mlir.parse(CFG)
    assert set(module.funcs) == {"k", "t"}
    assert module.funcs["k"].name == "llvm.func"
    assert "ttg.num-warps" in module.attrs and "ttg.target" in module.attrs
    assert int(str(module.attrs["ttg.num-warps"]).split(":")[0]) == 8


def test_padded_shared_intervals_lex() -> None:
    padded = "#ttg.padded_shared<[16:+4] {order = [1, 0], shape = [4, 16]}>"
    text = f"""
"builtin.module"() ({{
  "tt.func"() <{{sym_name = "k", function_type = () -> ()}}> ({{
    %0 = "ttg.local_alloc"() : () -> !ttg.memdesc<4x16xf16, {padded}, #ttg.shared_memory, mutable>
    "tt.return"() : () -> ()
  }}) : () -> ()
}}) : () -> ()
"""
    module = mlir.parse(text)
    alloc = module.funcs["k"].regions[0].blocks[0].ops[0]
    assert "[16:+4]" in (alloc.result_types[0].encoding or "")


def test_bare_dialect_enums_and_gpu_module_functions() -> None:
    text = """
"builtin.module"() ({
  "gpu.module"() <{sym_name = "kernels"}> ({
    "llvm.func"() <{sym_name = "k", function_type = !llvm.func<void (i32)>}> ({
    ^bb0(%arg0: i32):
      %0 = "llvm.mlir.constant"() <{value = -1 : i32}> : () -> i32
      %1 = "nvvm.shfl.sync"(%0, %arg0, %0) <{kind = #nvvm<shfl_kind bfly>}> : (i32, i32, i32) -> i32
      %2 = "gpu.all_reduce"(%1) <{op = #gpu<all_reduce_op xor>, uniform = false}> : (i32) -> i32
      "llvm.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
}) : () -> ()
"""
    module = mlir.parse(text)
    assert list(module.funcs) == ["k"]
    ops = module.funcs["k"].regions[0].blocks[0].ops
    assert str(ops[1].attrs["kind"]) == "#nvvm<shfl_kind bfly>"
    assert str(ops[2].attrs["op"]) == "#gpu<all_reduce_op xor>"


def test_vector_constants_and_hex_escapes() -> None:
    text = r"""
"builtin.module"() ({
  "llvm.func"() <{sym_name = "k", function_type = !llvm.func<void ()>}> ({
    %0 = "llvm.mlir.constant"() <{value = dense<1.000000e+00> : vector<4xf32>}> : ()
      -> vector<4xf32>
    %1 = "llvm.mlir.constant"() <{value = dense<[1, 2]> : vector<2xi32>}> : () -> vector<2xi32>
    %2 = "llvm.inline_asm"() <{asm_string = "mov.b32 $0, 0;\0A\09ld.global.b32 $0, [$1];",
      constraints = "=r,l"}> : () -> i32
    "llvm.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
"""
    module = mlir.parse(text)
    ops = module.funcs["k"].regions[0].blocks[0].ops
    c0 = ops[0].attrs["value"]
    assert c0.elem_type.kind == "float" and c0.elem_type.name == "f32" and tuple(c0.shape) == (4,)
    assert float(c0.value if not hasattr(c0.value, "__len__") else c0.value[0]) == 1.0
    c1 = ops[1].attrs["value"]
    assert c1.elem_type.kind == "int" and list(c1.value) == [1, 2]
    assert ops[2].attrs["asm_string"] == "mov.b32 $0, 0;\n\tld.global.b32 $0, [$1];"
    assert ops[0].result_types[0].kind == "vector" and ops[0].result_types[0].shape == (4,)
