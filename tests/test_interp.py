"""Control flow, regions and whole-function execution."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import block, func, interp, op, ptr, region, tensor, ty
from ttsem.ir_types import DenseAttr, Module, Op
from ttsem.memory import Memory
from ttsem.values import Unsupported

I32 = ty("i32")
I1 = ty("i1")
F32 = ty("f32")
T4 = tensor((4,), "i32")
T4B = tensor((4,), "i1")


def const(name: str, value: int, t=I32) -> op:
    attr = DenseAttr(value, t.elem or t, t.shape) if t.kind == "tensor" else value
    return op("arith.constant", [], t, attrs={"value": attr}, results=[name], operands=[])


def add(out: str, a: str, b: str, t=I32) -> op:
    return op("arith.addi", [t, t], t, results=[out], operands=[a, b])


def run(module: Module, name: str, args: list[object] | None = None, **kw) -> list[object]:
    machine = interp(module, **kw)
    return machine.run(name, list(args or []))


# --------------------------------------------------------------------------- scf.for


def test_scf_for_with_iter_args() -> None:
    body = region(
        block(
            [("%iv", I32), ("%acc", I32)],
            [
                add("%next", "%acc", "%iv"),
                op("scf.yield", [I32], [], operands=["%next"], results=[]),
            ],
        )
    )
    loop = op(
        "scf.for",
        [I32, I32, I32, I32],
        I32,
        regions=[body],
        results=["%sum"],
        operands=["%lo", "%hi", "%st", "%zero"],
    )
    module = func(
        "k",
        [],
        [
            const("%lo", 0),
            const("%hi", 5),
            const("%st", 1),
            const("%zero", 0),
            loop,
            op("tt.return", [I32], [], operands=["%sum"], results=[]),
        ],
    )
    assert run(module, "k")[0] == 10


def test_scf_for_with_two_iter_args_binds_both_results() -> None:
    body = region(
        block(
            [("%iv", I32), ("%a", I32), ("%b", I32)],
            [
                add("%na", "%a", "%iv"),
                add("%nb", "%b", "%a"),
                op("scf.yield", [I32, I32], [], operands=["%na", "%nb"], results=[]),
            ],
        )
    )
    loop = op(
        "scf.for",
        [I32] * 5,
        [I32, I32],
        regions=[body],
        results=["%p"],
        operands=["%lo", "%hi", "%st", "%zero", "%zero"],
    )
    module = func(
        "k",
        [],
        [
            const("%lo", 0),
            const("%hi", 4),
            const("%st", 1),
            const("%zero", 0),
            loop,
            op("tt.return", [I32, I32], [], operands=["%p#0", "%p#1"], results=[]),
        ],
    )
    assert [int(np.asarray(v)) for v in run(module, "k")] == [6, 4]


def test_scf_for_with_a_zero_step_is_poison() -> None:
    from ttsem.values import Poison

    body = region(block([("%iv", I32)], [op("scf.yield", [], [], operands=[], results=[])]))
    loop = op("scf.for", [I32] * 3, [], regions=[body], results=[], operands=["%z", "%h", "%z"])
    module = func("k", [], [const("%z", 0), const("%h", 4), loop])
    with pytest.raises(Poison):
        run(module, "k")


# --------------------------------------------------------------------------- scf.if, while


def _if_module(cond_value: int) -> Module:
    then = region(block([], [op("scf.yield", [I32], [], operands=["%one"], results=[])]))
    other = region(block([], [op("scf.yield", [I32], [], operands=["%two"], results=[])]))
    branch = op("scf.if", [I1], I32, regions=[then, other], results=["%v"], operands=["%c"])
    return func(
        "k",
        [],
        [
            const("%one", 1),
            const("%two", 2),
            op(
                "arith.constant",
                [],
                I1,
                attrs={"value": bool(cond_value)},
                results=["%c"],
                operands=[],
            ),
            branch,
            op("tt.return", [I32], [], operands=["%v"], results=[]),
        ],
    )


@pytest.mark.parametrize(("cond", "expect"), [(1, 1), (0, 2)])
def test_scf_if_with_results(cond: int, expect: int) -> None:
    assert run(_if_module(cond), "k")[0] == expect


def test_scf_while() -> None:
    before = region(
        block(
            [("%x", I32)],
            [
                op(
                    "arith.cmpi",
                    [I32, I32],
                    I1,
                    attrs={"predicate": 2},
                    results=["%c"],
                    operands=["%x", "%limit"],
                ),
                op("scf.condition", [I1, I32], [], operands=["%c", "%x"], results=[]),
            ],
        )
    )
    after = region(
        block(
            [("%y", I32)],
            [add("%n", "%y", "%step"), op("scf.yield", [I32], [], operands=["%n"], results=[])],
        )
    )
    loop = op(
        "scf.while", [I32], I32, regions=[before, after], results=["%out"], operands=["%zero"]
    )
    module = func(
        "k",
        [],
        [
            const("%zero", 0),
            const("%step", 3),
            const("%limit", 10),
            loop,
            op("tt.return", [I32], [], operands=["%out"], results=[]),
        ],
    )
    assert run(module, "k")[0] == 12


def test_scf_index_switch() -> None:
    def arm(name: str, value: int) -> region:
        return region(
            block(
                [],
                [
                    const(name, value),
                    op("scf.yield", [I32], [], operands=[name], results=[]),
                ],
            )
        )

    switch = op(
        "scf.index_switch",
        [I32],
        I32,
        attrs={"cases": [0, 7]},
        regions=[arm("%d", 99), arm("%a", 10), arm("%b", 20)],
        results=["%v"],
        operands=["%key"],
    )
    module = func(
        "k",
        [],
        [const("%key", 7), switch, op("tt.return", [I32], [], operands=["%v"], results=[])],
    )
    assert run(module, "k")[0] == 20


def test_scf_index_switch_falls_back_to_the_default() -> None:
    default = region(
        block([], [const("%d", 99), op("scf.yield", [I32], [], operands=["%d"], results=[])])
    )
    case = region(
        block([], [const("%a", 10), op("scf.yield", [I32], [], operands=["%a"], results=[])])
    )
    switch = op(
        "scf.index_switch",
        [I32],
        I32,
        attrs={"cases": [0]},
        regions=[default, case],
        results=["%v"],
        operands=["%key"],
    )
    module = func(
        "k", [], [const("%key", 5), switch, op("tt.return", [I32], [], operands=["%v"], results=[])]
    )
    assert run(module, "k")[0] == 99


# --------------------------------------------------------------------------- cf


def test_cf_br() -> None:
    entry = block(
        [],
        [const("%v", 5), op("cf.br", [I32], [], successors=["^bb1"], operands=["%v"], results=[])],
    )
    target = block(
        [("%p", I32)], [op("tt.return", [I32], [], operands=["%p"], results=[])], label="^bb1"
    )
    body = region(entry, target)
    entry_op = op("tt.func", regions=[body])
    assert interp(Module(funcs={"k": entry_op})).run("k", [])[0] == 5


def _cond_br_module(taken: bool) -> Module:
    entry = block(
        [],
        [
            const("%a", 1),
            const("%b", 2),
            op("arith.constant", [], I1, attrs={"value": taken}, results=["%c"], operands=[]),
            op(
                "cf.cond_br",
                [I1, I32, I32],
                [],
                successors=["^bb1", "^bb2"],
                attrs={"operandSegmentSizes": [1, 1, 1]},
                operands=["%c", "%a", "%b"],
                results=[],
            ),
        ],
    )
    yes = block(
        [("%x", I32)], [op("tt.return", [I32], [], operands=["%x"], results=[])], label="^bb1"
    )
    no = block(
        [("%y", I32)], [op("tt.return", [I32], [], operands=["%y"], results=[])], label="^bb2"
    )
    entry_op = op("tt.func", regions=[region(entry, yes, no)])
    return Module(funcs={"k": entry_op})


@pytest.mark.parametrize(("taken", "expect"), [(True, 1), (False, 2)])
def test_cf_cond_br(taken: bool, expect: int) -> None:
    assert interp(_cond_br_module(taken)).run("k", [])[0] == expect


def test_a_branch_to_a_missing_block_is_an_error() -> None:
    entry = block([], [op("cf.br", [], [], successors=["^nope"], operands=[], results=[])])
    entry_op = op("tt.func", regions=[region(entry)])
    with pytest.raises(KeyError):
        interp(Module(funcs={"k": entry_op})).run("k", [])


# --------------------------------------------------------------------------- nested regions


def _combine(name: str, elem=I32) -> region:
    return region(
        block(
            [("%l", elem), ("%r", elem)],
            [
                op(
                    "arith.addi" if elem is I32 else "arith.addf",
                    [elem, elem],
                    elem,
                    results=["%s"],
                    operands=["%l", "%r"],
                ),
                op(f"{name}.return", [elem], [], operands=["%s"], results=[]),
            ],
        )
    )


def test_tt_reduce_runs_its_combine_region() -> None:
    src = np.arange(12, dtype=np.int32).reshape(3, 4)
    target = op(
        "tt.reduce",
        [tensor((3, 4), "i32")],
        tensor((3,), "i32"),
        attrs={"axis": 1},
        regions=[_combine("tt.reduce")],
    )
    machine = interp()
    machine.scopes = [{"%a0": src}]
    machine.eval_op(target)
    assert np.array_equal(machine.value("%r0"), src.sum(axis=1, dtype=np.int32))


def test_tt_reduce_over_floats() -> None:
    src = np.arange(6, dtype=np.float32).reshape(2, 3)
    target = op(
        "tt.reduce",
        [tensor((2, 3), "f32")],
        tensor((3,), "f32"),
        attrs={"axis": 0},
        regions=[_combine("tt.reduce", F32)],
    )
    machine = interp()
    machine.scopes = [{"%a0": src}]
    machine.eval_op(target)
    assert np.array_equal(machine.value("%r0"), src.sum(axis=0, dtype=np.float32))


def test_tt_reduce_with_two_operands() -> None:
    body = region(
        block(
            [("%la", I32), ("%lb", I32), ("%ra", I32), ("%rb", I32)],
            [
                add("%sa", "%la", "%ra"),
                op("arith.muli", [I32, I32], I32, results=["%sb"], operands=["%lb", "%rb"]),
                op("tt.reduce.return", [I32, I32], [], operands=["%sa", "%sb"], results=[]),
            ],
        )
    )
    a = np.arange(1, 7, dtype=np.int32).reshape(2, 3)
    t = tensor((2, 3), "i32")
    target = op("tt.reduce", [t, t], [tensor((3,), "i32")] * 2, attrs={"axis": 0}, regions=[body])
    machine = interp()
    machine.scopes = [{"%a0": a, "%a1": a}]
    machine.eval_op(target)
    assert np.array_equal(machine.value("%r0"), a.sum(0))
    assert np.array_equal(machine.value("%r1"), a.prod(0))


@pytest.mark.parametrize("reverse", [False, True])
def test_tt_scan_runs_its_combine_region(reverse: bool) -> None:
    src = np.arange(12, dtype=np.int32).reshape(3, 4)
    t = tensor((3, 4), "i32")
    target = op(
        "tt.scan",
        [t],
        t,
        attrs={"axis": 1, "reverse": reverse},
        regions=[_combine("tt.scan")],
    )
    machine = interp()
    machine.scopes = [{"%a0": src}]
    machine.eval_op(target)
    want = np.flip(np.cumsum(np.flip(src, 1), axis=1), 1) if reverse else np.cumsum(src, axis=1)
    assert np.array_equal(machine.value("%r0"), want.astype(np.int32))


def test_a_region_scope_does_not_leak() -> None:
    src = np.arange(4, dtype=np.int32).reshape(1, 4)
    target = op(
        "tt.reduce",
        [tensor((1, 4), "i32")],
        tensor((1,), "i32"),
        attrs={"axis": 1},
        regions=[_combine("tt.reduce")],
    )
    machine = interp()
    machine.scopes = [{"%a0": src}]
    machine.eval_op(target)
    with pytest.raises(KeyError):
        machine.value("%l")


# --------------------------------------------------------------------------- coverage, grid


def test_an_unknown_op_is_collected_and_raised() -> None:
    module = func("k", [], [op("tt.made_up", [], [])])
    machine = interp(module)
    with pytest.raises(Unsupported):
        machine.run("k", [])
    assert machine.unsupported == {"tt.made_up"}


def test_running_a_missing_function_names_what_is_there() -> None:
    with pytest.raises(KeyError):
        interp(func("k", [], [])).run("other", [])


def test_run_grid_visits_every_program_id() -> None:
    memory = Memory()
    seen = np.zeros(6, np.int32)
    memory.register(4096, seen)
    module = func(
        "k",
        [],
        [
            op("tt.get_program_id", [], I32, attrs={"axis": 0}, results=["%x"], operands=[]),
            op("tt.get_program_id", [], I32, attrs={"axis": 1}, results=["%y"], operands=[]),
            const("%three", 3),
            op("arith.muli", [I32, I32], I32, results=["%yy"], operands=["%y", "%three"]),
            add("%idx", "%x", "%yy"),
            op(
                "arith.constant",
                [],
                ptr("i32"),
                attrs={"value": 4096},
                results=["%base"],
                operands=[],
            ),
            op(
                "tt.addptr",
                [ptr("i32"), I32],
                ptr("i32"),
                results=["%p"],
                operands=["%base", "%idx"],
            ),
            const("%one", 1),
            op("tt.store", [ptr("i32"), I32], [], operands=["%p", "%one"], results=[]),
        ],
    )
    interp(module, memory, num_programs=(3, 2, 1)).run_grid("k", [])
    assert np.array_equal(seen, np.ones(6, np.int32))


# --------------------------------------------------------------------------- a whole kernel


def test_a_masked_copy_kernel_matches_numpy() -> None:
    """out[i] = in[i] + 1 for i < 3, in a 4-lane tile; the tail lane is left alone."""
    memory = Memory()
    src = np.arange(4, dtype=np.int32)
    dst = np.full(4, -1, np.int32)
    memory.register(0x1000, src)
    memory.register(0x2000, dst)
    pt = tensor((4,), ptr("i32"))
    body = [
        op("tt.make_range", [], T4, attrs={"start": 0, "end": 4}, results=["%r"], operands=[]),
        op("arith.constant", [], ptr("i32"), attrs={"value": 0x1000}, results=["%sb"], operands=[]),
        op("arith.constant", [], ptr("i32"), attrs={"value": 0x2000}, results=["%db"], operands=[]),
        op("tt.splat", [ptr("i32")], pt, results=["%sp"], operands=["%sb"]),
        op("tt.splat", [ptr("i32")], pt, results=["%dp"], operands=["%db"]),
        op("tt.addptr", [pt, T4], pt, results=["%si"], operands=["%sp", "%r"]),
        op("tt.addptr", [pt, T4], pt, results=["%di"], operands=["%dp", "%r"]),
        const("%three", 3, T4),
        op(
            "arith.cmpi",
            [T4, T4],
            T4B,
            attrs={"predicate": 2},
            results=["%m"],
            operands=["%r", "%three"],
        ),
        op("tt.load", [pt, T4B], T4, results=["%v"], operands=["%si", "%m"]),
        const("%one", 1, T4),
        op("arith.addi", [T4, T4], T4, results=["%w"], operands=["%v", "%one"]),
        op("tt.store", [pt, T4, T4B], [], operands=["%di", "%w", "%m"], results=[]),
        op("tt.return", [], [], operands=[], results=[]),
    ]
    interp(func("kernel", [], body), memory).run("kernel", [])
    assert np.array_equal(dst, np.array([1, 2, 3, -1], np.int32))
    assert memory.buffers()[0x2000] is dst


def test_in_kernel_tensormap_round_trips_into_a_descriptor_load() -> None:
    """`global_scratch_alloc`, `tensormap_create` and `reinterpret_tensor_descriptor` model a
    device-built TMA descriptor: the load reads the 2x2 block at (1, 0) of a 3x4 f32 tensor."""
    from conftest import ptr, tensor, tensordesc

    memory = Memory()
    memory.register(8192, np.arange(12, dtype=np.float32))
    create = op(
        "ttng.tensormap_create",
        [ptr("i8"), ptr("f32"), I32, I32, I32, I32, ty("i64"), I32, I32],
        [],
        attrs={"operandSegmentSizes": [1, 1, 2, 2, 1, 2], "elem_type": 7},
        operands=["%d", "%base", "%box1", "%box0", "%dim1", "%dim0", "%stride0", "%one", "%one"],
        results=[],
    )
    module = func(
        "k",
        [("%base", ptr("f32"))],
        [
            const("%box1", 2),
            const("%box0", 2),
            const("%dim1", 4),
            const("%dim0", 3),
            const("%one", 1),
            const("%stride0", 16, ty("i64")),
            const("%c1", 1),
            const("%c0", 0),
            op(
                "ttg.global_scratch_alloc",
                [],
                ptr("i8"),
                attrs={"nbytes": 128, "alignment": 128},
                results=["%d"],
            ),
            create,
            op("ttng.tensormap_fenceproxy_acquire", [ptr("i8")], [], operands=["%d"], results=[]),
            op(
                "ttng.reinterpret_tensor_descriptor",
                [ptr("i8")],
                tensordesc((2, 2), "f32"),
                operands=["%d"],
                results=["%desc"],
            ),
            op(
                "tt.descriptor_load",
                [tensordesc((2, 2), "f32"), I32, I32],
                tensor((2, 2), "f32"),
                operands=["%desc", "%c1", "%c0"],
                results=["%blk"],
            ),
            op("tt.return", [tensor((2, 2), "f32")], [], operands=["%blk"], results=[]),
        ],
    )
    (got,) = run(module, "k", [np.int64(8192)], memory=memory)
    assert np.array_equal(got, np.array([[4.0, 5.0], [8.0, 9.0]], np.float32))


def test_tt_call_runs_the_callee_and_returns_its_results() -> None:
    callee = Op(
        name="tt.func",
        attrs={"sym_name": "add2"},
        regions=[
            region(
                block(
                    [("%x", I32)],
                    [
                        const("%two", 2),
                        add("%y", "%x", "%two"),
                        op("tt.return", [I32], [], operands=["%y"], results=[]),
                    ],
                )
            )
        ],
    )
    module = func(
        "k",
        [],
        [
            const("%five", 5),
            op(
                "tt.call", [I32], I32, attrs={"callee": "@add2"}, operands=["%five"], results=["%r"]
            ),
            op("tt.return", [I32], [], operands=["%r"], results=[]),
        ],
    )
    module.funcs["add2"] = callee
    (got,) = run(module, "k", [])
    assert int(got) == 7


def test_ub_poison_is_a_zero_of_its_type() -> None:
    module = func(
        "k",
        [],
        [
            op("ub.poison", [], tensor((2, 3), "i32"), results=["%p"]),
            op("tt.return", [tensor((2, 3), "i32")], [], operands=["%p"], results=[]),
        ],
    )
    (got,) = run(module, "k", [])
    assert got.shape == (2, 3) and got.dtype == np.int32 and not got.any()
