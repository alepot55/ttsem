# What TorchInductor 2.14 wrote for `(x > 0.5).to(torch.int64).cumsum(0) * 2` at 16,384
# elements with cpu_backend="triton" (the companion `emit_cpu.py`): a decoupled-lookback split
# scan of 64-bit values, whose partials go through plain stores and loads of a scratch buffer.
# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph

def get_raw_stream(_):
    return 0


aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_size_stride_grouped = torch._C._dynamo.guards.assert_size_stride_grouped
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# Topologically Sorted Source Nodes: [gt, to, cumsum, mul], Original ATen: [aten.gt, aten._to_copy, aten.cumsum, aten.mul]
# Source node to ATen node mapping:
#   cumsum => cumsum
#   gt => gt
#   mul => mul
#   to => convert_element_type
# Graph fragment:
#   %arg0_1 : Tensor "f32[16384][1]cpu" = PlaceHolder[target=arg0_1]
#   %cumsum : Tensor "i64[16384][1]cpu" = PlaceHolder[target=cumsum]
#   %gt : Tensor "b8[16384][1]cpu"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%arg0_1, 0.5), kwargs = {})
#   %convert_element_type : Tensor "i64[16384][1]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt, torch.int64), kwargs = {})
#   %cumsum : Tensor "i64[16384][1]cpu"[num_users=1] = call_function[target=torch.ops.aten.cumsum.default](args = (%convert_element_type, 0), kwargs = {})
#   %mul : Tensor "i64[16384][1]cpu"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cumsum, 2), kwargs = {})
#   return %cumsum,%mul
triton_spl_fused__to_copy_cumsum_gt_mul_0 = async_compile.triton('triton_spl_fused__to_copy_cumsum_gt_mul_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_cpu()

@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0

@triton_heuristics.split_scan(
    size_hints={'x': 1, 'r0_': 16384},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*i64', 'in_ptr0': '*fp32', 'ws_ptr': '*u8', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cpu', index=None, multi_processor_count=16, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SplitScanGrid', 'kernel_name': 'triton_spl_fused__to_copy_cumsum_gt_mul_0', 'mutated_arg_names': ['in_out_ptr0', 'ws_ptr'], 'optimize_mem': True, 'no_x_dim': True, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'r0_': 327680}, 'backend_hash': '03172E6127E8FE10B6890F45680EE3C5F2D9396F1CBB9EB2BB227060CB2FDCA4', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_spl_fused__to_copy_cumsum_gt_mul_0(in_out_ptr0, in_ptr0, ws_ptr, xnumel, r0_numel, R0_BLOCK : tl.constexpr):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    r0_numel = 16384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(1) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_offset = tl.program_id(0) * R0_BLOCK
    r0_index = r0_offset + tl.arange(0, R0_BLOCK)[:]
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = tl.full([1], 0.5, tl.float32)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp2.to(tl.int64)
    tmp4 = tmp3.to(tl.int64)
    tmp5 = tl.broadcast_to(tmp4, [R0_BLOCK])
    tmp6 = tl.reduce(tmp5, 0, _triton_helper_fn_add0)
    tmp7 = triton_helpers.exclusive_scan_decoupled_lookback_64(
        ws_ptr.to(tl.pointer_type(tl.uint64)) + xoffset * 3 * tl.num_programs(0),
        tmp6,
        tl.program_id(0),
        _triton_helper_fn_add0,
    )
    tmp8 = tl.associative_scan(tmp5, 0, _triton_helper_fn_add0)
    tmp9 = _triton_helper_fn_add0(tmp7, tmp8)
    tmp10 = tl.where(roffset == 0, tmp8, tmp9)
    tmp11 = tl.full([1], 2, tl.int64)
    tmp12 = tmp10 * tmp11
    tl.store(in_out_ptr0 + (tl.broadcast_to(r0_0, [R0_BLOCK])), tmp12, r0_mask)
''', device_str='cpu')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, = args
        args.clear()
        assert_size_stride(arg0_1, (16384, ), (1, ), 'input')
        buf0 = empty_strided_cpu((16384, ), (1, ), torch.int64)
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [gt, to, cumsum, mul], Original ATen: [aten.gt, aten._to_copy, aten.cumsum, aten.mul]
        workspace_0 = empty_strided_cpu((1536, ), (1, ), torch.uint8)
        workspace_0.zero_()
        raw_streamNone = get_raw_stream(None)
        triton_spl_fused__to_copy_cumsum_gt_mul_0.run(buf1, arg0_1, workspace_0, 1, 16384, stream=raw_streamNone)
        del workspace_0
        del arg0_1
        return (buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    return [arg0_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cpu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))

