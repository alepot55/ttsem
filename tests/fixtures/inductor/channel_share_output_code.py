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
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
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


# Topologically Sorted Source Nodes: [sum_1, x, mul], Original ATen: [aten.sum, aten.div, aten.mul, aten.copy_]
# Source node to ATen node mapping:
#   mul => mul
#   sum_1 => sum_1
#   x => div
# Graph fragment:
#   %copy_ : Tensor "f32[3, 64, 64][4096, 64, 1]cuda:0" = PlaceHolder[target=copy_]
#   %div : Tensor "f32[3, 64, 64][4096, 64, 1]cuda:0" = PlaceHolder[target=div]
#   %sum_1 : Tensor "f32[1, 64, 64][4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%arg0_1, [0], True), kwargs = {})
#   %div : Tensor "f32[3, 64, 64][4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg0_1, %sum_1), kwargs = {})
#   %mul : Tensor "f32[3, 64, 64][4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, 2.0), kwargs = {})
#   %copy_ : Tensor "f32[3, 64, 64][4096, 64, 1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg0_1, %div), kwargs = {})
#   return %mul,%div,%buf3
triton_poi_fused_copy__div_mul_sum_0 = async_compile.triton('triton_poi_fused_copy__div_mul_sum_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy__div_mul_sum_0', 'mutated_arg_names': ['in_ptr0', 'out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '568339AA377E4FC0C6A5D38B74F02C2CBA1C400D191E4138AF32BAE843721057', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 294912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy__div_mul_sum_0(in_ptr0, out_ptr0, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 4096)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (4096 + x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (8192 + x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp0 / tmp5)
    tmp7 = 2.0
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
    tl.store(out_ptr2 + (x2), tmp6, None)
''', device_str='cuda')


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
        assert_size_stride(arg0_1, (3, 64, 64), (4096, 64, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((3, 64, 64), (4096, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sum_1, x, mul], Original ATen: [aten.sum, aten.div, aten.mul, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_poi_fused_copy__div_mul_sum_0.run(arg0_1, buf0, arg0_1, 12288, stream=stream0)
            del arg0_1
        return (buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((3, 64, 64), (4096, 64, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
