"builtin.module"() ({
  "tt.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}], function_type = (!tt.ptr<f32>, !tt.ptr<f32>, !tt.ptr<i32>, i32) -> (), sym_name = "kload", sym_visibility = "public"}> ({
  ^bb0(%p: !tt.ptr<f32>, %out: !tt.ptr<f32>, %q: !tt.ptr<i32>, %m: i32):
    %acc = "arith.constant"() <{value = dense<0.000000e+00> : tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>}> : () -> tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %acc_0 = "scf.for"(%1, %m, %0, %acc) ({
    ^bb0(%_i: i32, %acc_1: tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>):
      %bound = "tt.load"(%q) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (!tt.ptr<i32>) -> i32
      %acc_2 = "scf.for"(%1, %bound, %0, %acc_1) ({
      ^bb0(%j: i32, %acc_3: tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>):
        %acc_9 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>}> : () -> tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
        "scf.yield"(%acc_9) : (tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> ()
      }) : (i32, i32, i32, tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
      "scf.yield"(%acc_2) : (tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> ()
    }) {tt.flatten} : (i32, i32, i32, tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    "tt.return"() : () -> ()
  }) {noinline = false} : () -> ()
}) {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} : () -> ()