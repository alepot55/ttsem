"builtin.module"() ({
  "tt.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {}, {tt.divisibility = 16 : i32}], function_type = (!tt.ptr<f32>, !tt.ptr<f32>, !tt.ptr<f32>, i32, i32) -> (), sym_name = "kernel", sym_visibility = "public"}> ({
  ^bb0(%p: !tt.ptr<f32>, %out: !tt.ptr<f32>, %sink: !tt.ptr<f32>, %m: i32, %n: i32):
    %acc = "arith.constant"() <{value = dense<0.000000e+00> : tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>}> : () -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %3 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %offs = "arith.constant"() <{value = dense<0> : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>}> : () -> tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %4 = "arith.cmpi"(%m, %3) <{predicate = 0 : i64}> : (i32, i32) -> i1
    %5 = "scf.if"(%4) ({
      %14 = "arith.cmpi"(%n, %3) <{predicate = 4 : i64}> : (i32, i32) -> i1
      "llvm.intr.assume"(%14) <{op_bundle_sizes = array<i32>}> : (i1) -> ()
      %acc_13 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>}> : () -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
      "scf.yield"(%acc_13) : (tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> ()
    }, {
      %acc_0 = "scf.for"(%3, %m, %0, %acc) ({
      ^bb0(%i: i32, %acc_1: tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>):
        %acc_2 = "scf.for"(%3, %n, %0, %acc_1) ({
        ^bb0(%j: i32, %acc_3: tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>):
          %acc_9 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>}> : () -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
          %10 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %11 = "tt.addptr"(%sink, %10) : (!tt.ptr<f32>, i32) -> !tt.ptr<f32>
          %12 = "tt.splat"(%11) : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
          %13 = "tt.addptr"(%12, %offs) : (tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
          "tt.store"(%13, %acc_9) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32}> : (tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> ()
          "scf.yield"(%acc_9) : (tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> ()
        }) : (i32, i32, i32, tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
        "scf.yield"(%acc_2) : (tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> ()
      }) {tt.flatten} : (i32, i32, i32, tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
      "scf.yield"(%acc_0) : (tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> ()
    }) : (i1) -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    "tt.return"() : () -> ()
  }) {noinline = false} : () -> ()
}) {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} : () -> ()