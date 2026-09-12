"builtin.module"() ({
  "tt.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}], function_type = (!tt.ptr<i32>, !tt.ptr<i32>, !tt.ptr<i32>) -> (), sym_name = "fuzz", sym_visibility = "public"}> ({
  ^bb0(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>):
    %3 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %4 = "arith.constant"() <{value = 64 : i32}> : () -> i32
    %5 = "arith.constant"() <{value = 16 : i32}> : () -> i32
    %7 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %8 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %11 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<16x16xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [16, 8]}>>}> : () -> tensor<16x16xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [16, 8]}>>
    %12 = "arith.constant"() <{value = dense<0> : tensor<256xi32, #ttg.linear<{register = [[1], [128]], lane = [[2], [4], [16], [32], [64]], warp = [[8], [0], [0]], block = []}>>}> : () -> tensor<256xi32, #ttg.linear<{register = [[1], [128]], lane = [[2], [4], [16], [32], [64]], warp = [[8], [0], [0]], block = []}>>
    "scf.for"(%7, %3, %8) ({
    ^bb0(%arg3: i32):
      %20 = "scf.for"(%7, %4, %5, %11) ({
      ^bb0(%arg6: i32, %arg7: tensor<16x16xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [16, 8]}>>):
        %76 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<16x16xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [16, 8]}>>}> : () -> tensor<16x16xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [16, 8]}>>
        "scf.yield"(%76) : (tensor<16x16xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [16, 8]}>>) -> ()
      }) {tt.num_stages = 2 : i32} : (i32, i32, i32, tensor<16x16xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [16, 8]}>>) -> tensor<16x16xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [16, 8]}>>
      %21 = "tt.reshape"(%20) : (tensor<16x16xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [16, 8]}>>) -> tensor<256xf32, #ttg.linear<{register = [[1], [128]], lane = [[2], [4], [16], [32], [64]], warp = [[8], [0], [0]], block = []}>>
      %22 = "arith.fptosi"(%21) : (tensor<256xf32, #ttg.linear<{register = [[1], [128]], lane = [[2], [4], [16], [32], [64]], warp = [[8], [0], [0]], block = []}>>) -> tensor<256xi32, #ttg.linear<{register = [[1], [128]], lane = [[2], [4], [16], [32], [64]], warp = [[8], [0], [0]], block = []}>>
      %23 = "scf.while"(%7) ({
      ^bb0(%arg5: i32):
        %35 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %36 = "arith.cmpi"(%35, %7) <{predicate = 1 : i64}> : (i32, i32) -> i1
        "scf.condition"(%36, %arg5) : (i1, i32) -> ()
      }, {
      ^bb0(%arg4: i32):
        %26 = "arith.maxsi"(%22, %12) : (tensor<256xi32, #ttg.linear<{register = [[1], [128]], lane = [[2], [4], [16], [32], [64]], warp = [[8], [0], [0]], block = []}>>, tensor<256xi32, #ttg.linear<{register = [[1], [128]], lane = [[2], [4], [16], [32], [64]], warp = [[8], [0], [0]], block = []}>>) -> tensor<256xi32, #ttg.linear<{register = [[1], [128]], lane = [[2], [4], [16], [32], [64]], warp = [[8], [0], [0]], block = []}>>
        %32 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        "scf.yield"(%32) : (i32) -> ()
      }) : (i32) -> i32
      "scf.yield"() : () -> ()
    }) {tt.flatten} : (i32, i32, i32) -> ()
    "tt.return"() : () -> ()
  }) {noinline = false} : () -> ()
}) {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} : () -> ()
