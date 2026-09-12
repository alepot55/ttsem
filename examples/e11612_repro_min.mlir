"builtin.module"() ({
  "tt.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}], function_type = (!tt.ptr<i32>) -> (), sym_name = "kernel", sym_visibility = "public"}> ({
  ^bb0(%arg0: !tt.ptr<i32>):
    %0 = "arith.constant"() <{value = 4 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %2 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %3 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    "scf.for"(%1, %3, %2) ({
    ^bb0(%arg1: i32):
      %4 = "scf.for"(%1, %0, %2, %1) ({
      ^bb0(%arg4: i32, %arg5: i32):
        %9 = "arith.addi"(%arg5, %arg4) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "scf.yield"(%9) : (i32) -> ()
      }) : (i32, i32, i32, i32) -> i32
      %5 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %6 = "arith.addi"(%4, %5) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "tt.store"(%arg0, %6) <{cache = 1 : i32, evict = 1 : i32}> : (!tt.ptr<i32>, i32) -> ()
      "scf.yield"() : () -> ()
    }) {tt.flatten} : (i32, i32, i32) -> ()
    "tt.return"() : () -> ()
  }) {noinline = false} : () -> ()
}) {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} : () -> ()
