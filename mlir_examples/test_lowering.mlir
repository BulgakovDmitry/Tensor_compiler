module {
  func.func @sum_first_n(%n: index) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = arith.constant 0 : i32

    %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
      %iv_i32 = arith.index_cast %i : index to i32
      %next = arith.addi %acc, %iv_i32 : i32
      scf.yield %next : i32
    }

    return %sum : i32
  }
}
