cmake -S . -B build -DGRAPH_DUMP=ON \
-DMLIR_DIR=$HOME/Desktop/coding/llvm-project/build/lib/cmake/mlir \
-DLLVM_DIR=$HOME/Desktop/coding/llvm-project/build/lib/cmake/llvm

cmake --build build