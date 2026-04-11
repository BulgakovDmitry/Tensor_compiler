#ifndef INCLUDE_LOWERING_LLVMTOASMLOWERING_H
#define INCLUDE_LOWERING_LLVMTOASMLOWERING_H

#include "mlir/Support/LogicalResult.h"

namespace llvm {
class Module;
class raw_pwrite_stream;
} // namespace llvm

#include <string>

namespace tensor_compiler {

mlir::LogicalResult generateAssembly(
    llvm::Module *llvmModule,
    const std::string &triple,
    unsigned optLevel,
    llvm::raw_pwrite_stream &os
);
} // namespace tensor_compiler

#endif // INCLUDE_LOWERING_LLVMTOASMLOWERING_H
