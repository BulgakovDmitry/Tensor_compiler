#ifndef INCLUDE_LOWERING_LLVMTOLLVMIR_H
#define INCLUDE_LOWERING_LLVMTOLLVMIR_H

#include "llvm/IR/LLVMContext.h"
#include "mlir/IR/BuiltinOps.h"

namespace llvm {
class Module;
}

namespace tensor_compiler {
std::unique_ptr<llvm::Module> LLVMToLLVMIR(
    llvm::LLVMContext &llvmCtx,
    mlir::OwningOpRef<mlir::ModuleOp> &mlirModule
);
} // namespace tensor_compiler

#endif // INCLUDE_LOWERING_LLVMTOLLVMIR_H
