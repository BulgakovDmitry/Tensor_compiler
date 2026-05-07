#include "Lowering/LLVMToLLVMIR.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

namespace tensor_compiler {
std::unique_ptr<llvm::Module> LLVMToLLVMIR(
    llvm::LLVMContext &llvmCtx,
    mlir::OwningOpRef<mlir::ModuleOp> &mlirModule) {
    if (!mlirModule) {
        llvm::errs() << "Error: No module to export\n";
        return nullptr;
    }

    std::unique_ptr<llvm::Module> llvmModule =
        translateModuleToLLVMIR(*mlirModule, llvmCtx, "tensor_network");

    if (!llvmModule) {
        llvm::errs() << "Error: translateModuleToLLVMIR returned nullptr\n";

        llvm::errs() << "=== Module content before export ===\n";
        mlirModule->print(llvm::errs());
        llvm::errs() << "\n=== End module ===\n";

        return nullptr;
    }

    return llvmModule;
}

} // namespace tensor_compiler
