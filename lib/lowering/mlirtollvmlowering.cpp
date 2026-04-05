#include "lowering/mlirtollvmlowering.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Export.h"

using namespace mlir;
using namespace tensor_compiler;

LogicalResult MLIRToLLVMLowering::lower(OwningOpRef<ModuleOp> &&mlirModule) {
    if (!mlirModule) {
        llvm::errs() << "Error: Received null MLIR module\n";
        return failure();
    }

    mlir::DialectRegistry registry;
    registerConvertMemRefToLLVMInterface(registry);
    context_.appendDialectRegistry(registry);

    mlirModule_ = std::move(mlirModule);
    PassManager pm(&context_);

    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
    // pm.addPass(createConvertSCFToCFPass());

    pm.addPass(createConvertMathToLLVMPass());
    pm.addPass(createArithToLLVMConversionPass());

    pm.addPass(createFinalizeMemRefToLLVMConversionPass());

    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(createConvertFuncToLLVMPass());

    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    if (failed(pm.run(*mlirModule_))) {
        llvm::errs() << "Lowering to LLVM dialect failed!\n";
        return failure();
    }
    return success();
}

std::unique_ptr<llvm::Module> MLIRToLLVMLowering::MLIRToLLVMLowering::exportToLLVM() {
    if (!mlirModule_) return nullptr;
    llvm::LLVMContext llvmCtx;
    return translateModuleToLLVMIR(*mlirModule_, llvmCtx, "tensor_network");
}
