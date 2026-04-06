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
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

using namespace mlir;

namespace tensor_compiler {

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
    if (!mlirModule_) {
        llvm::errs() << "Error: No module to export\n";
        return nullptr;
    }

    mlir::DialectRegistry registry;
    mlir::registerBuiltinDialectTranslation(registry); 
    mlir::registerLLVMDialectTranslation(registry);

    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();

    llvm::LLVMContext llvmCtx;
    auto llvmModule = translateModuleToLLVMIR(*mlirModule_, llvmCtx, "tensor_network");

    if (!llvmModule) {
        llvm::errs() << "Error: translateModuleToLLVMIR returned nullptr\n";

        llvm::errs() << "=== Module content before export ===\n";
        mlirModule_->print(llvm::errs());
        llvm::errs() << "\n=== End module ===\n";

        return nullptr;
    }

    return llvmModule;
}


} // namespace tensor_compiler
