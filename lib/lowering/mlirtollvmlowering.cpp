#include "lowering/mlirtollvmlowering.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

using namespace mlir;

namespace tensor_compiler {

MLIRToLLVMLowering::MLIRToLLVMLowering(mlir::MLIRContext &context) : context_{context} {}

LogicalResult MLIRToLLVMLowering::lower(OwningOpRef<ModuleOp> &&mlirModule) {
    if (!mlirModule) {
        llvm::errs() << "Error: Received null MLIR module\n";
        return failure();
    }

    mlirModule_ = std::move(mlirModule);
    PassManager pm(&context_);

    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    pm.addNestedPass<func::FuncOp>(createConvertElementwiseToLinalgPass());
    pm.addPass(createPrintOpStatsPass());

    bufferization::OneShotBufferizationOptions bufferizationOptions;
    bufferizationOptions.bufferizeFunctionBoundaries = true;
    bufferizationOptions.setFunctionBoundaryTypeConversion(
        bufferization::LayoutMapOption::IdentityLayoutMap);
    pm.addPass(bufferization::createOneShotBufferizePass(bufferizationOptions));
    pm.addPass(createPrintOpStatsPass());

    pm.addPass(bufferization::createBufferResultsToOutParamsPass());
    pm.addPass(createPrintOpStatsPass());

    pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
    pm.addPass(createConvertSCFToCFPass());
    pm.addPass(createPrintOpStatsPass());

    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

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
