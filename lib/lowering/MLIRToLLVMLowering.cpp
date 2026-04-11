#include "lowering/MLIRToLLVMLowering.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include <iostream>

using namespace mlir;

namespace tensor_compiler {

MLIRToLLVMLowering::MLIRToLLVMLowering(mlir::MLIRContext &context)
    : context_{context}
    , mlirModule_(nullptr) {}

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

    pm.addPass(arith::createArithBufferizePass());
    pm.addPass(func::createFuncBufferizePass());
    pm.addNestedPass<func::FuncOp>(createLinalgBufferizePass());
    pm.addNestedPass<func::FuncOp>(tensor::createTensorBufferizePass());

    bufferization::OneShotBufferizationOptions bufferizationOptions;
    bufferizationOptions.bufferizeFunctionBoundaries = true;
    bufferizationOptions.setFunctionBoundaryTypeConversion(
        bufferization::LayoutMapOption::IdentityLayoutMap);
    pm.addPass(bufferization::createOneShotBufferizePass(bufferizationOptions));

    pm.addPass(bufferization::createBufferResultsToOutParamsPass());

    pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
    pm.addPass(createConvertSCFToCFPass());

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
        llvm::errs() << "=== FAILED: pm.run ===\n";
        mlirModule_->print(llvm::errs());
        return failure();
    }

    return success();
}

std::unique_ptr<llvm::Module> MLIRToLLVMLowering::MLIRToLLVMLowering::exportToLLVM(llvm::LLVMContext &llvmCtx) {
    if (!mlirModule_) {
        llvm::errs() << "Error: No module to export\n";
        return nullptr;
    }

    std::unique_ptr<llvm::Module> llvmModule = translateModuleToLLVMIR(*mlirModule_, llvmCtx, "tensor_network");

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
