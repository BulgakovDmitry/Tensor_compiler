#include "Lowering/MLIRToLLVM.h"
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

using namespace mlir;

namespace tensor_compiler {
LogicalResult MLIRToLLVM(MLIRContext &context,
                        OwningOpRef<ModuleOp> &mlirModule) {
    if (!mlirModule) {
        llvm::errs() << "Error: Received null MLIR module\n";
        return failure();
    }

    PassManager pm(&context);

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

    if (failed(pm.run(*mlirModule))) {
        llvm::errs() << "=== FAILED: pm.run ===\n";
        mlirModule->print(llvm::errs());
        return failure();
    }

    return success();
}

} // namespace tensor_compiler
