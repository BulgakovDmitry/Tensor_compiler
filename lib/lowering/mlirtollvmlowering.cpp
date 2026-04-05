#include "lowering/mlirtollvmlowering.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Export.h"

namespace tensor_compiler {

mlir::LogicalResult MLIRToLLVMLowering::lower(mlir::OwningOpRef<mlir::ModuleOp> &&mlirModule) {
    if (!mlirModule) {
        llvm::errs() << "Error: Received null MLIR module\n";
        return mlir::failure();
    }

    mlirModule_ = std::move(mlirModule);
    mlir::PassManager pm(&context_);

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
    pm.addPass(mlir::createConvertSCFToCFPass());

    mlir::LLVMTypeConverter typeConverter(&context_);

    //pm.addPass(mlir)
}

} // tensor_compiler
