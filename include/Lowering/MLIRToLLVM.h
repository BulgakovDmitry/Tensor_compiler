#ifndef INCLUDE_LOWERING_MLIRTOLLVM_H
#define INCLUDE_LOWERING_MLIRTOLLVM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/MLIRContext.h"

namespace tensor_compiler {
mlir::LogicalResult MLIRToLLVM(
  mlir::MLIRContext &context,
  mlir::OwningOpRef<mlir::ModuleOp> &mlirModule
);
} // namespace tensor_compiler

#endif // INCLUDE_LOWERING_MLIRTOLLVM_H
