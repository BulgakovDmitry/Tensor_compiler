#ifndef INCLUDE_LOWERING_MLIRTOLLVMLOWERING_H
#define INCLUDE_LOWERING_MLIRTOLLVMLOWERING_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace tensor_compiler {

class MLIRToLLVMLowering final {
private:
  mlir::MLIRContext &context_;
  mlir::OwningOpRef<mlir::ModuleOp> mlirModule_;

public:
  
};

} // namespace tensor_compiler

#endif // INCLUDE_LOWERING_DRIVER_H
