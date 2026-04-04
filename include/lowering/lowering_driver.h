#ifndef INCLUDE_LOWERING_DRIVER_HPP
#define INCLUDE_LOWERING_DRIVER_HPP

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace tensor_compiler {

class MLIRToLLVMLowering final {
private:
  mlir::MLIRContext &context_;
  mlir::OwningOpRef<mlir::ModuleOp> mlirModule_;
};

} // namespace tensor_compiler

#endif // INCLUDE_LOWERING_DRIVER_HPP
