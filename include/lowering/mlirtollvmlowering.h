#ifndef INCLUDE_LOWERING_MLIRTOLLVMLOWERING_H
#define INCLUDE_LOWERING_MLIRTOLLVMLOWERING_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LogicalResult.h"

#include <memory>

namespace llvm {
class Module;
}

namespace tensor_compiler {

class MLIRToLLVMLowering final {
private:
  mlir::MLIRContext &context_;
  mlir::OwningOpRef<mlir::ModuleOp> mlirModule_;

public:
  MLIRToLLVMLowering(mlir::MLIRContext &context) : context_{context} {}

  mlir::LogicalResult lower(mlir::OwningOpRef<mlir::ModuleOp> &&mlirModule);
  std::unique_ptr<llvm::Module> exportToLLVM();
};

} // namespace tensor_compiler

#endif // INCLUDE_LOWERING_DRIVER_H
