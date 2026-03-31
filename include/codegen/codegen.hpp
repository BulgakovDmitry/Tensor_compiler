#ifndef INCLUDE_CODEGEN_CODEGEN_HPP
#define INCLUDE_CODEGEN_CODEGEN_HPP

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"

namespace tensor_compiler {

class Codegen {
  private:
  public:
    Codegen();

    mlir::OwningOpRef<mlir::ModuleOp> generate(const Graph& graph);
};

} // namespace tensor_compiler

#endif // INCLUDE_CODEGEN_CODEGEN_HPP
