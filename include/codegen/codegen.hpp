#ifndef INCLUDE_CODEGEN_CODEGEN_HPP
#define INCLUDE_CODEGEN_CODEGEN_HPP

#include <memory>

#include "graph.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace tensor_compiler {

class Codegen {
  private:
    mlir::DialectRegistry registry_;
    std::unique_ptr<mlir::MLIRContext> context_;

  public:
    Codegen();

    mlir::OwningOpRef<mlir::ModuleOp> generate(const Graph &graph);

    mlir::MLIRContext &get_context() noexcept;
    const mlir::MLIRContext &get_context() const noexcept;
};

} // namespace tensor_compiler

#endif // INCLUDE_CODEGEN_CODEGEN_HPP
