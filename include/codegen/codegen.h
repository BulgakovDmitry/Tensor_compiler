#ifndef INCLUDE_CODEGEN_CODEGEN_H
#define INCLUDE_CODEGEN_CODEGEN_H

#include <memory>

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
#include "structure/graph.h"

namespace tensor_compiler {

class Codegen {
private:
  mlir::DialectRegistry registry_;
  std::unique_ptr<mlir::MLIRContext> context_;

public:
  Codegen();

  mlir::OwningOpRef<mlir::ModuleOp> generate(const Graph &graph);

  mlir::MLIRContext &getContext() noexcept;
  const mlir::MLIRContext &getContext() const noexcept;

private:
  mlir::Type convertElementType(int onnx_type) const;
  mlir::RankedTensorType convertTensorType(const Tensor &tensor) const;

  std::vector<mlir::Type> buildInputTypes(const Graph &graph) const;
  std::vector<mlir::Type> buildResultTypes(const Graph &graph) const;

  void bindFunctionInputs(const Graph &graph,
                            mlir::Block *entry_block,
                            std::unordered_map<std::string, mlir::Value> &values) const;


  std::vector<mlir::Value> collectReturnValues(
      const Graph &graph,
      const std::unordered_map<std::string, mlir::Value> &values) const;

  void genNodes(mlir::OpBuilder &builder,
                mlir::Location loc,
                const Graph &graph,
                std::unordered_map<std::string, mlir::Value> &values) const;

  void genNode(mlir::OpBuilder &builder,
                mlir::Location loc,
                const Node &node,
                std::unordered_map<std::string, mlir::Value> &values) const;

  void genMulNode(mlir::OpBuilder &builder,
                   mlir::Location loc,
                   const Node &node,
                   std::unordered_map<std::string, mlir::Value> &values) const;

  mlir::Value genConstantTensor(mlir::OpBuilder &builder,
                                mlir::Location loc,
                                const Tensor &tensor) const;

  void genConstants(mlir::OpBuilder &builder,
                     mlir::Location loc,
                     const Graph &graph,
                     std::unordered_map<std::string, mlir::Value> &values) const;

  void genAddNode(mlir::OpBuilder &builder,
                   mlir::Location loc,
                   const Node &node,
                   std::unordered_map<std::string, mlir::Value> &values) const;

  void genIdentityNode(mlir::OpBuilder &builder,
                        mlir::Location loc,
                        const Node &node,
                        std::unordered_map<std::string, mlir::Value> &values) const;

  void genSubNode(mlir::OpBuilder &builder,
                   mlir::Location loc,
                   const Node &node,
                   std::unordered_map<std::string, mlir::Value> &values) const;

  void genDivNode(mlir::OpBuilder &builder,
                   mlir::Location loc,
                   const Node &node,
                   std::unordered_map<std::string, mlir::Value> &values) const;

  void genReluNode(mlir::OpBuilder &builder,
                    mlir::Location loc,
                    const Node &node,
                    std::unordered_map<std::string, mlir::Value> &values) const;
};

} // namespace tensor_compiler

#endif // INCLUDE_CODEGEN_CODEGEN_H
