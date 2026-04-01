#include "../../include/codegen/codegen.hpp"

namespace tensor_compiler {

Codegen::Codegen() {
    registry_.insert<mlir::func::FuncDialect>();
    context_ = std::make_unique<mlir::MLIRContext>(registry_);
    context_->loadAllAvailableDialects();
}

mlir::MLIRContext &Codegen::get_context() noexcept {
    return *context_;
}

const mlir::MLIRContext &Codegen::get_context() const noexcept {
    return *context_;
}

mlir::OwningOpRef<mlir::ModuleOp> Codegen::generate(const Graph& graph) {
    mlir::OpBuilder builder(context_.get());
    
    mlir::Location loc = builder.getUnknownLoc();

    mlir::ModuleOp module = mlir::ModuleOp::create(loc);

    std::string funcName = graph.get_name().empty() ? "main" : graph.get_name();

    mlir::FunctionType funcType = builder.getFunctionType(
        llvm::ArrayRef<mlir::Type>{},
        llvm::ArrayRef<mlir::Type>{}
    );

    mlir::func::FuncOp func = mlir::func::FuncOp::create(loc, funcName, funcType);

    mlir::Block *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    builder.create<mlir::func::ReturnOp>(loc);

    module.push_back(func);

    return module;
}

} // namespace tensor_compiler