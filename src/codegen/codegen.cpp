#include "../../include/codegen/codegen.hpp"

namespace tensor_compiler {

Codegen::Codegen() {
    registry_.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                     mlir::tensor::TensorDialect>();

    context_ = std::make_unique<mlir::MLIRContext>(registry_);
    context_->loadAllAvailableDialects();
}

mlir::MLIRContext &Codegen::get_context() noexcept { return *context_; }

const mlir::MLIRContext &Codegen::get_context() const noexcept {
    return *context_;
}

mlir::OwningOpRef<mlir::ModuleOp> Codegen::generate(const Graph &graph) {
    mlir::OpBuilder builder(context_.get());
    mlir::Location loc = builder.getUnknownLoc();

    mlir::ModuleOp module = mlir::ModuleOp::create(loc);

    std::string func_name =
        graph.get_name().empty() ? "main" : graph.get_name();

    std::vector<mlir::Type> input_types;
    input_types.reserve(graph.get_inputs().size());

    for (const auto &input_name : graph.get_inputs()) {
        const Tensor *tensor = graph.get_tensor(input_name);
        if (!tensor) {
            throw std::runtime_error("graph input tensor not found: " +
                                     input_name);
        }
        input_types.push_back(convert_tensor_type(*tensor));
    }

    std::vector<mlir::Type> result_types;
    result_types.reserve(graph.get_outputs().size());

    for (const auto &output_name : graph.get_outputs()) {
        const Tensor *tensor = graph.get_tensor(output_name);
        if (!tensor) {
            throw std::runtime_error("graph output tensor not found: " +
                                     output_name);
        }
        result_types.push_back(convert_tensor_type(*tensor));
    }

    mlir::FunctionType func_type =
        builder.getFunctionType(input_types, result_types);
    mlir::func::FuncOp func =
        mlir::func::FuncOp::create(loc, func_name, func_type);

    mlir::Block *entry_block = func.addEntryBlock();
    builder.setInsertionPointToStart(entry_block);

    if (graph.get_outputs().empty()) {
        builder.create<mlir::func::ReturnOp>(loc);
    } else {
        if (entry_block->getNumArguments() == 0) {
            throw std::runtime_error("cannot return function argument: graph "
                                     "has outputs but no inputs");
        }

        mlir::Value returned_value = entry_block->getArgument(0);
        builder.create<mlir::func::ReturnOp>(loc, returned_value);
    }

    module.push_back(func);
    return module;
}

mlir::Type Codegen::convert_element_type(int onnx_type) const {
    switch (onnx_type) {
    case onnx::TensorProto_DataType_FLOAT:
        return mlir::Float32Type::get(context_.get());

    case onnx::TensorProto_DataType_DOUBLE:
        return mlir::Float64Type::get(context_.get());

    case onnx::TensorProto_DataType_INT64:
        return mlir::IntegerType::get(context_.get(), 64);

    case onnx::TensorProto_DataType_INT32:
        return mlir::IntegerType::get(context_.get(), 32);

    default:
        throw std::runtime_error("unsupported ONNX tensor element type");
    }
}

mlir::RankedTensorType
Codegen::convert_tensor_type(const Tensor &tensor) const {
    const auto &shape = tensor.get_shape();
    auto elem_type = convert_element_type(tensor.get_type());
    return mlir::RankedTensorType::get(shape, elem_type);
}

} // namespace tensor_compiler