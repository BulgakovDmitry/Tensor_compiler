#include <unordered_map>
#include <vector>
#include <string>
#include "codegen/Codegen.h"

namespace tensor_compiler {

namespace {

mlir::Value getBoundValue(
    const std::unordered_map<std::string, mlir::Value> &values,
    const std::string &name,
    const char *opName) {

    auto it = values.find(name);
    if (it == values.end()) {
        throw std::runtime_error(std::string(opName) +
                                 " input not bound: " + name);
    }
    return it->second;
}

void checkUnaryNodeShape(const Node &node, const char *opName) {
    if (node.inputs().size() != 1) {
        throw std::runtime_error(std::string(opName) +
                                 " node must have exactly 1 input");
    }
    if (node.outputs().size() != 1) {
        throw std::runtime_error(std::string(opName) +
                                 " node must have exactly 1 output");
    }
}

void checkBinaryNodeShape(const Node &node, const char *opName) {
    if (node.inputs().size() != 2) {
        throw std::runtime_error(std::string(opName) +
                                 " node must have exactly 2 inputs");
    }
    if (node.outputs().size() != 1) {
        throw std::runtime_error(std::string(opName) +
                                 " node must have exactly 1 output");
    }
}

} // namespace

Codegen::Codegen(mlir::MLIRContext &context) : context_(context) {}

mlir::MLIRContext &Codegen::getContext() noexcept { return context_; }

const mlir::MLIRContext &Codegen::getContext() const noexcept {
    return context_;
}

mlir::OwningOpRef<mlir::ModuleOp> Codegen::generate(const Graph &graph) {
    mlir::OpBuilder builder(&context_);
    mlir::Location loc = builder.getUnknownLoc();
    mlir::ModuleOp module = mlir::ModuleOp::create(loc);

    std::string funcName = graph.name().empty() ? "main" : graph.name();

    std::vector<mlir::Type> inputTypes  = buildInputTypes(graph);
    std::vector<mlir::Type> resultTypes = buildResultTypes(graph);

    mlir::FunctionType funcType = builder.getFunctionType(inputTypes, resultTypes);
    auto func = mlir::func::FuncOp::create(loc, funcName, funcType);

    mlir::Block *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    std::unordered_map<std::string, mlir::Value> values;

    bindFunctionInputs(graph, entryBlock, values);
    genConstants(builder, loc, graph, values);
    genNodes(builder, loc, graph, values);

    auto returnValues = collectReturnValues(graph, values);
    builder.create<mlir::func::ReturnOp>(loc, returnValues);

    module.push_back(func);
    return module;
}

mlir::Type Codegen::convertElementType(int onnx_type) const {
    switch (onnx_type) {
    case onnx::TensorProto_DataType_FLOAT:
        return mlir::Float32Type::get(&context_);

    case onnx::TensorProto_DataType_DOUBLE:
        return mlir::Float64Type::get(&context_);

    case onnx::TensorProto_DataType_INT64:
        return mlir::IntegerType::get(&context_, 64);

    case onnx::TensorProto_DataType_INT32:
        return mlir::IntegerType::get(&context_, 32);

    default:
        throw std::runtime_error("unsupported ONNX tensor element type");
    }
}

mlir::RankedTensorType
Codegen::convertTensorType(const Tensor &tensor) const {
    const auto &shape = tensor.shape();
    auto elem_type = convertElementType(tensor.type());
    return mlir::RankedTensorType::get(shape, elem_type);
}

std::vector<mlir::Type> Codegen::buildInputTypes(const Graph &graph) const {
    std::vector<mlir::Type> inputTypes;
    inputTypes.reserve(graph.inputs().size());

    for (const auto &inputName : graph.inputs()) {
        const Tensor *tensor = graph.tensor(inputName);
        if (!tensor) {
            throw std::runtime_error("graph input tensor not found: " + inputName);
        }
        inputTypes.push_back(convertTensorType(*tensor));
    }

    return inputTypes;
}

std::vector<mlir::Type> Codegen::buildResultTypes(const Graph &graph) const {
    std::vector<mlir::Type> resultTypes;
    resultTypes.reserve(graph.outputs().size());

    for (const auto &outputName : graph.outputs()) {
        const Tensor *tensor = graph.tensor(outputName);
        if (!tensor) {
            throw std::runtime_error("graph output tensor not found: " + outputName);
        }
        resultTypes.push_back(convertTensorType(*tensor));
    }

    return resultTypes;
}

void Codegen::bindFunctionInputs(
    const Graph &graph,
    mlir::Block *entry_block,
    std::unordered_map<std::string, mlir::Value> &values) const {

    for (size_t i = 0; i < graph.inputs().size(); ++i) {
        values[graph.inputs()[i]] = entry_block->getArgument(i);
    }
}

std::vector<mlir::Value> Codegen::collectReturnValues(
    const Graph &graph,
    const std::unordered_map<std::string, mlir::Value> &values) const {

    std::vector<mlir::Value> returnValues;
    returnValues.reserve(graph.outputs().size());

    for (const auto &outputName : graph.outputs()) {
        auto it = values.find(outputName);
        if (it == values.end()) {
            throw std::runtime_error("no MLIR value bound for output: " + outputName);
        }
        returnValues.push_back(it->second);
    }

    return returnValues;
}

void Codegen::genNodes(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const Graph &graph,
    std::unordered_map<std::string, mlir::Value> &values) const {

    for (const auto &node : graph.nodes()) {
        genNode(builder, loc, node, values);
    }
}

void Codegen::genNode(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const Node &node,
    std::unordered_map<std::string, mlir::Value> &values) const {

    const std::string &opcode = node.opcode();

    if (opcode == "Mul") {
        genMulNode(builder, loc, node, values);
        return;
    }

    if (opcode == "Add") {
        genAddNode(builder, loc, node, values);
        return;
    }

    if (opcode == "Identity") {
        genIdentityNode(builder, loc, node, values);
        return;
    }

    if (opcode == "Sub") {
        genSubNode(builder, loc, node, values);
        return;
    }

    if (opcode == "Div") {
        genDivNode(builder, loc, node, values);
        return;
    }

    if (opcode == "Relu") {
        genReluNode(builder, loc, node, values);
        return;
    }

    throw std::runtime_error("unsupported opcode: " + opcode);
}

void Codegen::genMulNode(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const Node &node,
    std::unordered_map<std::string, mlir::Value> &values) const {

    checkBinaryNodeShape(node, "Mul");

    const std::string &lhsName = node.inputs()[0];
    const std::string &rhsName = node.inputs()[1];
    const std::string &outName = node.outputs()[0];

    mlir::Value lhs = getBoundValue(values, lhsName, "Mul");
    mlir::Value rhs = getBoundValue(values, rhsName, "Mul");

    auto mulOp = builder.create<mlir::arith::MulFOp>(loc, lhs, rhs);
    values[outName] = mulOp.getResult();
}

void Codegen::genAddNode(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const Node &node,
    std::unordered_map<std::string, mlir::Value> &values) const {

    checkBinaryNodeShape(node, "Add");

    const std::string &lhsName = node.inputs()[0];
    const std::string &rhsName = node.inputs()[1];
    const std::string &outName = node.outputs()[0];

    mlir::Value lhs = getBoundValue(values, lhsName, "Add");
    mlir::Value rhs = getBoundValue(values, rhsName, "Add");

    auto addOp = builder.create<mlir::arith::AddFOp>(loc, lhs, rhs);
    values[outName] = addOp.getResult();
}

mlir::Value Codegen::genConstantTensor(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const Tensor &tensor) const {

    auto type = convertTensorType(tensor);

    if (tensor.type() != onnx::TensorProto_DataType_FLOAT) {
        throw std::runtime_error("only FLOAT constant tensors are supported for now");
    }

    const auto &raw = tensor.data();
    size_t elementCount = 1;
    for (int64_t d : tensor.shape()) {
        elementCount *= static_cast<size_t>(d);
    }

    if (raw.size() != elementCount * sizeof(float)) {
        throw std::runtime_error("constant tensor raw data size does not match shape");
    }

    std::vector<float> data(elementCount);
    std::memcpy(data.data(), raw.data(), raw.size());

    auto attr = mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(data));
    auto cst = builder.create<mlir::arith::ConstantOp>(loc, type, attr);
    return cst.getResult();
}

void Codegen::genConstants(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const Graph &graph,
    std::unordered_map<std::string, mlir::Value> &values) const {

    for (const auto &[name, tensor] : graph.tensors()) {
        if (!tensor.isConstant()) {
            continue;
        }

        values[name] = genConstantTensor(builder, loc, tensor);
    }
}

void Codegen::genIdentityNode(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const Node &node,
    std::unordered_map<std::string, mlir::Value> &values) const {

    (void)builder;
    (void)loc;

    checkUnaryNodeShape(node, "Identity");

    const std::string &inName = node.inputs()[0];
    const std::string &outName = node.outputs()[0];

    values[outName] = getBoundValue(values, inName, "Identity");
}

void Codegen::genSubNode(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const Node &node,
    std::unordered_map<std::string, mlir::Value> &values) const {

    checkBinaryNodeShape(node, "Sub");

    const std::string &lhsName = node.inputs()[0];
    const std::string &rhsName = node.inputs()[1];
    const std::string &outName = node.outputs()[0];

    mlir::Value lhs = getBoundValue(values, lhsName, "Sub");
    mlir::Value rhs = getBoundValue(values, rhsName, "Sub");

    auto subOp = builder.create<mlir::arith::SubFOp>(loc, lhs, rhs);
    values[outName] = subOp.getResult();
}

void Codegen::genDivNode(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const Node &node,
    std::unordered_map<std::string, mlir::Value> &values) const {

    checkBinaryNodeShape(node, "Div");

    const std::string &lhsName = node.inputs()[0];
    const std::string &rhsName = node.inputs()[1];
    const std::string &outName = node.outputs()[0];

    mlir::Value lhs = getBoundValue(values, lhsName, "Div");
    mlir::Value rhs = getBoundValue(values, rhsName, "Div");

    auto divOp = builder.create<mlir::arith::DivFOp>(loc, lhs, rhs);
    values[outName] = divOp.getResult();
}

void Codegen::genReluNode(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const Node &node,
    std::unordered_map<std::string, mlir::Value> &values) const {

    checkUnaryNodeShape(node, "Relu");

    const std::string &inName = node.inputs()[0];
    const std::string &outName = node.outputs()[0];

    mlir::Value input = getBoundValue(values, inName, "Relu");
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());

    if (!type) {
        throw std::runtime_error("Relu currently expects RankedTensorType input");
    }

    if (!type.getElementType().isF32()) {
        throw std::runtime_error("Relu currently supports only f32 tensors");
    }

    size_t elementCount = 1;
    for (int64_t d : type.getShape()) {
        elementCount *= static_cast<size_t>(d);
    }

    std::vector<float> zeros(elementCount, 0.0f);
    auto zeroAttr = mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(zeros));
    auto zero = builder.create<mlir::arith::ConstantOp>(loc, type, zeroAttr);

    auto maxOp = builder.create<mlir::arith::MaximumFOp>(loc, input, zero.getResult());
    values[outName] = maxOp.getResult();
}

} // namespace tensor_compiler
