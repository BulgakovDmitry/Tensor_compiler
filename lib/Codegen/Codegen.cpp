#include <unordered_map>
#include <vector>
#include <string>
#include "Codegen/Codegen.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

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

const Attribute::AttrValue *getAttributeValue(const Node &node,
                                              const std::string &name) {
    auto it = node.attributes().find(name);
    if (it == node.attributes().end()) {
        return nullptr;
    }
    return &it->second.value();
}

int64_t getIntAttribute(const Node &node, const std::string &name,
                        int64_t defaultValue) {
    const Attribute::AttrValue *value = getAttributeValue(node, name);
    if (!value) {
        return defaultValue;
    }

    const auto *intValue = std::get_if<int64_t>(value);
    if (!intValue) {
        throw std::runtime_error("attribute '" + name + "' must be an integer");
    }
    return *intValue;
}

std::vector<int64_t> getIntVectorAttribute(
    const Node &node, const std::string &name,
    std::vector<int64_t> defaultValue) {
    const Attribute::AttrValue *value = getAttributeValue(node, name);
    if (!value) {
        return defaultValue;
    }

    const auto *vectorValue = std::get_if<std::vector<int64_t>>(value);
    if (!vectorValue) {
        throw std::runtime_error("attribute '" + name +
                                 "' must be an integer vector");
    }
    return *vectorValue;
}

void requireSize(const std::vector<int64_t> &values, size_t size,
                 const std::string &name) {
    if (values.size() != size) {
        throw std::runtime_error("Conv attribute '" + name +
                                 "' has unexpected rank");
    }
}

bool hasPadding(const std::vector<int64_t> &pads) {
    for (int64_t pad : pads) {
        if (pad != 0) {
            return true;
        }
    }
    return false;
}

int64_t checkedPositiveDim(int64_t dim, const std::string &name) {
    if (dim <= 0) {
        throw std::runtime_error("Conv requires static positive " + name);
    }
    return dim;
}

mlir::DenseIntElementsAttr getI64VectorAttr(mlir::OpBuilder &builder,
                                            llvm::ArrayRef<int64_t> values) {
    auto type = mlir::RankedTensorType::get(
        {static_cast<int64_t>(values.size())}, builder.getI64Type());
    return mlir::DenseIntElementsAttr::get(type, values);
}

std::vector<mlir::Value> collectDynamicDims(mlir::OpBuilder &builder,
                                            mlir::Location loc,
                                            mlir::Value source,
                                            mlir::RankedTensorType type) {
    std::vector<mlir::Value> dynamicDims;
    for (auto [idx, dim] : llvm::enumerate(type.getShape())) {
        if (mlir::ShapedType::isDynamic(dim)) {
            dynamicDims.push_back(
                builder.create<mlir::tensor::DimOp>(loc, source, idx));
        }
    }
    return dynamicDims;
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

    constexpr const char* ENTRY_FUNC_NAME = "tensorCompForwardImpl";

    std::string funcName = ENTRY_FUNC_NAME;

    std::vector<mlir::Type> inputTypes  = buildInputTypes(graph);
    std::vector<mlir::Type> resultTypes = buildResultTypes(graph);

    mlir::FunctionType funcType = builder.getFunctionType(inputTypes, resultTypes);
    auto func = mlir::func::FuncOp::create(loc, funcName, funcType);
    func.setPublic();

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
        genNode(builder, loc, node, graph, values);
    }
}

void Codegen::genNode(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const Node &node,
    const Graph &graph,
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

    if (opcode == "Conv") {
        genConvNode(builder, loc, graph, node, values);
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

void Codegen::genConvNode(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const Graph &graph,
    const Node &node,
    std::unordered_map<std::string, mlir::Value> &values) const {

    if (node.inputs().size() < 2 || node.inputs().size() > 3) {
        throw std::runtime_error("Conv node must have 2 or 3 inputs");
    }
    if (node.outputs().size() != 1) {
        throw std::runtime_error("Conv node must have exactly 1 output");
    }
    if (node.inputs().size() == 3) {
        throw std::runtime_error("Conv bias input is not supported yet");
    }

    const std::string &inputName = node.inputs()[0];
    const std::string &filterName = node.inputs()[1];
    const std::string &outName = node.outputs()[0];

    mlir::Value input = getBoundValue(values, inputName, "Conv");
    mlir::Value filter = getBoundValue(values, filterName, "Conv");

    auto inputType =
        mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    auto filterType =
        mlir::dyn_cast<mlir::RankedTensorType>(filter.getType());

    if (!inputType || !filterType) {
        throw std::runtime_error("Conv expects ranked tensor operands");
    }
    if (inputType.getRank() != 4 || filterType.getRank() != 4) {
        throw std::runtime_error("Conv currently supports only 2D NCHW/FCHW");
    }
    if (!inputType.getElementType().isF32() ||
        !filterType.getElementType().isF32()) {
        throw std::runtime_error("Conv currently supports only f32 tensors");
    }

    const Tensor *filterTensor = graph.tensor(filterName);
    if (!filterTensor) {
        throw std::runtime_error("Conv filter tensor metadata not found: " +
                                 filterName);
    }

    int64_t group = getIntAttribute(node, "group", 1);
    if (group != 1) {
        throw std::runtime_error("Conv group != 1 is not supported yet");
    }

    auto strides = getIntVectorAttribute(node, "strides", {1, 1});
    auto dilations = getIntVectorAttribute(node, "dilations", {1, 1});
    auto pads = getIntVectorAttribute(node, "pads", {0, 0, 0, 0});
    requireSize(strides, 2, "strides");
    requireSize(dilations, 2, "dilations");
    requireSize(pads, 4, "pads");

    const auto inputShape = inputType.getShape();
    const auto filterShape = filterType.getShape();

    int64_t channels = checkedPositiveDim(inputShape[1], "input channels");
    int64_t inputH = checkedPositiveDim(inputShape[2], "input height");
    int64_t inputW = checkedPositiveDim(inputShape[3], "input width");
    int64_t filters = checkedPositiveDim(filterShape[0], "filter count");
    int64_t filterChannels =
        checkedPositiveDim(filterShape[1], "filter channels");
    int64_t kernelH = checkedPositiveDim(filterShape[2], "kernel height");
    int64_t kernelW = checkedPositiveDim(filterShape[3], "kernel width");

    if (channels != filterChannels) {
        throw std::runtime_error("Conv input/filter channel mismatch");
    }

    int64_t strideH = checkedPositiveDim(strides[0], "stride height");
    int64_t strideW = checkedPositiveDim(strides[1], "stride width");
    int64_t dilationH = checkedPositiveDim(dilations[0], "dilation height");
    int64_t dilationW = checkedPositiveDim(dilations[1], "dilation width");

    int64_t paddedH = inputH + pads[0] + pads[2];
    int64_t paddedW = inputW + pads[1] + pads[3];
    int64_t effectiveKernelH = dilationH * (kernelH - 1) + 1;
    int64_t effectiveKernelW = dilationW * (kernelW - 1) + 1;
    int64_t outH = (paddedH - effectiveKernelH) / strideH + 1;
    int64_t outW = (paddedW - effectiveKernelW) / strideW + 1;

    if (outH <= 0 || outW <= 0) {
        throw std::runtime_error("Conv computed non-positive output shape");
    }

    mlir::Value convInput = input;
    if (hasPadding(pads)) {
        std::vector<int64_t> paddedShape(inputShape.begin(), inputShape.end());
        paddedShape[2] = paddedH;
        paddedShape[3] = paddedW;
        auto paddedType = mlir::RankedTensorType::get(
            paddedShape, inputType.getElementType());

        auto zero = builder.create<mlir::arith::ConstantOp>(
            loc, builder.getF32FloatAttr(0.0f));
        std::vector<mlir::OpFoldResult> low = {
            builder.getIndexAttr(0),
            builder.getIndexAttr(0),
            builder.getIndexAttr(pads[0]),
            builder.getIndexAttr(pads[1])};
        std::vector<mlir::OpFoldResult> high = {
            builder.getIndexAttr(0),
            builder.getIndexAttr(0),
            builder.getIndexAttr(pads[2]),
            builder.getIndexAttr(pads[3])};
        auto padOp = builder.create<mlir::tensor::PadOp>(
            loc,
            paddedType,
            input,
            low,
            high,
            zero.getResult(),
            false);
        convInput = padOp.getResult();
    }

    std::vector<int64_t> outShape = {
        inputShape[0], filters, outH, outW};
    auto outType = mlir::RankedTensorType::get(
        outShape, inputType.getElementType());

    std::vector<mlir::Value> dynamicDims =
        collectDynamicDims(builder, loc, input, outType);
    auto empty = builder.create<mlir::tensor::EmptyOp>(
        loc, outType, dynamicDims);
    auto zero = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getF32FloatAttr(0.0f));
    auto filled = builder.create<mlir::linalg::FillOp>(
        loc,
        mlir::TypeRange{outType},
        mlir::ValueRange{zero.getResult()},
        mlir::ValueRange{empty.getResult()});

    auto conv = builder.create<mlir::linalg::Conv2DNchwFchwOp>(
        loc,
        mlir::TypeRange{outType},
        mlir::ValueRange{convInput, filter},
        mlir::ValueRange{filled.getResult(0)},
        getI64VectorAttr(builder, strides),
        getI64VectorAttr(builder, dilations));

    values[outName] = conv.getResult(0);
}

} // namespace tensor_compiler
