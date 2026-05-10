#include <cstring>
#include <unordered_map>
#include <vector>
#include <string>
#include <limits>
#include "Codegen/Codegen.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
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

float getFloatAttribute(const Node &node, const std::string &name,
                        float defaultValue) {
    const Attribute::AttrValue *value = getAttributeValue(node, name);
    if (!value) {
        return defaultValue;
    }

    const auto *floatValue = std::get_if<float>(value);
    if (!floatValue) {
        throw std::runtime_error("attribute '" + name + "' must be a float");
    }
    return *floatValue;
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

std::vector<int64_t> readI64TensorData(const Tensor &tensor,
                                       const char *opName) {
    if (tensor.type() != onnx::TensorProto_DataType_INT64) {
        throw std::runtime_error(std::string(opName) +
                                 " axes input must be an INT64 constant tensor");
    }

    size_t elementCount = 1;
    for (int64_t d : tensor.shape()) {
        if (d < 0) {
            throw std::runtime_error(std::string(opName) +
                                     " axes tensor must have static shape");
        }
        elementCount *= static_cast<size_t>(d);
    }

    const auto &raw = tensor.data();
    if (raw.size() != elementCount * sizeof(int64_t)) {
        throw std::runtime_error(std::string(opName) +
                                 " axes raw data size does not match shape");
    }

    std::vector<int64_t> data(elementCount);
    std::memcpy(data.data(), raw.data(), raw.size());
    return data;
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

void checkBatchNormParamType(mlir::Value value, int64_t channels,
                             const char *name) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    if (!type || type.getRank() != 1 || !type.getElementType().isF32()) {
        throw std::runtime_error(std::string("BatchNormalization ") + name +
                                 " must be a rank-1 f32 tensor");
    }
    if (!mlir::ShapedType::isDynamic(type.getShape()[0]) &&
        type.getShape()[0] != channels) {
        throw std::runtime_error(std::string("BatchNormalization ") + name +
                                 " channel count mismatch");
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

    if (opcode == "BatchNormalization") {
        genBatchNormalizationNode(builder, loc, node, values);
        return;
    }

    if (opcode == "MaxPool") {
        genMaxPoolNode(builder, loc, node, values);
        return;
    }

    if (opcode == "ReduceMean") {
        genReduceMeanNode(builder, loc, node, values);
        return;
    }

    if (opcode == "Reshape") {
        genReshapeNode(builder, loc, node, values);
        return;
    }

    if (opcode == "Squeeze") {
        genSqueezeNode(builder, loc, graph, node, values);
        return;
    }

    if (opcode == "MatMul") {
        genMatMulNode(builder, loc, node, values);
        return;
    }

    if (opcode == "Softmax") {
        genSoftmaxNode(builder, loc, node, values);
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

    const auto &raw = tensor.data();
    size_t elementCount = 1;
    for (int64_t d : tensor.shape()) {
        elementCount *= static_cast<size_t>(d);
    }

    if (tensor.type() == onnx::TensorProto_DataType_FLOAT) {
        if (raw.size() != elementCount * sizeof(float)) {
            throw std::runtime_error(
                "constant tensor raw data size does not match shape");
        }

        std::vector<float> data(elementCount);
        std::memcpy(data.data(), raw.data(), raw.size());

        auto attr =
            mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(data));
        auto cst = builder.create<mlir::arith::ConstantOp>(loc, type, attr);
        return cst.getResult();
    }

    if (tensor.type() == onnx::TensorProto_DataType_INT64) {
        if (raw.size() != elementCount * sizeof(int64_t)) {
            throw std::runtime_error(
                "constant tensor raw data size does not match shape");
        }

        std::vector<int64_t> data(elementCount);
        std::memcpy(data.data(), raw.data(), raw.size());

        auto attr = mlir::DenseIntElementsAttr::get(
            mlir::cast<mlir::ShapedType>(type), llvm::ArrayRef<int64_t>(data));
        auto cst = builder.create<mlir::arith::ConstantOp>(loc, type, attr);
        return cst.getResult();
    }

    throw std::runtime_error("unsupported constant tensor element type");
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

    std::vector<mlir::Value> dynamicDims =
        collectDynamicDims(builder, loc, input, type);
    auto empty = builder.create<mlir::tensor::EmptyOp>(
        loc, type, dynamicDims);

    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.reserve(static_cast<size_t>(type.getRank()));
    for (int64_t i = 0; i < type.getRank(); ++i) {
        exprs.push_back(builder.getAffineDimExpr(i));
    }
    auto map = mlir::AffineMap::get(type.getRank(), 0, exprs,
                                    builder.getContext());

    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
        static_cast<size_t>(type.getRank()),
        mlir::utils::IteratorType::parallel);

    auto relu = builder.create<mlir::linalg::GenericOp>(
        loc,
        mlir::TypeRange{type},
        mlir::ValueRange{input},
        mlir::ValueRange{empty.getResult()},
        llvm::ArrayRef<mlir::AffineMap>{map, map},
        iteratorTypes,
        [](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
           mlir::ValueRange args) {
            auto zero = nestedBuilder.create<mlir::arith::ConstantOp>(
                nestedLoc, nestedBuilder.getF32FloatAttr(0.0f));
            auto max = nestedBuilder.create<mlir::arith::MaximumFOp>(
                nestedLoc, args[0], zero.getResult());
            nestedBuilder.create<mlir::linalg::YieldOp>(
                nestedLoc, max.getResult());
        });

    values[outName] = relu.getResult(0);
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
    const std::string &inputName = node.inputs()[0];
    const std::string &filterName = node.inputs()[1];
    const std::string &outName = node.outputs()[0];
    bool hasBias = node.inputs().size() == 3 && !node.inputs()[2].empty();

    mlir::Value input = getBoundValue(values, inputName, "Conv");
    mlir::Value filter = getBoundValue(values, filterName, "Conv");
    mlir::Value bias;
    if (hasBias) {
        bias = getBoundValue(values, node.inputs()[2], "Conv");
    }

    auto inputType =
        mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    auto filterType =
        mlir::dyn_cast<mlir::RankedTensorType>(filter.getType());
    auto biasType =
        hasBias ? mlir::dyn_cast<mlir::RankedTensorType>(bias.getType())
                : mlir::RankedTensorType{};

    if (!inputType || !filterType) {
        throw std::runtime_error("Conv expects ranked tensor operands");
    }
    if (hasBias && !biasType) {
        throw std::runtime_error("Conv expects ranked tensor bias");
    }
    if (inputType.getRank() != 4 || filterType.getRank() != 4) {
        throw std::runtime_error("Conv currently supports only 2D NCHW/FCHW");
    }
    if (hasBias && biasType.getRank() != 1) {
        throw std::runtime_error("Conv bias must be a rank-1 tensor");
    }
    if (!inputType.getElementType().isF32() ||
        !filterType.getElementType().isF32() ||
        (hasBias && !biasType.getElementType().isF32())) {
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
    if (hasBias && !mlir::ShapedType::isDynamic(biasType.getShape()[0]) &&
        biasType.getShape()[0] != filters) {
        throw std::runtime_error("Conv bias channel count mismatch");
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

    mlir::Value init;
    if (hasBias) {
        auto *ctx = builder.getContext();
        auto n = builder.getAffineDimExpr(0);
        auto c = builder.getAffineDimExpr(1);
        auto h = builder.getAffineDimExpr(2);
        auto w = builder.getAffineDimExpr(3);
        auto channelMap = mlir::AffineMap::get(4, 0, {c}, ctx);
        auto outputMap = mlir::AffineMap::get(4, 0, {n, c, h, w}, ctx);
        llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
            4, mlir::utils::IteratorType::parallel);

        auto biasFill = builder.create<mlir::linalg::GenericOp>(
            loc,
            mlir::TypeRange{outType},
            mlir::ValueRange{bias},
            mlir::ValueRange{empty.getResult()},
            llvm::ArrayRef<mlir::AffineMap>{channelMap, outputMap},
            iteratorTypes,
            [](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
               mlir::ValueRange args) {
                nestedBuilder.create<mlir::linalg::YieldOp>(
                    nestedLoc, args[0]);
            });
        init = biasFill.getResult(0);
    } else {
        auto zero = builder.create<mlir::arith::ConstantOp>(
            loc, builder.getF32FloatAttr(0.0f));
        auto filled = builder.create<mlir::linalg::FillOp>(
            loc,
            mlir::TypeRange{outType},
            mlir::ValueRange{zero.getResult()},
            mlir::ValueRange{empty.getResult()});
        init = filled.getResult(0);
    }

    auto conv = builder.create<mlir::linalg::Conv2DNchwFchwOp>(
        loc,
        mlir::TypeRange{outType},
        mlir::ValueRange{convInput, filter},
        mlir::ValueRange{init},
        getI64VectorAttr(builder, strides),
        getI64VectorAttr(builder, dilations));

    values[outName] = conv.getResult(0);
}

void Codegen::genBatchNormalizationNode(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const Node &node,
    std::unordered_map<std::string, mlir::Value> &values) const {

    if (node.inputs().size() != 5) {
        throw std::runtime_error(
            "BatchNormalization node must have exactly 5 inputs");
    }
    if (node.outputs().empty()) {
        throw std::runtime_error(
            "BatchNormalization node must have at least 1 output");
    }
    if (getIntAttribute(node, "training_mode", 0) != 0) {
        throw std::runtime_error(
            "BatchNormalization training_mode is not supported");
    }

    mlir::Value input = getBoundValue(values, node.inputs()[0],
                                      "BatchNormalization");
    mlir::Value scale = getBoundValue(values, node.inputs()[1],
                                      "BatchNormalization");
    mlir::Value bias = getBoundValue(values, node.inputs()[2],
                                     "BatchNormalization");
    mlir::Value mean = getBoundValue(values, node.inputs()[3],
                                     "BatchNormalization");
    mlir::Value var = getBoundValue(values, node.inputs()[4],
                                    "BatchNormalization");

    auto inputType =
        mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType || inputType.getRank() != 4 ||
        !inputType.getElementType().isF32()) {
        throw std::runtime_error(
            "BatchNormalization supports only rank-4 f32 NCHW input");
    }

    int64_t channels =
        checkedPositiveDim(inputType.getShape()[1], "BatchNormalization channels");
    checkBatchNormParamType(scale, channels, "scale");
    checkBatchNormParamType(bias, channels, "bias");
    checkBatchNormParamType(mean, channels, "mean");
    checkBatchNormParamType(var, channels, "var");

    std::vector<mlir::Value> dynamicDims =
        collectDynamicDims(builder, loc, input, inputType);
    auto empty = builder.create<mlir::tensor::EmptyOp>(
        loc, inputType, dynamicDims);

    auto *ctx = builder.getContext();
    auto n = builder.getAffineDimExpr(0);
    auto c = builder.getAffineDimExpr(1);
    auto h = builder.getAffineDimExpr(2);
    auto w = builder.getAffineDimExpr(3);
    auto tensorMap = mlir::AffineMap::get(4, 0, {n, c, h, w}, ctx);
    auto channelMap = mlir::AffineMap::get(4, 0, {c}, ctx);

    llvm::SmallVector<mlir::AffineMap> indexingMaps = {
        tensorMap, channelMap, channelMap, channelMap, channelMap, tensorMap};
    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
        4, mlir::utils::IteratorType::parallel);

    float epsilon = getFloatAttribute(node, "epsilon", 1.0e-5f);
    auto generic = builder.create<mlir::linalg::GenericOp>(
        loc,
        mlir::TypeRange{inputType},
        mlir::ValueRange{input, scale, bias, mean, var},
        mlir::ValueRange{empty.getResult()},
        indexingMaps,
        iteratorTypes,
        [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
            mlir::ValueRange args) {
            mlir::Value x = args[0];
            mlir::Value scaleValue = args[1];
            mlir::Value biasValue = args[2];
            mlir::Value meanValue = args[3];
            mlir::Value varValue = args[4];

            auto eps = nestedBuilder.create<mlir::arith::ConstantOp>(
                nestedLoc, nestedBuilder.getF32FloatAttr(epsilon));
            auto centered = nestedBuilder.create<mlir::arith::SubFOp>(
                nestedLoc, x, meanValue);
            auto varWithEps = nestedBuilder.create<mlir::arith::AddFOp>(
                nestedLoc, varValue, eps.getResult());
            auto denom = nestedBuilder.create<mlir::math::SqrtOp>(
                nestedLoc, varWithEps.getResult());
            auto normalized = nestedBuilder.create<mlir::arith::DivFOp>(
                nestedLoc, centered.getResult(), denom.getResult());
            auto scaled = nestedBuilder.create<mlir::arith::MulFOp>(
                nestedLoc, normalized.getResult(), scaleValue);
            auto shifted = nestedBuilder.create<mlir::arith::AddFOp>(
                nestedLoc, scaled.getResult(), biasValue);
            nestedBuilder.create<mlir::linalg::YieldOp>(
                nestedLoc, shifted.getResult());
        });

    values[node.outputs()[0]] = generic.getResult(0);
}

void Codegen::genMaxPoolNode(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const Node &node,
    std::unordered_map<std::string, mlir::Value> &values) const {

    if (node.inputs().size() != 1) {
        throw std::runtime_error("MaxPool node must have exactly 1 input");
    }
    if (node.outputs().empty()) {
        throw std::runtime_error("MaxPool node must have at least 1 output");
    }
    if (node.outputs().size() > 1) {
        throw std::runtime_error("MaxPool indices output is not supported yet");
    }
    if (getIntAttribute(node, "ceil_mode", 0) != 0) {
        throw std::runtime_error("MaxPool ceil_mode is not supported yet");
    }

    mlir::Value input = getBoundValue(values, node.inputs()[0], "MaxPool");
    auto inputType =
        mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType || inputType.getRank() != 4 ||
        !inputType.getElementType().isF32()) {
        throw std::runtime_error("MaxPool supports only rank-4 f32 NCHW input");
    }

    auto kernel = getIntVectorAttribute(node, "kernel_shape", {});
    auto strides = getIntVectorAttribute(node, "strides", {1, 1});
    auto pads = getIntVectorAttribute(node, "pads", {0, 0, 0, 0});
    auto dilations = getIntVectorAttribute(node, "dilations", {1, 1});
    requireSize(kernel, 2, "kernel_shape");
    requireSize(strides, 2, "strides");
    requireSize(pads, 4, "pads");
    requireSize(dilations, 2, "dilations");

    int64_t kernelH = checkedPositiveDim(kernel[0], "MaxPool kernel height");
    int64_t kernelW = checkedPositiveDim(kernel[1], "MaxPool kernel width");
    int64_t strideH = checkedPositiveDim(strides[0], "MaxPool stride height");
    int64_t strideW = checkedPositiveDim(strides[1], "MaxPool stride width");
    int64_t dilationH =
        checkedPositiveDim(dilations[0], "MaxPool dilation height");
    int64_t dilationW =
        checkedPositiveDim(dilations[1], "MaxPool dilation width");

    const auto inputShape = inputType.getShape();
    int64_t channels = checkedPositiveDim(inputShape[1], "MaxPool channels");
    int64_t inputH = checkedPositiveDim(inputShape[2], "MaxPool input height");
    int64_t inputW = checkedPositiveDim(inputShape[3], "MaxPool input width");

    int64_t paddedH = inputH + pads[0] + pads[2];
    int64_t paddedW = inputW + pads[1] + pads[3];
    int64_t effectiveKernelH = dilationH * (kernelH - 1) + 1;
    int64_t effectiveKernelW = dilationW * (kernelW - 1) + 1;
    int64_t outH = (paddedH - effectiveKernelH) / strideH + 1;
    int64_t outW = (paddedW - effectiveKernelW) / strideW + 1;
    if (outH <= 0 || outW <= 0) {
        throw std::runtime_error("MaxPool computed non-positive output shape");
    }

    mlir::Value poolInput = input;
    if (hasPadding(pads)) {
        std::vector<int64_t> paddedShape(inputShape.begin(), inputShape.end());
        paddedShape[2] = paddedH;
        paddedShape[3] = paddedW;
        auto paddedType = mlir::RankedTensorType::get(
            paddedShape, inputType.getElementType());

        auto negInf = builder.create<mlir::arith::ConstantOp>(
            loc, builder.getF32FloatAttr(
                     -std::numeric_limits<float>::infinity()));
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
            negInf.getResult(),
            false);
        poolInput = padOp.getResult();
    }

    std::vector<int64_t> outShape = {
        inputShape[0], channels, outH, outW};
    auto outType = mlir::RankedTensorType::get(
        outShape, inputType.getElementType());

    std::vector<mlir::Value> dynamicDims =
        collectDynamicDims(builder, loc, input, outType);
    auto window = builder.create<mlir::tensor::EmptyOp>(
        loc,
        mlir::RankedTensorType::get({kernelH, kernelW},
                                    inputType.getElementType()),
        mlir::ValueRange{});
    auto empty = builder.create<mlir::tensor::EmptyOp>(
        loc, outType, dynamicDims);
    auto negInf = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
    auto filled = builder.create<mlir::linalg::FillOp>(
        loc,
        mlir::TypeRange{outType},
        mlir::ValueRange{negInf.getResult()},
        mlir::ValueRange{empty.getResult()});

    auto pool = builder.create<mlir::linalg::PoolingNchwMaxOp>(
        loc,
        mlir::TypeRange{outType},
        mlir::ValueRange{poolInput, window.getResult()},
        mlir::ValueRange{filled.getResult(0)},
        getI64VectorAttr(builder, strides),
        getI64VectorAttr(builder, dilations));

    values[node.outputs()[0]] = pool.getResult(0);
}

void Codegen::genReduceMeanNode(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const Node &node,
    std::unordered_map<std::string, mlir::Value> &values) const {

    checkUnaryNodeShape(node, "ReduceMean");

    mlir::Value input = getBoundValue(values, node.inputs()[0], "ReduceMean");
    auto inputType =
        mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType || inputType.getRank() != 4 ||
        !inputType.getElementType().isF32()) {
        throw std::runtime_error(
            "ReduceMean supports only rank-4 f32 NCHW input");
    }

    if (getIntAttribute(node, "keepdims", 1) != 1) {
        throw std::runtime_error(
            "ReduceMean keepdims=0 is not supported yet");
    }

    auto axes = getIntVectorAttribute(node, "axes", {});
    if (axes.empty()) {
        axes = {0, 1, 2, 3};
    }
    for (int64_t &axis : axes) {
        if (axis < 0) {
            axis += inputType.getRank();
        }
    }
    if (axes.size() != 2 || axes[0] != 2 || axes[1] != 3) {
        throw std::runtime_error(
            "ReduceMean currently supports only axes=[2,3]");
    }

    const auto inputShape = inputType.getShape();
    int64_t height = checkedPositiveDim(inputShape[2], "ReduceMean height");
    int64_t width = checkedPositiveDim(inputShape[3], "ReduceMean width");

    std::vector<int64_t> outShape(inputShape.begin(), inputShape.end());
    outShape[2] = 1;
    outShape[3] = 1;
    auto outType = mlir::RankedTensorType::get(
        outShape, inputType.getElementType());

    std::vector<mlir::Value> dynamicDims =
        collectDynamicDims(builder, loc, input, outType);
    auto sumEmpty = builder.create<mlir::tensor::EmptyOp>(
        loc, outType, dynamicDims);
    auto zero = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getF32FloatAttr(0.0f));
    auto zeroed = builder.create<mlir::linalg::FillOp>(
        loc,
        mlir::TypeRange{outType},
        mlir::ValueRange{zero.getResult()},
        mlir::ValueRange{sumEmpty.getResult()});

    auto *ctx = builder.getContext();
    auto n = builder.getAffineDimExpr(0);
    auto c = builder.getAffineDimExpr(1);
    auto h = builder.getAffineDimExpr(2);
    auto w = builder.getAffineDimExpr(3);
    auto inputMap = mlir::AffineMap::get(4, 0, {n, c, h, w}, ctx);
    auto outputMap = mlir::AffineMap::get(
        4, 0, {n, c, builder.getAffineConstantExpr(0),
               builder.getAffineConstantExpr(0)}, ctx);

    llvm::SmallVector<mlir::utils::IteratorType> reductionIterators = {
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::reduction,
        mlir::utils::IteratorType::reduction};

    auto sum = builder.create<mlir::linalg::GenericOp>(
        loc,
        mlir::TypeRange{outType},
        mlir::ValueRange{input},
        mlir::ValueRange{zeroed.getResult(0)},
        llvm::ArrayRef<mlir::AffineMap>{inputMap, outputMap},
        reductionIterators,
        [](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
           mlir::ValueRange args) {
            auto added = nestedBuilder.create<mlir::arith::AddFOp>(
                nestedLoc, args[0], args[1]);
            nestedBuilder.create<mlir::linalg::YieldOp>(
                nestedLoc, added.getResult());
        });

    auto meanEmpty = builder.create<mlir::tensor::EmptyOp>(
        loc, outType, dynamicDims);
    auto outMap = mlir::AffineMap::get(4, 0, {n, c, h, w}, ctx);
    llvm::SmallVector<mlir::utils::IteratorType> parallelIterators(
        4, mlir::utils::IteratorType::parallel);
    float scale = 1.0f / static_cast<float>(height * width);
    auto mean = builder.create<mlir::linalg::GenericOp>(
        loc,
        mlir::TypeRange{outType},
        mlir::ValueRange{sum.getResult(0)},
        mlir::ValueRange{meanEmpty.getResult()},
        llvm::ArrayRef<mlir::AffineMap>{outMap, outMap},
        parallelIterators,
        [scale](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
                mlir::ValueRange args) {
            auto scaleValue = nestedBuilder.create<mlir::arith::ConstantOp>(
                nestedLoc, nestedBuilder.getF32FloatAttr(scale));
            auto scaled = nestedBuilder.create<mlir::arith::MulFOp>(
                nestedLoc, args[0], scaleValue.getResult());
            nestedBuilder.create<mlir::linalg::YieldOp>(
                nestedLoc, scaled.getResult());
        });

    values[node.outputs()[0]] = mean.getResult(0);
}

void Codegen::genReshapeNode(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const Node &node,
    std::unordered_map<std::string, mlir::Value> &values) const {

    checkBinaryNodeShape(node, "Reshape");

    mlir::Value input = getBoundValue(values, node.inputs()[0], "Reshape");
    mlir::Value shape = getBoundValue(values, node.inputs()[1], "Reshape");

    auto inputType = mlir::dyn_cast<mlir::TensorType>(input.getType());
    auto shapeType =
        mlir::dyn_cast<mlir::RankedTensorType>(shape.getType());
    if (!inputType) {
        throw std::runtime_error("Reshape expects tensor input");
    }
    if (!shapeType || shapeType.getRank() != 1 ||
        (!shapeType.getElementType().isSignlessInteger() &&
         !shapeType.getElementType().isIndex())) {
        throw std::runtime_error(
            "Reshape shape input must be a rank-1 integer tensor");
    }

    mlir::TensorType resultType;
    int64_t resultRank = shapeType.getShape()[0];
    if (mlir::ShapedType::isDynamic(resultRank)) {
        resultType = mlir::UnrankedTensorType::get(inputType.getElementType());
    } else {
        if (resultRank < 0) {
            throw std::runtime_error("Reshape result rank must be non-negative");
        }
        std::vector<int64_t> resultShape(
            static_cast<size_t>(resultRank), mlir::ShapedType::kDynamic);
        resultType = mlir::RankedTensorType::get(
            resultShape, inputType.getElementType());
    }

    auto reshape = builder.create<mlir::tensor::ReshapeOp>(
        loc, resultType, input, shape);
    values[node.outputs()[0]] = reshape.getResult();
}

void Codegen::genSqueezeNode(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const Graph &graph,
    const Node &node,
    std::unordered_map<std::string, mlir::Value> &values) const {

    if (node.inputs().empty() || node.inputs().size() > 2) {
        throw std::runtime_error("Squeeze node must have 1 or 2 inputs");
    }
    if (node.outputs().size() != 1) {
        throw std::runtime_error("Squeeze node must have exactly 1 output");
    }

    mlir::Value input = getBoundValue(values, node.inputs()[0], "Squeeze");
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType) {
        throw std::runtime_error("Squeeze expects ranked tensor input");
    }

    const int64_t rank = inputType.getRank();
    std::vector<bool> squeezed(static_cast<size_t>(rank), false);

    std::vector<int64_t> axes;
    const Attribute::AttrValue *axesAttr = getAttributeValue(node, "axes");
    if (axesAttr) {
        const auto *axisVector = std::get_if<std::vector<int64_t>>(axesAttr);
        if (!axisVector) {
            throw std::runtime_error("Squeeze attribute 'axes' must be an integer vector");
        }
        axes = *axisVector;
    } else if (node.inputs().size() == 2) {
        const Tensor *axesTensor = graph.tensor(node.inputs()[1]);
        if (!axesTensor || !axesTensor->isConstant()) {
            throw std::runtime_error(
                "Squeeze axes input must be a constant initializer");
        }
        axes = readI64TensorData(*axesTensor, "Squeeze");
    }

    if (axes.empty() && !axesAttr && node.inputs().size() == 1) {
        for (int64_t i = 0; i < rank; ++i) {
            if (inputType.getShape()[i] == 1) {
                squeezed[static_cast<size_t>(i)] = true;
            } else if (mlir::ShapedType::isDynamic(inputType.getShape()[i])) {
                throw std::runtime_error(
                    "Squeeze without axes requires static input dimensions");
            }
        }
    } else {
        for (int64_t axis : axes) {
            if (axis < 0) {
                axis += rank;
            }
            if (axis < 0 || axis >= rank) {
                throw std::runtime_error("Squeeze axis is out of range");
            }
            if (squeezed[static_cast<size_t>(axis)]) {
                throw std::runtime_error("Squeeze axes must be unique");
            }

            int64_t dim = inputType.getShape()[axis];
            if (!mlir::ShapedType::isDynamic(dim) && dim != 1) {
                throw std::runtime_error(
                    "Squeeze can only remove dimensions of size 1");
            }
            squeezed[static_cast<size_t>(axis)] = true;
        }
    }

    std::vector<int64_t> resultShape;
    resultShape.reserve(static_cast<size_t>(rank));
    for (int64_t i = 0; i < rank; ++i) {
        if (!squeezed[static_cast<size_t>(i)]) {
            resultShape.push_back(inputType.getShape()[i]);
        }
    }

    auto resultType = mlir::RankedTensorType::get(
        resultShape, inputType.getElementType());

    if (resultShape.empty()) {
        llvm::SmallVector<mlir::Value> indices;
        indices.reserve(static_cast<size_t>(rank));
        for (int64_t i = 0; i < rank; ++i) {
            indices.push_back(builder.create<mlir::arith::ConstantIndexOp>(
                loc, 0));
        }
        auto scalar = builder.create<mlir::tensor::ExtractOp>(
            loc, inputType.getElementType(), input, indices);
        auto tensor = builder.create<mlir::tensor::FromElementsOp>(
            loc, resultType, scalar.getResult());
        values[node.outputs()[0]] = tensor.getResult();
        return;
    }

    llvm::SmallVector<mlir::ReassociationIndices> reassociation;
    llvm::SmallVector<int64_t> leadingSqueezed;
    for (int64_t i = 0; i < rank; ++i) {
        if (squeezed[static_cast<size_t>(i)]) {
            if (reassociation.empty()) {
                leadingSqueezed.push_back(i);
            } else {
                reassociation.back().push_back(i);
            }
            continue;
        }

        mlir::ReassociationIndices group;
        if (!leadingSqueezed.empty()) {
            group.append(leadingSqueezed);
            leadingSqueezed.clear();
        }
        group.push_back(i);
        reassociation.push_back(std::move(group));
    }

    if (!leadingSqueezed.empty()) {
        reassociation.back().append(leadingSqueezed);
    }

    auto squeeze = builder.create<mlir::tensor::CollapseShapeOp>(
        loc, resultType, input, reassociation);
    values[node.outputs()[0]] = squeeze.getResult();
}

void Codegen::genMatMulNode(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const Node &node,
    std::unordered_map<std::string, mlir::Value> &values) const {

    checkBinaryNodeShape(node, "MatMul");

    mlir::Value lhs = getBoundValue(values, node.inputs()[0], "MatMul");
    mlir::Value rhs = getBoundValue(values, node.inputs()[1], "MatMul");

    auto lhsType = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
    auto rhsType = mlir::dyn_cast<mlir::RankedTensorType>(rhs.getType());
    if (!lhsType || !rhsType) {
        throw std::runtime_error("MatMul expects ranked tensor inputs");
    }
    if (lhsType.getRank() != 2 || rhsType.getRank() != 2) {
        throw std::runtime_error(
            "MatMul currently supports only rank-2 tensor inputs");
    }
    if (!lhsType.getElementType().isF32() || !rhsType.getElementType().isF32()) {
        throw std::runtime_error("MatMul currently supports only f32 tensors");
    }
    if (lhsType.getElementType() != rhsType.getElementType()) {
        throw std::runtime_error("MatMul input element types must match");
    }

    int64_t lhsK = lhsType.getShape()[1];
    int64_t rhsK = rhsType.getShape()[0];
    if (!mlir::ShapedType::isDynamic(lhsK) &&
        !mlir::ShapedType::isDynamic(rhsK) && lhsK != rhsK) {
        throw std::runtime_error("MatMul reduction dimension mismatch");
    }

    std::vector<int64_t> resultShape = {
        lhsType.getShape()[0],
        rhsType.getShape()[1],
    };
    auto resultType = mlir::RankedTensorType::get(
        resultShape, lhsType.getElementType());

    std::vector<mlir::Value> dynamicDims;
    if (mlir::ShapedType::isDynamic(resultShape[0])) {
        dynamicDims.push_back(builder.create<mlir::tensor::DimOp>(loc, lhs, 0));
    }
    if (mlir::ShapedType::isDynamic(resultShape[1])) {
        dynamicDims.push_back(builder.create<mlir::tensor::DimOp>(loc, rhs, 1));
    }

    auto empty = builder.create<mlir::tensor::EmptyOp>(
        loc, resultShape, lhsType.getElementType(), dynamicDims);
    auto zero = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getF32FloatAttr(0.0f));
    auto init = builder.create<mlir::linalg::FillOp>(
        loc, mlir::TypeRange{resultType}, mlir::ValueRange{zero.getResult()},
        mlir::ValueRange{empty.getResult()});
    auto matmul = builder.create<mlir::linalg::MatmulOp>(
        loc, mlir::TypeRange{resultType}, mlir::ValueRange{lhs, rhs},
        mlir::ValueRange{init.getResult(0)});

    values[node.outputs()[0]] = matmul.getResult(0);
}

void Codegen::genSoftmaxNode(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const Node &node,
    std::unordered_map<std::string, mlir::Value> &values) const {

    checkUnaryNodeShape(node, "Softmax");

    mlir::Value input = getBoundValue(values, node.inputs()[0], "Softmax");
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType) {
        throw std::runtime_error("Softmax expects ranked tensor input");
    }
    if (!inputType.getElementType().isF32()) {
        throw std::runtime_error("Softmax currently supports only f32 tensors");
    }

    int64_t rank = inputType.getRank();
    int64_t axis = getIntAttribute(node, "axis", -1);
    if (axis < 0) {
        axis += rank;
    }
    if (axis < 0 || axis >= rank) {
        throw std::runtime_error("Softmax axis is out of range");
    }

    std::vector<mlir::Value> dynamicDims =
        collectDynamicDims(builder, loc, input, inputType);
    auto empty = builder.create<mlir::tensor::EmptyOp>(
        loc, inputType.getShape(), inputType.getElementType(), dynamicDims);
    auto softmax = builder.create<mlir::linalg::SoftmaxOp>(
        loc, mlir::TypeRange{inputType}, input, empty.getResult(),
        static_cast<uint64_t>(axis));

    values[node.outputs()[0]] = *softmax.getResult().begin();
}

} // namespace tensor_compiler
