#include "structure/graph.h"
#include "handlers.h"

namespace tensor_compiler {

// ----------------------------------------------------------------------------
// @section Implementations
// Implementations
// ----------------------------------------------------------------------------
Graph::Graph(const onnx::GraphProto &graph) : name_{graph.name()} {
    for (const auto &initializer : graph.initializer()) {
        auto tensor = handleTensor(initializer);
        addTensor(std::move(tensor));
    }

    for (const auto &input : graph.input()) {
        auto tensor = handleTensor(input, Tensor_kind::input);
        addTensor(std::move(tensor));
        addInput(input.name());
    }

    std::size_t node_idx = 0;
    for (const auto &node : graph.node()) {
        auto new_node = handleNode(node_idx, node);
        addNode(std::move(new_node));
    }

    for (const auto &output : graph.output()) {
        Tensor tensor = handleTensor(output, Tensor_kind::output);
        addTensor(std::move(tensor));
        addOutput(output.name());
    }
}

const std::string &Graph::name() const { return name_; }
const T_map &Graph::tensors() const { return tensors_; }
const std::vector<Node> &Graph::nodes() const { return nodes_; }
const std::vector<std::string> &Graph::inputs() const {
    return inputs_;
}
const std::vector<std::string> &Graph::outputs() const {
    return outputs_;
}

void Graph::setName(std::string name) { name_ = std::move(name); }

void Graph::setInputs(const std::vector<std::string> &inputs) {
    inputs_ = inputs;
}
void Graph::setOutputs(const std::vector<std::string> &outputs) {
    outputs_ = outputs;
}

void Graph::addTensor(Tensor tensor) {
    tensors_.insert_or_assign(tensor.name(), std::move(tensor));
}

void Graph::addNode(Node node) { nodes_.push_back(std::move(node)); }

void Graph::addInput(const std::string &input) {
    inputs_.push_back(input);
}
void Graph::addOutput(const std::string &output) {
    outputs_.push_back(output);
}

const Tensor *Graph::tensor(const std::string &name) const {
    auto it = tensors_.find(name);
    if (it != tensors_.end())
        return &(it->second);
    return nullptr;
}

Tensor Graph::handleTensor(const onnx::TensorProto &t) {
    Tensor tensor{};
    tensor.setName(t.name());
    tensor.setDim(t.dims());
    tensor.setType(t.data_type());
    tensor.setData(extractTensorBytes(t));
    tensor.setKind(Tensor_kind::constant);
    return tensor;
}

Tensor Graph::handleTensor(const onnx::ValueInfoProto &t,
                                   const Tensor_kind &type) {
    Tensor tensor{};
    tensor.setName(t.name());
    tensor.setShape(extractDims(t));
    tensor.setType(extractElemType(t));
    tensor.setKind(type);
    return tensor;
}

void Graph::handleNodeIRTensor(Node &new_node,
                                         const onnx::NodeProto &node,
                                         const std::string &name) {
    if (name.empty())
        return;
    if (!tensor(name)) {
        Tensor t{};
        t.setName(name);
        t.setKind(Tensor_kind::intermediate);
        addTensor(std::move(t));
    }
}

Node Graph::handleNode(std::size_t &node_idx,
                               const onnx::NodeProto &node) {
    Node new_node{node.name(), node.op_type(), node_idx++};
    new_node.setInputs(node.input());
    new_node.setOutputs(node.output());
    new_node.parseAttributes(node);

    for (const auto &name : new_node.inputs())
        handleNodeIRTensor(new_node, node, name);

    for (const auto &name : new_node.outputs())
        handleNodeIRTensor(new_node, node, name);

    return new_node;
}

}
