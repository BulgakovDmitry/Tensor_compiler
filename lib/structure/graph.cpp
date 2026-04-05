#include "structure/graph.h"

namespace tensor_compiler {

// ----------------------------------------------------------------------------
// @section Implementations
// Implementations
// ----------------------------------------------------------------------------
Graph::Graph(const onnx::GraphProto &graph) : name_{graph.name()} {
    for (const auto &initializer : graph.initializer()) {
        auto tensor = handle_tensor(initializer);
        add_tensor(std::move(tensor));
    }

    for (const auto &input : graph.input()) {
        auto tensor = handle_tensor(input, Tensor_kind::input);
        add_tensor(std::move(tensor));
        add_input(input.name());
    }

    std::size_t node_idx = 0;
    for (const auto &node : graph.node()) {
        auto new_node = handle_node(node_idx, node);
        add_node(std::move(new_node));
    }

    for (const auto &output : graph.output()) {
        Tensor tensor = handle_tensor(output, Tensor_kind::output);
        add_tensor(std::move(tensor));
        add_output(output.name());
    }
}

const std::string &Graph::get_name() const { return name_; }
const T_map &Graph::get_tensors() const { return tensors_; }
const std::vector<Node> &Graph::get_nodes() const { return nodes_; }
const std::vector<std::string> &Graph::get_inputs() const {
    return inputs_;
}
const std::vector<std::string> &Graph::get_outputs() const {
    return outputs_;
}

void Graph::set_name(std::string name) { name_ = std::move(name); }

void Graph::set_inputs(const std::vector<std::string> &inputs) {
    inputs_ = inputs;
}
void Graph::set_outputs(const std::vector<std::string> &outputs) {
    outputs_ = outputs;
}

void Graph::add_tensor(Tensor tensor) {
    tensors_.insert_or_assign(tensor.get_name(), std::move(tensor));
}

void Graph::add_node(Node node) { nodes_.push_back(std::move(node)); }

void Graph::add_input(const std::string &input) {
    inputs_.push_back(input);
}
void Graph::add_output(const std::string &output) {
    outputs_.push_back(output);
}

const Tensor *Graph::get_tensor(const std::string &name) const {
    auto it = tensors_.find(name);
    if (it != tensors_.end())
        return &(it->second);
    return nullptr;
}

Tensor Graph::handle_tensor(const onnx::TensorProto &t) {
    Tensor tensor{};
    tensor.set_name(t.name());
    tensor.set_dim(t.dims());
    tensor.set_type(t.data_type());
    tensor.set_data(extract_tensor_bytes(t));
    tensor.set_kind(Tensor_kind::constant);
    return tensor;
}

Tensor Graph::handle_tensor(const onnx::ValueInfoProto &t,
                                   const Tensor_kind &type) {
    Tensor tensor{};
    tensor.set_name(t.name());
    tensor.set_dim(extract_dims(t));
    tensor.set_type(extract_elem_type(t));
    tensor.set_kind(type);
    return tensor;
}

void Graph::handle_node_ir_tensor(Node &new_node,
                                         const onnx::NodeProto &node,
                                         const std::string &name) {
    if (name.empty())
        return;
    if (!get_tensor(name)) {
        Tensor t{};
        t.set_name(name);
        t.set_kind(Tensor_kind::intermediate);
        add_tensor(std::move(t));
    }
}

Node Graph::handle_node(std::size_t &node_idx,
                               const onnx::NodeProto &node) {
    Node new_node{node.name(), node.op_type(), node_idx++};
    new_node.set_inputs(node.input());
    new_node.set_outputs(node.output());
    new_node.parse_attributes(node);

    for (const auto &name : new_node.get_inputs())
        handle_node_ir_tensor(new_node, node, name);

    for (const auto &name : new_node.get_outputs())
        handle_node_ir_tensor(new_node, node, name);

    return new_node;
}

}
