#ifndef INCLUDE_GRAPH_HPP
#define INCLUDE_GRAPH_HPP

#include "handlers.hpp"
#include "node.hpp"
#include "tensor.hpp"
#include "handlers.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace tensor_compiler {

using T_map = std::unordered_map<std::string, Tensor>;

/// @brief Represents a computation graph consisting of nodes and tensors.
///
/// A graph owns a collection of tensors and nodes. It tracks the input and
/// output tensors of the entire graph. The graph can be built by adding
/// tensors and nodes, and querying them by name.
class Graph final {
  private:
    std::string name_;
    T_map tensors_;
    std::vector<Node> nodes_;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;

  public:
    /// @brief Construct the compute graph from an ONNX model graph
    /// @param graph onnx::GraphProto for building Graph.
    explicit Graph(const onnx::GraphProto &graph);

    /// @brief Get the graph name.
    /// @return const reference to name string.
    const std::string &get_name() const;

    /// @brief Get the map of tensors (name -> Tensor).
    /// @return const reference to T_map.
    const T_map &get_tensors() const;

    /// @brief Get the list of nodes in the graph.
    /// @return const reference to vector of Node.
    const std::vector<Node> &get_nodes() const;

    /// @brief Get the list of graph input tensor names.
    /// @return const reference to vector of strings.
    const std::vector<std::string> &get_inputs() const;

    /// @brief Get the list of graph output tensor names.
    /// @return const reference to vector of strings.
    const std::vector<std::string> &get_outputs() const;

    /// @brief Get a tensor by name.
    /// @param name Tensor name.
    /// @return Pointer to the tensor, or nullptr if not found.
    const Tensor *get_tensor(const std::string &name) const;

private:
    /// @brief Set the graph name.
    /// @param name New name.
    void set_name(std::string name);

    /// @brief Set the list of graph input tensor names.
    /// @param inputs Vector of input names.
    void set_inputs(const std::vector<std::string> &inputs);

    /// @brief Set the list of graph output tensor names.
    /// @param outputs Vector of output names.
    void set_outputs(const std::vector<std::string> &outputs);

    /// @brief Add a tensor to the graph.
    ///
    /// If a tensor with the same name already exists, it is replaced.
    /// @param tensor The tensor to add.
    void add_tensor(Tensor tensor);

    /// @brief Add a node to the graph.
    /// @param node The node to add.
    void add_node(Node node);

    /// @brief Append a name to the list of graph inputs.
    /// @param input Input tensor name.
    void add_input(const std::string &input);

    /// @brief Append a name to the list of graph outputs.
    /// @param output Output tensor name.
    void add_output(const std::string &output);

    /// @brief Convert an ONNX TensorProto to a Tensor object.
    /// @param t The ONNX TensorProto to convert.
    /// @return Tensor with name, dims, type, data and kind=constant.
    Tensor handle_tensor(const onnx::TensorProto &t);

    /// @brief Convert an ONNX ValueInfoProto to a Tensor object.
    /// @param t The ONNX ValueInfoProto to convert.
    /// @param type The kind to assign to the tensor (input/output).
    /// @return Tensor with name, dims, type and specified kind.
    Tensor handle_tensor(const onnx::ValueInfoProto &t,
                         const Tensor_kind &type);

    /// @brief Ensure a tensor exists in the graph for a node's I/O.
    ///
    /// If the tensor name is non-empty and not already present,
    /// creates a new intermediate tensor entry.
    /// @param new_node Reference to the node being built (unused, kept for API
    /// symmetry).
    /// @param node The ONNX NodeProto containing the tensor reference.
    /// @param name The tensor name to check/register.
    void handle_node_ir_tensor(Node &new_node, const onnx::NodeProto &node,
                               const std::string &name);

    /// @brief Convert an ONNX NodeProto to a Node object.
    /// @param node_idx Reference to a counter for assigning node indices.
    /// @param node The ONNX NodeProto to convert.
    /// @return Node with name, op_type, index, I/O lists and parsed attributes.
    Node handle_node(std::size_t &node_idx, const onnx::NodeProto &node);
};

// ----------------------------------------------------------------------------
// @section Implementations
// Implementations
// ----------------------------------------------------------------------------
inline Graph::Graph(const onnx::GraphProto &graph) : name_{graph.name()} {
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

inline const std::string &Graph::get_name() const { return name_; }
inline const T_map &Graph::get_tensors() const { return tensors_; }
inline const std::vector<Node> &Graph::get_nodes() const { return nodes_; }
inline const std::vector<std::string> &Graph::get_inputs() const {
    return inputs_;
}
inline const std::vector<std::string> &Graph::get_outputs() const {
    return outputs_;
}

inline void Graph::set_name(std::string name) { name_ = std::move(name); }

inline void Graph::set_inputs(const std::vector<std::string> &inputs) {
    inputs_ = inputs;
}
inline void Graph::set_outputs(const std::vector<std::string> &outputs) {
    outputs_ = outputs;
}

inline void Graph::add_tensor(Tensor tensor) {
    tensors_.insert_or_assign(tensor.get_name(), std::move(tensor));
}

inline void Graph::add_node(Node node) { nodes_.push_back(std::move(node)); }

inline void Graph::add_input(const std::string &input) {
    inputs_.push_back(input);
}
inline void Graph::add_output(const std::string &output) {
    outputs_.push_back(output);
}

inline const Tensor *Graph::get_tensor(const std::string &name) const {
    auto it = tensors_.find(name);
    if (it != tensors_.end())
        return &(it->second);
    return nullptr;
}

inline Tensor Graph::handle_tensor(const onnx::TensorProto &t) {
    Tensor tensor{};
    tensor.set_name(t.name());
    tensor.set_dim(t.dims());
    tensor.set_type(t.data_type());
    tensor.set_data(extract_tensor_bytes(t));
    tensor.set_kind(Tensor_kind::constant);
    return tensor;
}

inline Tensor Graph::handle_tensor(const onnx::ValueInfoProto &t,
                                   const Tensor_kind &type) {
    Tensor tensor{};
    tensor.set_name(t.name());
    tensor.set_dim(extract_dims(t));
    tensor.set_type(extract_elem_type(t));
    tensor.set_kind(type);
    return tensor;
}

inline void Graph::handle_node_ir_tensor(Node &new_node,
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

inline Node Graph::handle_node(std::size_t &node_idx,
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

} // namespace tensor_compiler

#endif // INCLUDE_GRAPH_HPP
