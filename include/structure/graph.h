#ifndef INCLUDE_GRAPH_H
#define INCLUDE_GRAPH_H

#include "handlers.h"
#include "node.h"
#include "tensor.h"
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
  Tensor handle_tensor(const onnx::ValueInfoProto &t, const Tensor_kind &type);

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

} // namespace tensor_compiler

#endif // INCLUDE_GRAPH_H
