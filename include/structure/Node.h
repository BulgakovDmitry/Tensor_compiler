#ifndef INCLUDE_NODE_H
#define INCLUDE_NODE_H

#include "Attribute.h"
#include "onnx.pb.h"
#include "Tensor.h"
#include <cstddef>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace tensor_compiler {

using Attributes = std::unordered_map<std::string, Attribute>;

/// @brief Represents a node in the computation graph.
///
/// A node corresponds to an operator in the neural network. It stores the *
/// operator type (opcode), its inputs/outputs as tensor names, and a set of
/// attributes that parameterize the operator.
class Node final {
public:
  using node_id = std::size_t;
  using name_t = google::protobuf::RepeatedPtrField<std::string>;

private:
  node_id id_{0};
  std::string opcode_;
  std::string name_;

  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;

  Attributes attributes_;

public:
  /// @brief Construct a new Node object.
  /// @param name Human-readable name of the node.
  /// @param opcode Operator type (e.g., "Conv", "Relu").
  /// @param id Unique identifier (default 0).
  Node(const std::string &name, const std::string &opcode, node_id id = 0)
      : id_{id}, opcode_{opcode}, name_{name} {}

  /// @brief Set the node's name.
  /// @param name New name.
  void setName(const std::string &name);

  /// @brief Get the node's unique identifier.
  /// @return node_id.
  node_id id() const;

  /// @brief Get the operator type.
  /// @return const reference to opcode string.
  const std::string &opcode() const;

  /// @brief Get the node's name.
  /// @return const reference to name string.
  const std::string &name() const;

  /// @brief Get the list of input tensor names.
  /// @return const reference to vector of strings.
  const std::vector<std::string> &inputs() const;

  /// @brief Get the list of output tensor names.
  /// @return const reference to vector of strings.
  const std::vector<std::string> &outputs() const;

  /// @brief Get the node's attributes map.
  /// @return const reference to Attributes map.
  const Attributes &attributes() const;

  /// @brief Set the inputs from a vector of strings.
  /// @param inputs Vector of input names.
  void setInputs(const std::vector<std::string> &inputs);

  /// @brief Set the inputs from a protobuf repeated field.
  /// @param inputs Protobuf repeated field of strings.
  void setInputs(const name_t &inputs);

  /// @brief Set the outputs from a vector of strings.
  /// @param outputs Vector of output names.
  void setOutputs(const std::vector<std::string> &outputs);

  /// @brief Set the outputs from a protobuf repeated field.
  /// @param outputs Protobuf repeated field of strings.
  void setOutputs(const name_t &outputs);

  /// @brief Parse attributes from an ONNX NodeProto.
  ///
  /// Reads the attribute field of the protobuf and stores them as Attribute
  /// objects. Supported types: FLOAT, INT, STRING, FLOATS, INTS.
  /// @param node ONNX NodeProto message.
  void parseAttributes(const onnx::NodeProto &node);

  /// @brief Set an attribute value.
  /// @param name Attribute name.
  /// @param value Attribute value (variant type).
  void setAttribute(const std::string &name, const Attribute::AttrValue &value);

  /// @brief Check if an attribute exists.
  /// @param name Attribute name.
  /// @return true if present, false otherwise.
  bool hasAttribute(const std::string &name) const;

private:
  void addInput(const std::string &input);
  void addOutput(const std::string &output);
};

} // namespace tensor_compiler

#endif // INCLUDE_NODE_H
