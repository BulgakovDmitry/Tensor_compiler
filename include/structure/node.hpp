#ifndef INCLUDE_NODE_HPP
#define INCLUDE_NODE_HPP

#include "attribute.hpp"
#include "onnx.pb.h"
#include "tensor.hpp"
#include <cstddef>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace tensor_compiler {

using Attributes = std::unordered_map<std::string, Attribute>;

/**
 * @brief Represents a node in the computation graph.
 *
 * A node corresponds to an operator in the neural network. It stores the * operator type (opcode), its inputs/outputs as tensor names, and a set of
 * attributes that parameterize the operator.
 */
class Node {
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
    /**
     * @brief Construct a new Node object.
     * @param name Human-readable name of the node.
     * @param opcode Operator type (e.g., "Conv", "Relu").
     * @param id Unique identifier (default 0).
     */
    Node(const std::string &name, const std::string &opcode, node_id id = 0)
        : id_{id}, name_{name}, opcode_{opcode} {}

    /**
     * @brief Set the node's name.
     * @param name New name.
     */
    void set_name(const std::string &name);

    /**
     * @brief Get the node's unique identifier.
     * @return node_id.
     */
    node_id get_id() const;

    /**
     * @brief Get the operator type.
     * @return const reference to opcode string.
     */
    const std::string &get_opcode() const;

    /**
     * @brief Get the node's name.
     * @return const reference to name string.
     */
    const std::string &get_name() const;

    /**
     * @brief Get the list of input tensor names.
     * @return const reference to vector of strings.
     */
    const std::vector<std::string> &get_inputs() const;

    /**
     * @brief Get the list of output tensor names.
     * @return const reference to vector of strings.
     */
    const std::vector<std::string> &get_outputs() const;

    /**
     * @brief Get the node's attributes map.
     * @return const reference to Attributes map.
     */
    const Attributes &get_attributes() const;

    /**
     * @brief Set the inputs from a vector of strings.
     * @param inputs Vector of input names.
     */
    void set_inputs(const std::vector<std::string> &inputs);

    /**
     * @brief Set the inputs from a protobuf repeated field.
     * @param inputs Protobuf repeated field of strings.
     */
    void set_inputs(const name_t &inputs);

    /**
     * @brief Set the outputs from a vector of strings.
     * @param outputs Vector of output names.
     */
    void set_outputs(const std::vector<std::string> &outputs);

    /**
     * @brief Set the outputs from a protobuf repeated field.
     * @param outputs Protobuf repeated field of strings.
     */
    void set_outputs(const name_t &outputs);

    /**
     * @brief Parse attributes from an ONNX NodeProto.
     *
     * Reads the attribute field of the protobuf and stores them as Attribute
     * objects. Supported types: FLOAT, INT, STRING, FLOATS, INTS.
     * @param node ONNX NodeProto message.
     */
    void parse_attributes(const onnx::NodeProto &node);

    /**
     * @brief Set an attribute value.
     * @param name Attribute name.
     * @param value Attribute value (variant type).
     */
    void set_attribute(const std::string &name,
                       const Attribute::AttrValue &value);

    /**
     * @brief Check if an attribute exists.
     * @param name Attribute name.
     * @return true if present, false otherwise.
     */
    bool has_attribute(const std::string &name) const;

  private:
    void add_input(const std::string &input);
    void add_output(const std::string &output);
};

// ----------------------------------------------------------------------------
// @section Implementations
// Implementation of node methods.
// ----------------------------------------------------------------------------
inline void Node::set_name(const std::string &name) { name_ = name; }

inline Node::node_id Node::get_id() const { return id_; }
inline const std::string &Node::get_opcode() const { return opcode_; }
inline const std::string &Node::get_name() const { return name_; }
inline const std::vector<std::string> &Node::get_inputs() const {
    return inputs_;
}
inline const std::vector<std::string> &Node::get_outputs() const {
    return outputs_;
}
inline const Attributes &Node::get_attributes() const { return attributes_; }

inline void Node::set_inputs(const std::vector<std::string> &inputs) {
    inputs_ = inputs;
}

inline void Node::set_inputs(const name_t &inputs) {
    inputs_.clear();
    inputs_.reserve(static_cast<std::size_t>(inputs.size()));
    for (const auto &s : inputs)
        inputs_.push_back(s);
}

inline void Node::set_outputs(const std::vector<std::string> &outputs) {
    outputs_ = outputs;
}

inline void Node::set_outputs(const name_t &outputs) {
    outputs_.clear();
    outputs_.reserve(static_cast<std::size_t>(outputs.size()));
    for (const auto &s : outputs)
        outputs_.push_back(s);
}

inline void Node::parse_attributes(const onnx::NodeProto &node) {
    for (const auto &attr : node.attribute()) {
        const std::string &name = attr.name();

        switch (attr.type()) {
        case onnx::AttributeProto_AttributeType_FLOAT: {
            set_attribute(name, attr.f());
            break;
        }

        case onnx::AttributeProto_AttributeType_INT: {
            set_attribute(name, static_cast<int64_t>(attr.i()));
            break;
        }

        case onnx::AttributeProto_AttributeType_STRING: {
            set_attribute(name, attr.s());
            break;
        }

        case onnx::AttributeProto_AttributeType_FLOATS: {
            std::vector<float> v;
            v.reserve(attr.floats_size());
            for (int i = 0; i < attr.floats_size(); ++i)
                v.push_back(attr.floats(i));
            set_attribute(name, v);
            break;
        }

        case onnx::AttributeProto_AttributeType_INTS: {
            std::vector<int64_t> v;
            v.reserve(attr.ints_size());
            for (int i = 0; i < attr.ints_size(); ++i)
                v.push_back(static_cast<int64_t>(attr.ints(i)));
            set_attribute(name, v);
            break;
        }

        default:
            break;
        }
    }
}

inline void Node::add_input(const std::string &input) {
    inputs_.push_back(input);
}
inline void Node::add_output(const std::string &output) {
    outputs_.push_back(output);
}

inline void Node::set_attribute(const std::string &name,
                                const Attribute::AttrValue &value) {
    attributes_[name] = Attribute{name, value};
}

inline bool Node::has_attribute(const std::string &name) const {
    return attributes_.find(name) != attributes_.end();
}

} // namespace tensor_compiler

#endif // INCLUDE_NODE_HPP
