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

class Node {
  public:
    using node_id = std::size_t;
    using value_id = std::size_t;
    using Attributes = std::unordered_map<std::string, Attribute>;
    using name_t = google::protobuf::RepeatedPtrField<std::string>;

  private:
    node_id id_;
    std::string opcode_;
    std::string name_;

    std::vector<value_id> inputs_;
    std::vector<value_id> outputs_;
    Attributes attributes_;

  public:
    Node(const std::string &name, const std::string &opcode, node_id id = 0)
        : id_{id}, name_{name}, opcode_{opcode} {}

    void set_name(const std::string &name);

    node_id get_id() const;
    const std::string &get_opcode() const;
    const std::string &get_name() const;
    const std::vector<value_id> &get_inputs() const;
    const std::vector<value_id> &get_outputs() const;
    const Attributes &get_attributes() const;

    void set_inputs(const std::vector<value_id> &inputs);
    void set_inputs(const name_t &inputs);

    void set_outputs(const std::vector<value_id> &outputs);
    void set_outputs(const name_t &outputs);
    void parse_attributes(const onnx::NodeProto &node);

    void set_attribute(const std::string &name,
                       const Attribute::AttrValue &value);

    bool has_attribute(const std::string &name) const;

    bool replace_input(value_id old_input, value_id new_input);
    bool replace_output(value_id old_output, value_id new_output);

  private:
    void add_input(value_id input);
    void add_output(value_id output);
};

// ----------------------------------------------------------------------------
// @section Implementations
// Implementation of node methods.
// ----------------------------------------------------------------------------
inline void Node::set_name(const std::string &name) { name_ = name; }

inline Node::node_id Node::get_id() const { return id_; }
inline const std::string &Node::get_opcode() const { return opcode_; }
inline const std::string &Node::get_name() const { return name_; }
inline const std::vector<Node::value_id> &Node::get_inputs() const {
    return inputs_;
}
inline const std::vector<Node::value_id> &Node::get_outputs() const {
    return outputs_;
}
inline const Node::Attributes &Node::get_attributes() const {
    return attributes_;
}

inline void Node::set_inputs(const std::vector<value_id> &inputs) {
    inputs_ = inputs;
}

inline void Node::set_inputs(const name_t &inputs) {
    inputs_.clear();
    for (const auto &input : inputs) {
        inputs_.push_back(std::hash<std::string>{}(input));
    }
}

inline void Node::set_outputs(const std::vector<value_id> &outputs) {
    outputs_ = outputs;
}

inline void Node::set_outputs(const name_t &outputs) {
    outputs_.clear();
    for (const auto &output : outputs) {
        outputs_.push_back(std::hash<std::string>{}(output));
    }
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

inline void Node::add_input(value_id input) { inputs_.push_back(input); }
inline void Node::add_output(value_id output) { outputs_.push_back(output); }

inline void Node::set_attribute(const std::string &name,
                                const Attribute::AttrValue &value) {
    attributes_[name] = Attribute{name, value};
}

inline bool Node::has_attribute(const std::string &name) const {
    return attributes_.find(name) != attributes_.end();
}

inline bool Node::replace_input(value_id old_input, value_id new_input) {
    for (auto &input : inputs_) {
        if (input == old_input) {
            input = new_input;
            return true;
        }
    }
    return false;
}

inline bool Node::replace_output(value_id old_output, value_id new_output) {
    for (auto &output : outputs_) {
        if (output == old_output) {
            output = new_output;
            return true;
        }
    }
    return false;
}

} // namespace tensor_compiler

#endif // INCLUDE_NODE_HPP
