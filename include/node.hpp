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

using node_id = std::size_t;
using value_id = std::size_t;
using Attributes = std::unordered_map<std::string, Attribute>;

static constexpr value_id invalid_value = static_cast<value_id>(-1);

class Node {
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

    void set_inputs(const std::vector<value_id>& inputs);
    void set_outputs(const std::vector<value_id>& outputs);
    void parse_attributes(const onnx::NodeProto &node);

    template <typename T> void set_attribute(const std::string &name, T value);

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
void Node::set_name(const std::string &name) { name_ = name; }

node_id Node::get_id() const { return id_; }
const std::string &Node::get_opcode() const { return opcode_; }
const std::string &Node::get_name() const { return name_; }
const std::vector<value_id> &Node::get_inputs() const { return inputs_; }
const std::vector<value_id> &Node::get_outputs() const { return outputs_; }
const Attributes &Node::get_attributes() const { return attributes_; }

void Node::set_inputs(const std::vector<value_id>& inputs) {
    inputs_ = inputs;
}

void Node::set_outputs(const std::vector<value_id>& outputs) {
    outputs_ = outputs;
}

void Node::parse_attributes(const onnx::NodeProto &node) {
    // TODO
}

void Node::add_input(value_id input) { inputs_.push_back(input); }
void Node::add_output(value_id output) { outputs_.push_back(output); }

template <class T> void Node::set_attribute(const std::string &name, T value) {
    auto &a = attributes_[name];
    a.set_value(value);
}

bool Node::has_attribute(const std::string &name) const {
    return attributes_.find(name) != attributes_.end();
}

bool Node::replace_input(value_id old_input, value_id new_input) {
    for (auto &input : inputs_) {
        if (input == old_input) {
            input = new_input;
            return true;
        }
    }
    return false;
}

bool Node::replace_output(value_id old_output, value_id new_output) {
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
