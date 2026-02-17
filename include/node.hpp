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

enum class Opcode {
    Add,
    Sub,
    Mul,
    Div,
    Relu,
};

class Node {
  private:
    node_id id_;
    Opcode opcode;
    std::string name_;

    std::vector<value_id> inputs_;
    std::vector<value_id> outputs_;
    std::vector<Attribute> attributes_;

  public:
    Node(node_id id, const std::string &name, Opcode opcode)
        : id_{id}, name_{name}, opcode{opcode} {}

    void set_name(const std::string &name);

    node_id get_id() const;
    Opcode get_opcode() const;
    const std::string &get_name() const;
    const std::vector<value_id> &get_inputs();
    const std::vector<value_id> &get_outputs();
    const std::vector<Attribute> &get_attributes() const;

    void add_input(value_id input);
    void add_output(value_id output);

    template <typename T> void set_attribute(const std::string &name, T value);

    bool has_attribute(const std::string &name) const;

    bool replace_input(value_id old_input, value_id new_input);
};

// ----------------------------------------------------------------------------
// Implementations
// ----------------------------------------------------------------------------

void Node::set_name(const std::string &name) { name_ = name; }

node_id Node::get_id() const { return id_; }
Opcode Node::get_opcode() const { return opcode; }
const std::string &Node::get_name() const { return name_; }
const std::vector<value_id> &Node::get_inputs() { return inputs_; }
const std::vector<value_id> &Node::get_outputs() { return outputs_; }
const std::vector<Attribute> &Node::get_attributes() const {
    return attributes_;
}

void Node::add_input(value_id input) { inputs_.push_back(input); }
void Node::add_output(value_id output) { outputs_.push_back(output); }

template <typename T>
void Node::set_attribute(const std::string &name, T value) {
    for (auto &attr : attributes_) {
        if (attr.get_name() == name) {
            attr.set_value(value);
            return;
        }
    }
    attributes_.emplace_back(name, value);
}

bool Node::has_attribute(const std::string &name) const {
    for (const auto &attr : attributes_) {
        if (attr.get_name() == name) {
            return true;
        }
    }
    return false;
}

} // namespace tensor_compiler

#endif // INCLUDE_NODE_HPP
