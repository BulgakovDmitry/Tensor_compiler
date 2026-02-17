#ifndef INCLUDE_GRAPH_HPP
#define INCLUDE_GRAPH_HPP

#include "attribute.hpp"
#include "node.hpp"
#include "onnx.pb.h"
#include "tensor.hpp"
#include <cstddef>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#include <ostream>

namespace tensor_compiler {

using T_map = std::unordered_map<std::string, Tensor>;

class Graph {
  private:
    std::string name_;
    T_map tensors_;
    std::vector<Node> nodes_;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;

  public:
    Graph(const std::string &name) : name_{name} {}

    const std::string &get_name() const;
    const T_map &get_tensors() const;
    const std::vector<Node> &get_nodes() const;
    const std::vector<std::string> &get_inputs() const;
    const std::vector<std::string> &get_outputs() const;

    void set_name(std::string name);

    void add_tensor(Tensor tensor);

    void add_node(Node node);

    void set_inputs(std::vector<std::string> inputs);

    void set_outputs(std::vector<std::string> outputs);

    const Tensor *get_tensor(const std::string &name) const;

    void dump(std::ostream &os) const;
};

// ----------------------------------------------------------------------------
// @section Implementations
// Implementation of graph methods.
// ----------------------------------------------------------------------------
const std::string &Graph::get_name() const { return name_; }
const T_map &Graph::get_tensors() const { return tensors_; }
const std::vector<Node> &Graph::get_nodes() const { return nodes_; }
const std::vector<std::string> &Graph::get_inputs() const { return inputs_; }
const std::vector<std::string> &Graph::get_outputs() const { return outputs_; }

void Graph::set_name(std::string name) { name_ = name; }

void Graph::add_tensor(Tensor tensor) {
    tensors_.insert({tensor.get_name(), tensor});
}

void Graph::add_node(Node node) {
    nodes_.push_back(node);
}

void Graph::set_inputs(std::vector<std::string> inputs) {
    inputs_ = inputs;
}

void Graph::set_outputs(std::vector<std::string> outputs) {
    outputs_ = outputs;
}

const Tensor *Graph::get_tensor(const std::string &name) const {
    auto it = tensors_.find(name);
    if (it != tensors_.end()) {
        return &(it->second);
    }
    return nullptr;
}

} // namespace tensor_compiler

#endif // INCLUDE_GRAPH_HPP