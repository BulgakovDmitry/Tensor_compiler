#ifndef INCLUDE_GRAPH_HPP
#define INCLUDE_GRAPH_HPP

#include "attribute.hpp"
#include "node.hpp"
#include "onnx.pb.h"
#include "tensor.hpp"
#include <cstddef>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <ostream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace tensor_compiler {

using T_map = std::unordered_map<std::string, Tensor>;

class Graph {
  private:
    std::string name_;
    T_map tensors_;
    std::vector<Node> nodes_;
    std::vector<value_id> inputs_;
    std::vector<value_id> outputs_;

  public:
    Graph() = default;
    Graph(const std::string &name) : name_{name} {}

    const std::string &get_name() const;
    const T_map &get_tensors() const;
    const std::vector<Node> &get_nodes() const;
    const std::vector<value_id> &get_inputs() const;
    const std::vector<value_id> &get_outputs() const;

    void set_name(std::string name);
    void set_inputs(const std::vector<value_id> &inputs);
    void set_outputs(const std::vector<value_id> &outputs);

    void add_tensor(Tensor tensor);
    void add_node(Node node);
    void add_input(const value_id input);
    void add_output(const value_id output);

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
const std::vector<value_id> &Graph::get_inputs() const { return inputs_; }
const std::vector<value_id> &Graph::get_outputs() const { return outputs_; }

void Graph::set_name(std::string name) { name_ = name; }

void Graph::add_tensor(Tensor tensor) {
    tensors_.emplace(tensor.get_name(), tensor);
}

void Graph::add_node(Node node) { nodes_.push_back(node); }

void Graph::set_inputs(const std::vector<value_id> &inputs) {
    inputs_ = inputs;
}

void Graph::set_outputs(const std::vector<value_id> &outputs) {
    outputs_ = outputs;
}

void Graph::add_input(const value_id input) { inputs_.push_back(input); }
void Graph::add_output(const value_id output) { outputs_.push_back(output); }

const Tensor *Graph::get_tensor(const std::string &name) const {
    auto it = tensors_.find(name);
    if (it != tensors_.end()) {
        return &(it->second);
    }
    return nullptr;
}

} // namespace tensor_compiler

#endif // INCLUDE_GRAPH_HPP