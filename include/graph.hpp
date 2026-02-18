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

class Graph {
  public:
    using T_map = std::unordered_map<std::string, Tensor>;

  private:
    std::string name_;
    T_map tensors_;
    std::vector<Node> nodes_;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;

  public:
    Graph() = default;
    Graph(const std::string &name) : name_{name} {}

    const std::string &get_name() const;
    const T_map &get_tensors() const;
    const std::vector<Node> &get_nodes() const;
    const std::vector<std::string> &get_inputs() const;
    const std::vector<std::string> &get_outputs() const;

    void set_name(std::string name);
    void set_inputs(const std::vector<std::string> &inputs);
    void set_outputs(const std::vector<std::string> &outputs);

    void add_tensor(Tensor tensor);
    void add_node(Node node);
    void add_input(const std::string input);
    void add_output(const std::string output);

    const Tensor *get_tensor(const std::string &name) const;

    void dump(std::ostream &os) const;
};

// ----------------------------------------------------------------------------
// @section Implementations
// Implementation of graph methods.
// ----------------------------------------------------------------------------
inline const std::string &Graph::get_name() const { return name_; }
inline const Graph::T_map &Graph::get_tensors() const { return tensors_; }
inline const std::vector<Node> &Graph::get_nodes() const { return nodes_; }
inline const std::vector<std::string> &Graph::get_inputs() const {
    return inputs_;
}
inline const std::vector<std::string> &Graph::get_outputs() const {
    return outputs_;
}

inline void Graph::set_name(std::string name) { name_ = name; }

inline void Graph::add_tensor(Tensor tensor) {
    tensors_.emplace(tensor.get_name(), tensor);
}

inline void Graph::add_node(Node node) { nodes_.push_back(node); }

inline void Graph::set_inputs(const std::vector<std::string> &inputs) {
    inputs_ = inputs;
}

inline void Graph::set_outputs(const std::vector<std::string> &outputs) {
    outputs_ = outputs;
}

inline void Graph::add_input(const std::string input) {
    inputs_.push_back(input);
}
inline void Graph::add_output(const std::string output) {
    outputs_.push_back(output);
}

inline const Tensor *Graph::get_tensor(const std::string &name) const {
    auto it = tensors_.find(name);
    if (it != tensors_.end()) {
        return &(it->second);
    }
    return nullptr;
}

} // namespace tensor_compiler

#endif // INCLUDE_GRAPH_HPP