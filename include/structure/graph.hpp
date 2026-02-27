#ifndef INCLUDE_GRAPH_HPP
#define INCLUDE_GRAPH_HPP

#include "node.hpp"
#include "tensor.hpp"
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace tensor_compiler {

using T_map = std::unordered_map<std::string, Tensor>;

/**
 * @brief Represents a computation graph consisting of nodes and tensors.
 *
 * A graph owns a collection of tensors and nodes. It tracks the input and
 * output tensors of the entire graph. The graph can be built by adding
 * tensors and nodes, and querying them by name.
 */
class Graph {
  private:
    std::string name_;
    T_map tensors_;
    std::vector<Node> nodes_;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;

  public:
    Graph() = default;

    /**
     * @brief Construct a new Graph with a name.
     * @param name Graph name.
     */
    explicit Graph(const std::string &name) : name_{name} {}

    /**
     * @brief Get the graph name.
     * @return const reference to name string.
     */
    const std::string &get_name() const;

    /**
     * @brief Get the map of tensors (name -> Tensor).
     * @return const reference to T_map.
     */
    const T_map &get_tensors() const;

    /**
     * @brief Get the list of nodes in the graph.
     * @return const reference to vector of Node.
     */
    const std::vector<Node> &get_nodes() const;

    /**
     * @brief Get the list of graph input tensor names.
     * @return const reference to vector of strings.
     */
    const std::vector<std::string> &get_inputs() const;

    /**
     * @brief Get the list of graph output tensor names.
     * @return const reference to vector of strings.
     */
    const std::vector<std::string> &get_outputs() const;

    /**
     * @brief Set the graph name.
     * @param name New name.
     */
    void set_name(std::string name);

    /**
     * @brief Set the list of graph input tensor names.
     * @param inputs Vector of input names.
     */
    void set_inputs(const std::vector<std::string> &inputs);

    /**
     * @brief Set the list of graph output tensor names.
     * @param outputs Vector of output names.
     */
    void set_outputs(const std::vector<std::string> &outputs);

    /**
     * @brief Add a tensor to the graph.
     *
     * If a tensor with the same name already exists, it is replaced.
     * @param tensor The tensor to add.
     */
    void add_tensor(Tensor tensor);

    /**
     * @brief Add a node to the graph.
     * @param node The node to add.
     */
    void add_node(Node node);

    /**
     * @brief Append a name to the list of graph inputs.
     * @param input Input tensor name.
     */
    void add_input(const std::string &input);

    /**
     * @brief Append a name to the list of graph outputs.
     * @param output Output tensor name.
     */
    void add_output(const std::string &output);

    /**
     * @brief Get a tensor by name.
     * @param name Tensor name.
     * @return Pointer to the tensor, or nullptr if not found.
     */
    const Tensor *get_tensor(const std::string &name) const;
};

// ----------------------------------------------------------------------------
// @section Implementations
// Implementations
// ----------------------------------------------------------------------------
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

} // namespace tensor_compiler

#endif // INCLUDE_GRAPH_HPP
