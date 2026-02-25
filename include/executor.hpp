#ifndef INCLUDE_EXECUTOR_HPP
#define INCLUDE_EXECUTOR_HPP

#include "structure/attribute.hpp"
#include "structure/graph.hpp"
#include "structure/node.hpp"
#include "structure/tensor.hpp"
#include "utils.hpp"
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <deque>

namespace tensor_compiler {

class Executor {
  private:
    const Graph &graph_;
    T_map tensor_values_;

  public:
    Executor(const Graph &graph) : graph_{graph} {
        for (const auto &[name, tensor] : graph_.get_tensors()) {
            if (tensor.is_constant()) {
                tensor_values_.insert({name, tensor});
            }
        }
    }

    // T_map execute(const std::unordered_map<std::string, std::vector<float>>
    // //TODO
    //                   &input_values) {
    //     load_inputs(input_values);

    //     topological_sort();

    //     // ... + execution of all nodes and get output_values of compute
    //     graph

    //     return output_values;
    // }

    std::vector<const Node *> topological_sort();
  private:
    /**
     * @brief Loads input values to tensor_values_
     *
     * @param input_values
     * @return void
     */
    void load_inputs(const std::unordered_map<std::string, std::vector<float>>
                         &input_values);

};

// ----------------------------------------------------------------------------
// @section Implementations
// Implementation of executor methods.
// ----------------------------------------------------------------------------
inline void Executor::load_inputs(
    const std::unordered_map<std::string, std::vector<float>> &input_values) {
    for (const auto &input_name : graph_.get_inputs()) {
        auto it = input_values.find(input_name);
        if (it == input_values.end())
            throw std::runtime_error("Missing input value for : " + input_name);

        auto input_tensor = Tensor::create(
            input_name, graph_.get_tensor(input_name)->get_shape(), it->second,
            Tensor_kind::input);
        tensor_values_[input_name] = std::move(input_tensor);
    }
}

inline std::vector<const Node *> Executor::topological_sort() {
    std::unordered_map<std::string, const Node *> producer_of;
    // init producer_of ---------------------------------------------
    for (const auto &node : graph_.get_nodes()) {
        for (const auto &out : node.get_outputs()) {
            producer_of[out] = &node;
        }
    }
    // --------------------------------------------------------------

    std::unordered_map<const Node*, std::vector<const Node*>> adj;
    std::unordered_map<const Node*, std::size_t> in_degree;
    // init adj and in_degree ---------------------------------------
    for (const auto& node : graph_.get_nodes()) {
        const Node* p = &node;
        adj[p] = {};
        in_degree[p] = 0;
    }
    // --------------------------------------------------------------

    // To avoid duplicate edges
    std::unordered_set<std::pair<const Node*, const Node*>, Edge_hash> seen;

    // build edges from inputs --------------------------------------
    for (const auto& consumer_node : graph_.get_nodes()) {
        const Node* consumer = &consumer_node;

        for (const auto& input_tensor : consumer_node.get_inputs()) {
            auto it = producer_of.find(input_tensor);
            if (it == producer_of.end()) {
                continue;
            }
            const Node* producer = it->second;
            std::pair<const Node*, const Node*> edge{producer, consumer};
            if (seen.insert(edge).second) {
                adj[producer].push_back(consumer);
                in_degree[consumer] += 1;
            }
        }
    }
    // --------------------------------------------------------------

    std::deque<const Node*> kahn_queue;
    // init kahn_queue ----------------------------------------------
    for (const auto& node : graph_.get_nodes()) {
        const Node* p = &node;
        if (in_degree[p] == 0) kahn_queue.push_back(p);
    }
    // --------------------------------------------------------------

    const std::size_t num_of_nodes = graph_.get_nodes().size();     

    std::vector<const Node*> ready_nodes;
    ready_nodes.reserve(num_of_nodes);

    // add ready nodes to  vector -----------------------------------
    while (!kahn_queue.empty()) {
        const Node* ready_node = kahn_queue.front();
        kahn_queue.pop_front();
        ready_nodes.push_back(ready_node);

        for (const Node* output_tensors : adj[ready_node]) {
            if (--in_degree[output_tensors] == 0) {
                kahn_queue.push_back(output_tensors);
            }
        }
    }
    // --------------------------------------------------------------

    if (ready_nodes.size() != num_of_nodes) {
        throw std::runtime_error("Topological sort failed: "
                                 "graph has a cycle (or unresolved deps).");
    }

    return ready_nodes;
}

} // namespace tensor_compiler

#endif // INCLUDE_EXECUTOR_HPP
