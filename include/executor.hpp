#ifndef INCLUDE_EXECUTOR_HPP
#define INCLUDE_EXECUTOR_HPP

#include "structure/attribute.hpp"
#include "structure/graph.hpp"
#include "structure/node.hpp"
#include "structure/tensor.hpp"
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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

  private:
    /**
     * @brief Loads input values to tensor_values_
     *
     * @param input_values
     * @return void
     */
    void load_inputs(const std::unordered_map<std::string, std::vector<float>>
                         &input_values);

    std::vector<const Node *> topological_sort() {
        std::unordered_map<std::string, const Node *> producer_of;
        for (const auto &node : graph_.get_nodes()) {
            for (const auto &out : node.get_outputs())
                producer_of[out] = &node;
        }

        // ... + Kana's algorithm
    }
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
    for (const auto &node : graph_.get_nodes()) {
        for (const auto &out : node.get_outputs())
            producer_of[out] = &node;
    }

    // ... + Kana's algorithm
}

} // namespace tensor_compiler

#endif // INCLUDE_EXECUTOR_HPP
