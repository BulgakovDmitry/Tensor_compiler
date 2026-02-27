#ifndef UTILS_HPP
#define UTILS_HPP

#include "structure/graph.hpp"
#include "structure/node.hpp"
#include "structure/tensor.hpp"
#include <iostream>
#include <ostream>
#include <string>

namespace tensor_compiler {

struct Edge_hash {
    std::size_t
    operator()(const std::pair<const Node *, const Node *> &e) const noexcept {
        auto h1 = std::hash<const Node *>{}(e.first);
        auto h2 = std::hash<const Node *>{}(e.second);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

inline void dump_tensors(const Graph &graph, std::ostream &os) {
    os << "Graph name: " << graph.get_name() << "\n";
    os << "Tensors:\n";
    for (const auto &[name, tensor] : graph.get_tensors()) {
        os << "  " << name << ": type=" << tensor.get_type()
           << ", kind=" << static_cast<int>(tensor.get_kind()) << ", shape=[";
        for (size_t i = 0; i < tensor.get_shape().size(); ++i) {
            os << tensor.get_shape()[i];
            if (i < tensor.get_shape().size() - 1) {
                os << ", ";
            }
        }
        os << "]\n";
    }
}

inline void dump_nodes(const Graph &graph, std::ostream &os) {
    os << "Nodes:\n";
    for (const auto &node : graph.get_nodes()) {
        os << "  Node name: " << node.get_name() << "\n";
        os << "    Opcode: " << node.get_opcode() << "\n";
        os << "    Inputs: [";
        for (size_t i = 0; i < node.get_inputs().size(); ++i) {
            os << node.get_inputs()[i];
            if (i < node.get_inputs().size() - 1) {
                os << ", ";
            }
        }
        os << "]\n";

        os << "    Outputs: [";
        for (size_t i = 0; i < node.get_outputs().size(); ++i) {
            os << node.get_outputs()[i];
            if (i < node.get_outputs().size() - 1) {
                os << ", ";
            }
        }
        os << "]\n";
    }
}

} // namespace tensor_compiler

#endif // UTILS_HPP