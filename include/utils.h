#ifndef UTILS_H
#define UTILS_H

#include "structure/graph.h"
#include "structure/node.h"
#include "structure/tensor.h"
#include <iostream>
#include <ostream>
#include <string>

namespace tensor_compiler {

struct EdgeHash {
  std::size_t
  operator()(const std::pair<const Node *, const Node *> &e) const noexcept {
    auto h1 = std::hash<const Node *>{}(e.first);
    auto h2 = std::hash<const Node *>{}(e.second);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

inline void dumpTensors(const Graph &graph, std::ostream &os) {
  os << "Graph name: " << graph.name() << "\n";
  os << "Tensors:\n";
  for (const auto &[name, tensor] : graph.tensors()) {
    os << "  " << name << ": type=" << tensor.type()
       << ", kind=" << static_cast<int>(tensor.kind()) << ", shape=[";
    for (size_t i = 0; i < tensor.shape().size(); ++i) {
      os << tensor.shape()[i];
      if (i < tensor.shape().size() - 1) {
        os << ", ";
      }
    }
    os << "]\n";
  }
}

inline void dumpNodes(const Graph &graph, std::ostream &os) {
  os << "Nodes:\n";
  for (const auto &node : graph.nodes()) {
    os << "  Node name: " << node.name() << "\n";
    os << "    Opcode: " << node.opcode() << "\n";
    os << "    Inputs: [";
    for (size_t i = 0; i < node.inputs().size(); ++i) {
      os << node.inputs()[i];
      if (i < node.inputs().size() - 1) {
        os << ", ";
      }
    }
    os << "]\n";

    os << "    Outputs: [";
    for (size_t i = 0; i < node.outputs().size(); ++i) {
      os << node.outputs()[i];
      if (i < node.outputs().size() - 1) {
        os << ", ";
      }
    }
    os << "]\n";
  }
}

} // namespace tensor_compiler

#endif // UTILS_H
