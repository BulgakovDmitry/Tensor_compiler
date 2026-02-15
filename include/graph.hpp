#ifndef INCLUDE_GRAPH_HPP
#define INCLUDE_GRAPH_HPP

#include "node.hpp"

namespace tensor_compiler {

class Graph {
private:
  std::list<Node *> nodes_;

public:
  void add_node(Node *node) { nodes_.push_back(node); }
};

} // namespace tensor_compiler

#endif // INCLUDE_GRAPH_HPP