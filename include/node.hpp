#ifndef INCLUDE_NODE_HPP
#define INCLUDE_NODE_HPP

#include <cstddef>
#include <list>
#include <string>

namespace tensor_compiler {

using id_t = std::size_t;

class Node {
private:
    id_t id_;
    std::string name_;

    std::list<Node*> inputs_;
    std::list<Node*> outputs_;

public:
    Node(id_t id, const std::string& name) : id_(id), name_(name) {}

};

} // namespace tensor_compiler

#endif // INCLUDE_NODE_HPP