#ifndef INCLUDE_NODE_HPP
#define INCLUDE_NODE_HPP

#include "attribute.hpp"
#include "onnx.pb.h"
#include "tensor.hpp"
#include <cstddef>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace tensor_compiler {

using id_t = std::size_t;

enum class Opcode {
    Add,
    Sub,
    Mul,
    Div,
    Relu,
};

class Node {
  private:
    id_t id_;
    // std::string op_type_; //opcode
    std::string name_; //

    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    std::vector<Attribute> attributes_;

  public:
    Node(id_t id, const std::string &name) : id_{id}, name_{name} {}

    void set_name(std::string name);

    void add_attribute(Attribute attr);
};

} // namespace tensor_compiler

#endif // INCLUDE_NODE_HPP
