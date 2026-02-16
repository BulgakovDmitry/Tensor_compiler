#ifndef INCLUDE_NODE_HPP
#define INCLUDE_NODE_HPP

#include "onnx.pb.h"
#include <cstddef>
#include <list>
#include <string>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <variant>

namespace tensor_compiler {

enum class DataType {
    UNDEFINED,
    FLOAT,
    UINT8,
    INT8,
    INT16,
    INT32,
    INT64,
    // ...
};

using id_t = std::size_t;

class Tensor {
  private:
    std::string name_;
    onnx::TensorProto_DataType type_;
    std::vector<int64_t> shape_;
    bool is_constant_;
    std::vector<char> data_;

  public:
    Tensor(const std::string &name, onnx::TensorProto_DataType &type,
           std::vector<int64_t> shape, std::vector<char> &data, bool is_constant = false)
        : name_{name}, type_{type}, shape_{shape}, is_constant_{is_constant},
          data_{data} {}

    // getters
};

class Attribute {
  private:
    std::string name_;
    enum class Type { FLOAT, INT, STRING, TENSOR, FLOATS, INTS /* ... */ };

    using AttrValue = std::variant<float, int64_t, std::string,
                                    std::vector<float>,
                                    std::vector<int64_t>>;

    AttrValue data_;

  public:

  // getters and constructors for different types
};

class Node {
  private:
    id_t id_;
    std::string name_;

    std::list<Node *> inputs_;
    std::list<Node *> outputs_;
    std::vector<Attribute> attributes_;

  public:
    Node(id_t id, const std::string &name) : id_{id}, name_{name} {}

    void set_name(std::string name);
    void add_attribute(Attribute attr);
};

class Graph {
  private:
    std::string name_;
    std::unordered_map<std::string, Tensor> tensors_;
    std::vector<Node> nodes_;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
public:
    void set_name(std::string name);
    void add_tensor(Tensor tensor);
    void add_node(Node node);
    void set_inputs(std::vector<std::string> inputs);
    void set_outputs(std::vector<std::string> outputs);

    const Tensor* get_tensor(const std::string& name) const;
    void dump(std::ostream& os) const;

    // getters
};

} // namespace tensor_compiler

#endif // INCLUDE_NODE_HPP
