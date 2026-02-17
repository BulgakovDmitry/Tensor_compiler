#ifndef INCLUDE_GRAPH_HPP
#define INCLUDE_GRAPH_HPP

#include "onnx.pb.h"
#include <cstddef>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <string>
#include <variant>
#include <vector>
#include <unordered_map>
#include "tensor.hpp"
#include "attribute.hpp"
#include "node.hpp"

namespace tensor_compiler {

class Graph {
  private:
    using T_map = std::unordered_map<std::string, Tensor>;

    std::string name_;        
    T_map tensors_;         
    std::vector<Node> nodes_; 
    std::vector<std::string> inputs_;  
    std::vector<std::string> outputs_; 

  public:
    void set_name(std::string name);


    void add_tensor(Tensor tensor);

    void add_node(Node node);

    void set_inputs(std::vector<std::string> inputs);

    void set_outputs(std::vector<std::string> outputs);

    const Tensor *get_tensor(const std::string &name) const;

    void dump(std::ostream &os) const;

    // getters (to be implemented)
};

} // namespace tensor_compiler

#endif // INCLUDE_GRAPH_HPP