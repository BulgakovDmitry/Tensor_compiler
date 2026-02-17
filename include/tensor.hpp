#ifndef INCLUDE_TENSOR_HPP
#define INCLUDE_TENSOR_HPP

#include "onnx.pb.h"
#include <string>
#include <vector>

namespace tensor_compiler {

class Tensor {
  private:
    using data_type = onnx::TensorProto_DataType;

    std::string name_;
    data_type type_;
    bool is_constant_;
    std::vector<char> data_;
    std::vector<int64_t> shape_;

  public:
    Tensor(const std::string &name, data_type &type, std::vector<int64_t> shape,
           std::vector<char> &data, bool is_constant = false)
        : name_{name}, type_{type}, is_constant_{is_constant}, data_{data},
          shape_{shape} {}

    // getters and setters (to be implemented)
};

} // namespace tensor_compiler

#endif // INCLUDE_TENSOR_HPP