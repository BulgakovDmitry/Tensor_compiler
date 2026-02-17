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

    const std::string &get_name() const;
    data_type get_type() const;
    const std::vector<char> &get_data() const;
    const std::vector<int64_t> &get_shape() const;
    bool is_constant() const;
};


// ----------------------------------------------------------------------------
// @section Implementations
// Implementation of node methods.
// ----------------------------------------------------------------------------
const std::string &Tensor::get_name() const { return name_; }
Tensor::data_type Tensor::get_type() const { return type_; }
const std::vector<char> &Tensor::get_data() const { return data_; }
const std::vector<int64_t> &Tensor::get_shape() const { return shape_; }

bool Tensor::is_constant() const { return is_constant_; }


} // namespace tensor_compiler

#endif // INCLUDE_TENSOR_HPP