#ifndef INCLUDE_TENSOR_HPP
#define INCLUDE_TENSOR_HPP

#include "onnx.pb.h"
#include <string>
#include <vector>

namespace tensor_compiler {

enum class Tensor_kind {
    unknown = 0,
    input,
    output,
    intermediate,
    constant,
};

class Tensor {
  public:
    using data_type = onnx::TensorProto_DataType;

  private:
    std::string name_;
    data_type type_ = data_type::TensorProto_DataType_UNDEFINED;
    Tensor_kind kind_ = Tensor_kind::unknown;
    std::vector<char> data_;
    std::vector<int64_t> shape_;
    std::size_t dim_;

  public:
    Tensor() = default;
    Tensor(const std::string &name, data_type &type, std::vector<int64_t> shape,
           std::vector<char> &data, Tensor_kind kind = Tensor_kind::unknown)
        : name_{name}, type_{type}, kind_{kind}, data_{data}, shape_{shape} {}

    const std::string &get_name() const;
    data_type get_type() const;
    Tensor_kind get_kind() const;
    const std::vector<char> &get_data() const;
    const std::vector<int64_t> &get_shape() const;
    const std::size_t get_dim() const;

    void set_name(const std::string &name);
    void set_type(data_type type);
    void set_kind(Tensor_kind kind);
    void set_data(const std::vector<char> &data);
    void set_shape(const std::vector<int64_t> &shape);
    void set_dim(std::size_t dim);

    bool is_constant() const;
};

// ----------------------------------------------------------------------------
// @section Implementations
// Implementation of tensor methods.
// ----------------------------------------------------------------------------
const std::string &Tensor::get_name() const { return name_; }
Tensor::data_type Tensor::get_type() const { return type_; }
const std::vector<char> &Tensor::get_data() const { return data_; }
const std::vector<int64_t> &Tensor::get_shape() const { return shape_; }
Tensor_kind Tensor::get_kind() const { return kind_; }
const std::size_t Tensor::get_dim() const { return dim_; }

void Tensor::set_name(const std::string &name) { name_ = name; }
void Tensor::set_type(data_type type) { type_ = type; }
void Tensor::set_kind(Tensor_kind kind) { kind_ = kind; }
void Tensor::set_data(const std::vector<char> &data) { data_ = data; }
void Tensor::set_shape(const std::vector<int64_t> &shape) { shape_ = shape; }
void Tensor::set_dim(std::size_t dim) { dim_ = dim; }

bool Tensor::is_constant() const { return kind_ == Tensor_kind::constant; }

} // namespace tensor_compiler

#endif // INCLUDE_TENSOR_HPP