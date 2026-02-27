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

using data_type = onnx::TensorProto_DataType;
using dim_type = google::protobuf::RepeatedField<int64_t>;

class Tensor {
  private:
    std::string name_;
    int type_ = data_type::TensorProto_DataType_UNDEFINED;
    Tensor_kind kind_ = Tensor_kind::unknown;

    std::string data_;
    std::vector<int64_t> shape_;
    dim_type dim_;

  public:
    Tensor() = default;
    Tensor(const std::string &name, data_type type, std::vector<int64_t> shape,
           const std::string &data, Tensor_kind kind = Tensor_kind::unknown)
        : name_{name}, type_{type}, kind_{kind}, data_{data}, shape_{shape} {}

    static Tensor create(const std::string &name,
                         const std::vector<int64_t> &shape,
                         const std::vector<float> &data,
                         const Tensor_kind &kind);

    const std::string &get_name() const;
    const int get_type() const;
    Tensor_kind get_kind() const;
    const std::string &get_data() const;
    const std::vector<int64_t> &get_shape() const;
    const dim_type get_dim() const;

    void set_name(const std::string &name);
    void set_type(const int type);
    void set_kind(Tensor_kind kind);
    void set_data(const std::string &data);
    void set_shape(const std::vector<int64_t> &shape);
    void set_dim(const dim_type dim);

    bool is_constant() const;
};

// ----------------------------------------------------------------------------
// @section Implementations
// Implementation of tensor methods.
// ----------------------------------------------------------------------------
inline Tensor Tensor::create(const std::string &name,
                             const std::vector<int64_t> &shape,
                             const std::vector<float> &data,
                             const Tensor_kind &kind) {
    std::string raw_data;
    if (!data.empty()) {
        raw_data.assign(reinterpret_cast<const char *>(data.data()),
                        data.size() * sizeof(float));
    }

    return Tensor(name, data_type::TensorProto_DataType_FLOAT, shape, raw_data,
                  kind);
}

inline const std::string &Tensor::get_name() const { return name_; }
inline const int Tensor::get_type() const { return type_; }
inline const std::string &Tensor::get_data() const { return data_; }
inline const std::vector<int64_t> &Tensor::get_shape() const { return shape_; }
inline Tensor_kind Tensor::get_kind() const { return kind_; }
inline const dim_type Tensor::get_dim() const { return dim_; }

inline void Tensor::set_name(const std::string &name) { name_ = name; }
inline void Tensor::set_type(const int type) { type_ = type; }
inline void Tensor::set_kind(Tensor_kind kind) { kind_ = kind; }
inline void Tensor::set_data(const std::string &data) { data_ = data; }
inline void Tensor::set_shape(const std::vector<int64_t> &shape) {
    shape_ = shape;
}
inline void Tensor::set_dim(const dim_type dim) { dim_ = dim; }

inline bool Tensor::is_constant() const {
    return kind_ == Tensor_kind::constant;
}

} // namespace tensor_compiler

#endif // INCLUDE_TENSOR_HPP
