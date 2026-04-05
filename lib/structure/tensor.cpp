#include "structure/tensor.h"

namespace tensor_compiler {

// ----------------------------------------------------------------------------
// @section Implementations
// Implementation of tensor methods.
// ----------------------------------------------------------------------------
Tensor Tensor::create(const std::string &name,
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

const std::string &Tensor::get_name() const { return name_; }
int Tensor::get_type() const { return type_; }
const std::string &Tensor::get_data() const { return data_; }
const std::vector<int64_t> &Tensor::get_shape() const { return shape_; }
Tensor_kind Tensor::get_kind() const { return kind_; }
const dim_type Tensor::get_dim() const { return dim_; }

void Tensor::set_name(const std::string &name) { name_ = name; }
void Tensor::set_type(const int type) { type_ = type; }
void Tensor::set_kind(Tensor_kind kind) { kind_ = kind; }
void Tensor::set_data(const std::string &data) { data_ = data; }
void Tensor::set_shape(const std::vector<int64_t> &shape) {
    shape_ = shape;
}
void Tensor::set_dim(const dim_type dim) { dim_ = dim; }

bool Tensor::is_constant() const {
    return kind_ == Tensor_kind::constant;
}

} // namespace tensor_compiler
