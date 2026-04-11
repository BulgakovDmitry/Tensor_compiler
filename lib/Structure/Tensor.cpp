#include "Structure/Tensor.h"

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

const std::string &Tensor::name() const { return name_; }
int Tensor::type() const { return type_; }
const std::string &Tensor::data() const { return data_; }
const std::vector<int64_t> &Tensor::shape() const { return shape_; }
Tensor_kind Tensor::kind() const { return kind_; }
const dim_type Tensor::dim() const { return dim_; }

void Tensor::setName(const std::string &name) { name_ = name; }
void Tensor::setType(const int type) { type_ = type; }
void Tensor::setKind(Tensor_kind kind) { kind_ = kind; }
void Tensor::setData(const std::string &data) { data_ = data; }
void Tensor::setShape(const std::vector<int64_t> &shape) {
    shape_ = shape;
    dim_.Clear();
    for (int64_t d : shape_) {
        dim_.Add(d);
    }
}

void Tensor::setDim(const dim_type dim) {
    dim_ = dim;
    shape_.assign(dim_.begin(), dim_.end());
}

bool Tensor::isConstant() const {
    return kind_ == Tensor_kind::constant;
}

} // namespace tensor_compiler
