#ifndef INCLUDE_HANDLERS_H
#define INCLUDE_HANDLERS_H

#include "onnx.pb.h"

namespace tensor_compiler {

inline int extract_elem_type(const onnx::ValueInfoProto &v) {
  if (!v.has_type() || !v.type().has_tensor_type())
    return onnx::TensorProto_DataType_UNDEFINED;
  return v.type().tensor_type().elem_type();
}

inline std::vector<int64_t> extract_dims(const onnx::ValueInfoProto &v) {
    std::vector<int64_t> dims;

    if (!v.has_type() || !v.type().has_tensor_type() ||
        !v.type().tensor_type().has_shape()) {
        return dims;
    }

    const auto &shape = v.type().tensor_type().shape();
    dims.reserve(shape.dim_size());

    for (int i = 0; i < shape.dim_size(); ++i) {
        const auto &d = shape.dim(i);
        int64_t val = d.has_dim_value() ? static_cast<int64_t>(d.dim_value()) : -1;
        dims.push_back(val);
    }

    return dims;
}

inline std::string extract_tensor_bytes(const onnx::TensorProto &t) {
  if (!t.raw_data().empty())
    return t.raw_data();

  std::string out;

  if (t.float_data_size() > 0) {
    out.resize(sizeof(float) * static_cast<std::size_t>(t.float_data_size()));
    std::memcpy(out.data(), t.float_data().data(), out.size());
    return out;
  }

  if (t.int64_data_size() > 0) {
    out.resize(sizeof(int64_t) * static_cast<std::size_t>(t.int64_data_size()));
    std::memcpy(out.data(), t.int64_data().data(), out.size());
    return out;
  }

  return out;
}

} // namespace tensor_compiler

#endif // INCLUDE_HANDLERS_H
