#ifndef INCLUDE_TENSOR_H
#define INCLUDE_TENSOR_H

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

/// @brief Represents a tensor in the computation graph.
///
/// Stores tensor metadata: name, data type, shape, raw data, and its kind.
/// The raw data is stored as a string (binary blob). For float tensors,
/// a convenience factory method Tensor::create() is provided.
class Tensor final {
private:
  std::string name_;
  int type_ = data_type::TensorProto_DataType_UNDEFINED;
  Tensor_kind kind_ = Tensor_kind::unknown;

  std::string data_;
  std::vector<int64_t> shape_;
  dim_type dim_;

public:
  Tensor() = default;

  /// @brief Construct a new Tensor object.
  /// @param name Tensor name.
  /// @param type Data type (onnx::TensorProto_DataType).
  /// @param shape Vector of dimensions.
  /// @param data Raw data as string (binary).
  /// @param kind Tensor kind (default unknown).
  Tensor(const std::string &name, data_type type, std::vector<int64_t> shape,
         const std::string &data, Tensor_kind kind = Tensor_kind::unknown)
      : name_{name}, type_{type}, kind_{kind}, data_{data}, shape_{shape} {}

  /// @brief Create a float tensor from a vector of floats.
  /// Convenience method that packs the float data into a binary string.
  /// @param name Tensor name.
  /// @param shape Tensor shape.
  /// @param data Vector of float values.
  /// @param kind Tensor kind.
  /// @return Tensor object.
  static Tensor create(const std::string &name,
                       const std::vector<int64_t> &shape,
                       const std::vector<float> &data, const Tensor_kind &kind);

  /// @brief Get the tensor name.
  /// @return const reference to name string.
  const std::string &name() const;

  /// @brief Get the data type.
  /// @return int (onnx::TensorProto_DataType value).
  int type() const;

  /// @brief Get the tensor kind.
  /// @return Tensor_kind.
  Tensor_kind kind() const;

  /// @brief Get the raw data as a string.
  /// @return const reference to data string.
  const std::string &data() const;

  /// @brief Get the tensor shape.
  /// @return const reference to vector of dimensions.
  const std::vector<int64_t> &shape() const;

  /// @brief Get the protobuf dim field (may be unused).
  /// @return const dim_type.
  const dim_type dim() const;

  /// @brief Set the tensor name.
  /// @param name New name.
  void setName(const std::string &name);

  /// @brief Set the data type.
  /// @param type onnx::TensorProto_DataType value.
  void setType(const int type);

  /// @brief Set the tensor kind.
  /// @param kind Tensor_kind.
  void setKind(Tensor_kind kind);

  /// @brief Set the raw data.
  /// @param data Binary string.
  void setData(const std::string &data);

  /// @brief Set the tensor shape.
  /// @param shape Vector of dimensions.
  void setShape(const std::vector<int64_t> &shape);

  /// @brief Set the protobuf dim field.
  /// @param dim dim_type.
  void setDim(const dim_type dim);

  /// @brief Check if the tensor is a constant (initializer).
  /// @return true if kind_ == Tensor_kind::constant.
  bool isConstant() const;
};

} // namespace tensor_compiler

#endif // INCLUDE_TENSOR_H
