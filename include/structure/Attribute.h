#ifndef INCLUDE_ATTRIBUTE_H
#define INCLUDE_ATTRIBUTE_H

#include <string>
#include <variant>
#include <vector>

namespace tensor_compiler {

/// @brief Represents an attribute of a node.
///
/// An attribute holds a name and a value which can be of several types:
/// float, int64_t, string, vector<float>, vector<int64_t>. This class is a
/// simple wrapper around a variant.
class Attribute final {
public:
  using AttrValue =
      std::variant<float, int64_t, std::string, std::vector<float>,
                   std::vector<int64_t>, bool, std::vector<std::string>>;

private:
  std::string name_;

  AttrValue value_;

public:
  Attribute() = default;

  /// @brief Construct a new Attribute object.
  /// @param name Attribute name.
  /// @param value Attribute value.
  Attribute(const std::string &name, const AttrValue &value)
      : name_{name}, value_{value} {}

  /// @brief Get the attribute name.
  /// @return const reference to name string.
  const std::string &name() const { return name_; }

  /// @brief Get the attribute value.
  /// @return const reference to AttrValue variant.
  const AttrValue &value() const { return value_; }

  /// @brief Set the attribute value.
  /// @param newValue New variant value.
  void setValue(const AttrValue &newValue) { value_ = newValue; }
};

} // namespace tensor_compiler

#endif // INCLUDE_ATTRIBUTE_H
