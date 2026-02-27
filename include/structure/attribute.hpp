#ifndef INCLUDE_ATTRIBUTE_HPP
#define INCLUDE_ATTRIBUTE_HPP

#include <string>
#include <variant>
#include <vector>

namespace tensor_compiler {

/**
 * @brief Represents an attribute of a node.
 *
 * An attribute holds a name and a value which can be of several types:
 * float, int64_t, string, vector<float>, vector<int64_t>. This class is a
 * simple wrapper around a variant.
 */
class Attribute {
  public:
    using AttrValue = std::variant<float, int64_t, std::string,
                                   std::vector<float>, std::vector<int64_t>>;

  private:
    std::string name_;

    AttrValue value;

  public:
    Attribute() = default;

    /**
     * @brief Construct a new Attribute object.
     * @param name Attribute name.
     * @param value Attribute value.
     */
    Attribute(const std::string &name, const AttrValue &value)
        : name_{name}, value{value} {}

    /**
     * @brief Get the attribute name.
     * @return const reference to name string.
     */
    const std::string &get_name() const { return name_; }

    /**
    * @brief Get the attribute value.
    * @return const reference to AttrValue variant.
    */
    const AttrValue &get_value() const { return value; }

    /**
     * @brief Set the attribute value.
     * @param new_value New variant value.
     */
    void set_value(const AttrValue &new_value) { value = new_value; }
};

} // namespace tensor_compiler

#endif // INCLUDE_ATTRIBUTE_HPP
