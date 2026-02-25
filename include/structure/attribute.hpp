#ifndef INCLUDE_ATTRIBUTE_HPP
#define INCLUDE_ATTRIBUTE_HPP

#include <string>
#include <variant>
#include <vector>

namespace tensor_compiler {

class Attribute {
  public:
    using AttrValue = std::variant<float, int64_t, std::string,
                                   std::vector<float>, std::vector<int64_t>>;

  private:
    std::string name_;

    AttrValue value;

  public:
    Attribute() = default;
    Attribute(const std::string &name, const AttrValue &value)
        : name_{name}, value{value} {}

    const std::string &get_name() const { return name_; }
    const AttrValue &get_value() const { return value; }
    void set_value(const AttrValue &new_value) { value = new_value; }
};

} // namespace tensor_compiler

#endif // INCLUDE_ATTRIBUTE_HPP