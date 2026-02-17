#ifndef INCLUDE_ATTRIBUTE_HPP
#define INCLUDE_ATTRIBUTE_HPP

#include <string>
#include <variant>
#include <vector>

namespace tensor_compiler {

class Attribute {
  private:
    std::string name_; 

    using AttrValue = std::variant<float, int64_t, std::string,
                                   std::vector<float>, std::vector<int64_t>>;

    AttrValue data_;

  public:
    // constructors and getters for different types (to be implemented)
};

} // namespace tensor_compiler

#endif // INCLUDE_ATTRIBUTE_HPP