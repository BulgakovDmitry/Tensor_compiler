/**
 * @file node.hpp
 * @brief Defines the core data structures for representing a computation graph
 *        of a neural network, including tensors, attributes, nodes, and the
 * graph itself.
 */

#ifndef INCLUDE_NODE_HPP
#define INCLUDE_NODE_HPP

#include "onnx.pb.h"
#include <cstddef>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <string>
#include <variant>

namespace tensor_compiler {

/**
 * @brief Type alias for a unique identifier used to reference nodes.
 */
using id_t = std::size_t;

/**
 * @brief Represents a tensor in the computation graph.
 *
 * A tensor holds metadata such as name, data type, shape, and optionally
 * constant data (if it comes from an ONNX initializer). Tensors are uniquely
 * identified by their name within a graph.
 */
class Tensor {
  private:
    using data_type = onnx::TensorProto_DataType;

    std::string name_; ///< Unique name of the tensor.
    data_type type_;   ///< Data type (using ONNX enum for compatibility).
    bool is_constant_; ///< True if the tensor is a constant (weights, etc.).
    std::vector<char> data_;     ///< Raw byte data for constant tensors.
    std::vector<int64_t> shape_; ///< Dimensions of the tensor.

  public:
    /**
     * @brief Constructs a new Tensor object.
     *
     * @param name Unique name of the tensor.
     * @param type Data type of the tensor (ONNX enum).
     * @param shape Vector of dimension sizes.
     * @param data Raw byte data (for constants; may be empty for
     * non‑constants).
     * @param is_constant Flag indicating whether this tensor holds constant
     * data.
     */
    Tensor(const std::string &name, data_type &type, std::vector<int64_t> shape,
           std::vector<char> &data, bool is_constant = false)
        : name_{name}, type_{type}, is_constant_{is_constant}, data_{data},
          shape_{shape} {}

    // getters and setters (to be implemented)
};

/**
 * @brief Represents an attribute of an operation node.
 *
 * Attributes are named parameters that configure the behavior of an operation
 * (e.g., kernel_shape for Conv). They can hold values of various types.
 */
class Attribute {
  private:
    std::string name_; ///< Name of the attribute.

    /**
     * @brief Variant type capable of holding the actual attribute value.
     *
     * Supports the most common attribute types found in ONNX.
     */
    using AttrValue = std::variant<float, int64_t, std::string,
                                   std::vector<float>, std::vector<int64_t>>;

    AttrValue data_; ///< Stored attribute value.

  public:
    // constructors and getters for different types (to be implemented)
};

/**
 * @brief Represents a node (operation) in the computation graph.
 *
 * A node corresponds to an ONNX operator. It has a type (e.g., "Conv"),
 * a list of input and output tensor names, and a collection of attributes.
 */
class Node {
  private:
    id_t id_;
    std::string op_type_;
    std::string name_;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    std::vector<Attribute> attributes_; ///< List of operation attributes.

  public:
    /**
     * @brief Constructs a new Node object.
     *
     * @param id Unique identifier for the node.
     * @param name Optional name (can be empty).
     */
    Node(id_t id, const std::string &name) : id_{id}, name_{name} {}

    /**
     * @brief Sets or updates the node's name.
     * @param name New name.
     */
    void set_name(std::string name);

    /**
     * @brief Adds an attribute to the node.
     * @param attr The attribute to add.
     */
    void add_attribute(Attribute attr);
};

/**
 * @brief Represents the entire computation graph.
 *
 * The graph owns all tensors and nodes, and maintains lists of input and
 * output tensor names. It provides methods to build and query the graph.
 */
class Graph {
  private:
    using T_map = std::unordered_map<std::string, Tensor>;

    std::string name_;        ///< Name of the graph (from ONNX).
    T_map tensors_;           ///< Map from tensor name to Tensor object.
    std::vector<Node> nodes_; ///< List of nodes in topological order.
    std::vector<std::string> inputs_; ///< Names of input tensors, not constants
    std::vector<std::string> outputs_; ///< Names of output tensors.

  public:
    /**
     * @brief Sets the graph name.
     * @param name New graph name.
     */
    void set_name(std::string name);

    /**
     * @brief Adds a tensor to the graph.
     * @param tensor The tensor to add.
     */
    void add_tensor(Tensor tensor);

    /**
     * @brief Adds a node to the graph.
     * @param node The node to add.
     */
    void add_node(Node node);

    /**
     * @brief Sets the list of input tensor names.
     * @param inputs Vector of input names.
     */
    void set_inputs(std::vector<std::string> inputs);

    /**
     * @brief Sets the list of output tensor names.
     * @param outputs Vector of output names.
     */
    void set_outputs(std::vector<std::string> outputs);

    /**
     * @brief Retrieves a tensor by its name.
     * @param name The tensor name.
     * @return Pointer to the tensor, or nullptr if not found.
     */
    const Tensor *get_tensor(const std::string &name) const;

    void dump(std::ostream &os) const;

    // getters (to be implemented)
};

} // namespace tensor_compiler

#endif // INCLUDE_NODE_HPP
