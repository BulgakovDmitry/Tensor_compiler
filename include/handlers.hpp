#ifndef INCLUDE_HANDLERS_HPP
#define INCLUDE_HANDLERS_HPP

#include "structure/graph.hpp"
#include "structure/node.hpp"
#include "structure/tensor.hpp"

namespace tensor_compiler {

static int extract_elem_type(const onnx::ValueInfoProto &v) {
    if (!v.has_type() || !v.type().has_tensor_type())
        return onnx::TensorProto_DataType_UNDEFINED;
    return v.type().tensor_type().elem_type();
}

static Tensor::dim_type extract_dims(const onnx::ValueInfoProto &v) {
    Tensor::dim_type dims;
    if (!v.has_type() || !v.type().has_tensor_type() ||
        !v.type().tensor_type().has_shape())
        return dims;

    const auto &shape = v.type().tensor_type().shape();
    for (int i = 0; i < shape.dim_size(); ++i) {
        const auto &d = shape.dim(i);
        int64_t val =
            d.has_dim_value() ? static_cast<int64_t>(d.dim_value()) : -1;
        dims.Add(val);
    }
    return dims;
}

static std::string extract_tensor_bytes(const onnx::TensorProto &t) {
    if (!t.raw_data().empty())
        return t.raw_data();

    std::string out;

    if (t.float_data_size() > 0) {
        out.resize(sizeof(float) *
                   static_cast<std::size_t>(t.float_data_size()));
        std::memcpy(out.data(), t.float_data().data(), out.size());
        return out;
    }

    if (t.int64_data_size() > 0) {
        out.resize(sizeof(int64_t) *
                   static_cast<std::size_t>(t.int64_data_size()));
        std::memcpy(out.data(), t.int64_data().data(), out.size());
        return out;
    }

    return out;
}

inline Tensor handle_tensor(const onnx::TensorProto &t) {
    Tensor tensor{};
    tensor.set_name(t.name());
    tensor.set_dim(t.dims());
    tensor.set_type(t.data_type());
    tensor.set_data(extract_tensor_bytes(t));
    tensor.set_kind(Tensor_kind::constant);
    return tensor;
}

inline Tensor handle_tensor(const onnx::ValueInfoProto &t,
                            const Tensor_kind &type) {
    Tensor tensor{};
    tensor.set_name(t.name());
    tensor.set_dim(extract_dims(t));
    tensor.set_type(extract_elem_type(t));
    tensor.set_kind(type);
    return tensor;
}

inline void handle_node_ir_tensor(Graph &compute_graph, Node &new_node,
                                  const onnx::NodeProto &node,
                                  const std::string &name) {
    if (name.empty())
        return;
    if (!compute_graph.get_tensor(name)) {
        Tensor t{};
        t.set_name(name);
        t.set_kind(Tensor_kind::intermediate);
        compute_graph.add_tensor(std::move(t));
    }
}

inline Node handle_node(Graph &compute_graph, std::size_t &node_idx,
                        const onnx::NodeProto &node) {
    Node new_node{node.name(), node.op_type(), node_idx++};
    new_node.set_inputs(node.input());
    new_node.set_outputs(node.output());
    new_node.parse_attributes(node);

    for (const auto &name : new_node.get_inputs())
        handle_node_ir_tensor(compute_graph, new_node, node, name);

    for (const auto &name : new_node.get_outputs())
        handle_node_ir_tensor(compute_graph, new_node, node, name);

    return new_node;
}

} // namespace tensor_compiler

#endif // INCLUDE_HANDLERS_HPP
