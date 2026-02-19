#include "driver.hpp"
#include "graph.hpp"
#include "onnx.pb.h"
#include <cstring>
#include <fstream>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <iostream>
#include <string>

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

Graph build_compute_graph(const onnx::GraphProto &graph) {
    Graph compute_graph(graph.name());

    for (const auto &initializer : graph.initializer()) {
        Tensor tensor{};
        tensor.set_name(initializer.name());
        tensor.set_dim(initializer.dims());
        tensor.set_type(initializer.data_type());
        tensor.set_data(extract_tensor_bytes(initializer));
        tensor.set_kind(Tensor_kind::constant);
        compute_graph.add_tensor(std::move(tensor));
    }

    for (const auto &input : graph.input()) {
        Tensor tensor{};
        tensor.set_name(input.name());
        tensor.set_dim(extract_dims(input));
        tensor.set_type(extract_elem_type(input));
        tensor.set_kind(Tensor_kind::input);

        compute_graph.add_tensor(std::move(tensor));
        compute_graph.add_input(input.name());
    }

    std::size_t node_idx = 0;
    for (const auto &node : graph.node()) {
        Node new_node{node.name(), node.op_type(), node_idx++};
        new_node.set_inputs(node.input());
        new_node.set_outputs(node.output());
        new_node.parse_attributes(node);

        for (const auto &in_name : new_node.get_inputs()) {
            if (in_name.empty())
                continue;
            if (!compute_graph.get_tensor(in_name)) {
                Tensor t{};
                t.set_name(in_name);
                t.set_kind(Tensor_kind::intermediate);
                compute_graph.add_tensor(std::move(t));
            }
        }

        for (const auto &out_name : new_node.get_outputs()) {
            if (out_name.empty())
                continue;
            if (!compute_graph.get_tensor(out_name)) {
                Tensor t{};
                t.set_name(out_name);
                t.set_kind(Tensor_kind::intermediate);
                compute_graph.add_tensor(std::move(t));
            }
        }

        compute_graph.add_node(std::move(new_node));
    }

    for (const auto &output : graph.output()) {
        Tensor tensor{};
        tensor.set_name(output.name());
        tensor.set_dim(extract_dims(output));
        tensor.set_type(extract_elem_type(output));
        tensor.set_kind(Tensor_kind::output);

        compute_graph.add_tensor(std::move(tensor));
        compute_graph.add_output(output.name());
    }

    return compute_graph;
}

int driver(const std::string &model_onnx) {
    onnx::ModelProto model;
    std::fstream input(model_onnx, std::ios::in | std::ios::binary);

    if (!input.good()) {
        std::cerr << "Failed to open ONNX model file: " << model_onnx << "\n";
        return -1;
    }

    if (!model.ParseFromIstream(&input)) {
        std::cerr << "Failed to parse ONNX model.\n";
        return -1;
    }

    const auto &g = model.graph();
    std::cout << "Graph '" << g.name() << "' loaded.\n";
    std::cout << "Number of nodes: " << g.node_size() << "\n";

    auto compute_graph = build_compute_graph(g);

    // std::cout << "Compute graph tensors: " <<
    // compute_graph.get_tensors().size() << "\n"; std::cout << "Compute graph
    // nodes:   " << compute_graph.get_nodes().size() << "\n";

    return 0;
}

} // namespace tensor_compiler
