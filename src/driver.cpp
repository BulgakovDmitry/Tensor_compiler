#include "driver.hpp"
#include "graph.hpp"
#include "onnx.pb.h"
#include <fstream>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <iostream>

tensor_compiler::Graph
tensor_compiler::build_compute_graph(const onnx::GraphProto &graph) {
    Graph compute_graph(graph.name());

    for (const auto &initializer : graph.initializer()) {
        Tensor tensor{};

        tensor.set_name(initializer.name());
        tensor.set_dim(initializer.dims());
        tensor.set_type(initializer.data_type());
        tensor.set_data(initializer.raw_data());
        tensor.set_kind(Tensor_kind::constant);

        compute_graph.add_tensor(tensor);
    }

    auto extract_dims = [](const onnx::ValueInfoProto &v) -> Tensor::type_dim {
        Tensor::type_dim dims;
        if (!v.has_type() || !v.type().has_tensor_type() ||
            !v.type().tensor_type().has_shape())
            return dims;

        const auto &shape = v.type().tensor_type().shape();
        for (int i = 0; i < shape.dim_size(); ++i) {
            const auto &d = shape.dim(i);
            long int val = d.has_dim_value()
                               ? static_cast<long int>(d.dim_value())
                               : -1; // неизвестно -> -1
            dims.Add(val);
        }
        return dims;
    };

    for (const auto &input : graph.input()) {
        Tensor tensor{};

        tensor.set_name(input.name());
        tensor.set_dim(extract_dims(input));
        tensor.set_kind(Tensor_kind::input);

        compute_graph.add_tensor(tensor);
        compute_graph.add_input(tensor.get_name());
    }

    for (const auto &node : graph.node()) {
        Node new_node{node.name(), node.op_type()};

        new_node.set_inputs(node.input());
        new_node.set_outputs(node.output());
        new_node.parse_attributes(node);

        compute_graph.add_node(new_node);
    }

    for (const auto &output : graph.output()) {
        Tensor tensor{};
        tensor.set_name(output.name());
        tensor.set_dim(extract_dims(output));
        tensor.set_kind(Tensor_kind::output);

        compute_graph.add_tensor(tensor);
        compute_graph.add_output(tensor.get_name());
    }

    return compute_graph;
}

int tensor_compiler::driver(const std::string &model_onnx) {
    onnx::ModelProto model;
    std::fstream input(model_onnx, std::ios::in | std::ios::binary);

    if (!model.ParseFromIstream(&input)) {
        std::cerr << "Failed to parse ONNX model." << '\n';
        return -1;
    }

    const auto &graph = model.graph();
    std::cout << "Graph '" << graph.name() << "' loaded." << '\n';
    std::cout << "Number of nodes: " << graph.node_size() << '\n';

    auto compute_graph = build_compute_graph(graph);

    return 0;
}
