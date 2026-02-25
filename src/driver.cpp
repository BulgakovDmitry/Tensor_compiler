#include "driver.hpp"
#include "dump_path_gen.hpp"
#include "graphviz_dumper.hpp"
#include "handlers.hpp"
#include "onnx.pb.h"
#include "structure/graph.hpp"
#include "structure/tensor.hpp"
#include "utils.hpp"
#include <cstring>
#include <fstream>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <iostream>
#include <string>

namespace tensor_compiler {

Graph build_compute_graph(const onnx::GraphProto &graph) {
    Graph compute_graph(graph.name());

    for (const auto &initializer : graph.initializer()) {
        auto tensor = handle_tensor(initializer);
        compute_graph.add_tensor(std::move(tensor));
    }

    for (const auto &input : graph.input()) {
        auto tensor = handle_tensor(input, Tensor_kind::input);
        compute_graph.add_tensor(std::move(tensor));
        compute_graph.add_input(input.name());
    }

    std::size_t node_idx = 0;
    for (const auto &node : graph.node()) {
        auto new_node = handle_node(compute_graph, node_idx, node);
        compute_graph.add_node(std::move(new_node));
    }

    for (const auto &output : graph.output()) {
        Tensor tensor = handle_tensor(output, Tensor_kind::output);
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

    auto compute_graph = build_compute_graph(g);

    // debug zone -----------------------------------------
    topological_dump(compute_graph, std::cout);
    node_dump(compute_graph, std::cout);
    // ----------------------------------------------------

#ifdef GRAPH_DUMP
    // ____________GRAPH DUMP___________ //
    const auto paths = tensor_compiler::make_dump_paths();
    const std::string gv_file = paths.gv.string();
    const std::string svg_file = paths.svg.string();
    // dot dump/dump.gv -Tsvg -o dump/dump.svg

    std::ofstream gv(gv_file);
    if (!gv) {
        throw std::runtime_error("unable to open gv file\n");
    }
    Graphviz_dumper::dump(compute_graph, gv);
#endif

    return 0;
}

} // namespace tensor_compiler
