#include "driver.hpp"
#include "codegen/codegen.hpp"
#include "graph_dump/dump_path_gen.hpp"
#include "graph_dump/graphviz_dumper.hpp"
#include "onnx.pb.h"
#include "structure/graph.hpp"
#include <cstring>
#include <fstream>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <iostream>
#include <string>

namespace tensor_compiler {

int driver(const std::string &model_onnx) {
    onnx::ModelProto model;
    std::fstream input(model_onnx, std::ios::in | std::ios::binary);
    if (!input.good())
        throw std::runtime_error(
            "Failed to open ONNX model file: " + model_onnx + "\n");

    if (!model.ParseFromIstream(&input))
        throw std::runtime_error("Failed to parse ONNX model.\n");

    Graph compute_graph{model.graph()};

#ifdef GRAPH_DUMP
    // ____________GRAPH DUMP___________ //
    const auto paths = tensor_compiler::make_dump_paths();
    const std::string gv_file = paths.gv.string();
    const std::string svg_file = paths.svg.string();
    // dot dump/dump.gv -Tsvg -o dump/dump.svg

    std::ofstream gv(gv_file);
    if (!gv)
        throw std::runtime_error("unable to open gv file\n");

    Graphviz_dumper::dump(compute_graph, gv);
#endif

    return 0;
}

} // namespace tensor_compiler
