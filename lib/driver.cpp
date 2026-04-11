#include "Driver.h"
#include "codegen/Codegen.h"
#include "graph_dump/DumpPathGen.h"
#include "graph_dump/GraphvizDumper.h"
#include "onnx.pb.h"
#include "structure/Graph.h"
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
    const auto paths = tensor_compiler::makeDumpPaths();
    const std::string gv_file = paths.gv.string();
    const std::string svg_file = paths.svg.string();
    // dot dump/dump.gv -Tsvg -o dump/dump.svg

    std::ofstream gv(gv_file);
    if (!gv)
        throw std::runtime_error("unable to open gv file\n");

    GraphvizDumper::dump(compute_graph, gv);
#endif

    // tensor_compiler::Codegen codegen{};
    // auto module = codegen.generate(compute_graph);

    // module->print(llvm::outs());
    // llvm::outs() << "\n";

    return 0;
}

} // namespace tensor_compiler
