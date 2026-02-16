#ifndef INCLUDE_DRIVER_HPP
#define INCLUDE_DRIVER_HPP

#include "onnx.pb.h"
#include <fstream>
#include <google/protobuf/io/zero_copy_stream_impl.h>

namespace tensor_compiler {

inline int driver(const std::string &model_onnx) {
    onnx::ModelProto model;
    std::fstream input(model_onnx, std::ios::in | std::ios::binary);

    if (!model.ParseFromIstream(&input)) {
        std::cerr << "Failed to parse ONNX model." << '\n';
        return -1;
    }

  const auto &graph = model.graph();
  std::cout << "Graph '" << graph.name() << "' loaded." << '\n';
  std::cout << "Number of nodes: " << graph.node_size() << '\n';

  for (auto init : graph.initializer()) {
    //init.
  }

    return 0;
}

} // namespace tensor_compiler

#endif // INCLUDE_DRIVER_HPP
