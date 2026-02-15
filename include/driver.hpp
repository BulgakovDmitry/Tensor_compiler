#ifndef INCLUDE_DRIVER_HPP
#define INCLUDE_DRIVER_HPP

#include "onnx.pb.h"
#include <fstream>
#include <google/protobuf/io/zero_copy_stream_impl.h>

namespace blab {

inline int driver() {
  onnx::ModelProto model;
  std::fstream input("mnist-12.onnx", std::ios::in | std::ios::binary);

  if (!model.ParseFromIstream(&input)) {
    std::cerr << "Failed to parse ONNX model." << '\n';
    return -1;
  }

  const auto &graph = model.graph();
  std::cout << "Graph '" << graph.name() << "' loaded." << '\n';
  std::cout << "Number of nodes: " << graph.node_size() << '\n';

  return 0;
}

} // namespace blab

#endif // INCLUDE_DRIVER_HPP
