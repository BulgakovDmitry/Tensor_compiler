#include "onnx/onnx_pb.h"
#include <fstream>
#include <google/protobuf/io/zero_copy_stream_impl.h>

int main() {
  onnx::ModelProto model;
  std::fstream input("mnist-12.onnx", std::ios::in | std::ios::binary);

  if (!model.ParseFromIstream(&input)) {
    std::cerr << "Failed to parse ONNX model." << std::endl;
    return -1;
  }

  const auto &graph = model.graph();
  std::cout << "Graph '" << graph.name() << "' loaded." << std::endl;
  std::cout << "Number of nodes: " << graph.node_size() << std::endl;

  return 0;
}
