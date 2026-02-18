/**
 * @file driver.hpp
 * @brief Provides a high-level driver function to load and parse an ONNX model.
 */

#ifndef INCLUDE_DRIVER_HPP
#define INCLUDE_DRIVER_HPP

#include <string>
#include "graph.hpp"

namespace tensor_compiler {

/**
 * @brief Parses an ONNX model graph to compute graph
 *
 * @param graph
 * @return Graph
 */
Graph build_compute_graph(const auto &graph);

/**
 * @brief Loads and parses an ONNX model file
 *
 * @param model_onnx Path to the ONNX model file (e.g., "model.onnx").
 * @return 0 on success, -1 if the file cannot be parsed or opened.
 */
int driver(const std::string &model_onnx);

} // namespace tensor_compiler

#endif // INCLUDE_DRIVER_HPP
