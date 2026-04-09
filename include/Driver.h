/// @file driver.hpp
/// @brief Provides a high-level driver function to load and parse an ONNX
/// model.

#ifndef INCLUDE_DRIVER_H
#define INCLUDE_DRIVER_H

namespace tensor_compiler {

/// @brief Loads, parses and compile an ONNX model file
int driver(int argc, char *argv[]);

} // namespace tensor_compiler

#endif // INCLUDE_DRIVER_H
