#include "driver.hpp"
#include <iostream>

int main(int argc, char *argv[]) {
    std::cout << "# Tensor Compiler" << std::endl;
    std::cout << "# (c) RTCupid, BulgakovDmitry, 2026" << std::endl;
    if (argc > 1)
        tensor_compiler::driver(argv[1]);
    else
        std::cerr << "Error: Usage: " << argv[0] << " <onnx file>" << std::endl;
}
