#include "driver.hpp"
#include <iostream>

int main(int argc, char *argv[]) {
    if (argc > 1)
        try {
            tensor_compiler::driver(argv[1]);
        } catch (const std::exception &e) {
            std::cerr << "error: " << e.what() << "\n";
            return 1;
        } catch (...) {
            std::cerr << "unknown error\n";
            return 2;
        }
    else
        std::cerr << "Error: Usage: " << argv[0] << " <onnx file>" << std::endl;
}
