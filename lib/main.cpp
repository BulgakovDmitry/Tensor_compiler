#include "driver.h"
#include <iostream>

int main(int argc, char *argv[]) {
    try {
        tensor_compiler::driver(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "unknown error\n";
        return 2;
    }

    return 0;
}
