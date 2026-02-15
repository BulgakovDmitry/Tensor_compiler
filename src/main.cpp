#include "driver.hpp"

int main(int argc, char* argv[]) {

    if (argc > 1)
        blab::driver(argv[1]);

    return 0;
}
