find_package(Threads REQUIRED)
find_package(GTest REQUIRED)
include(GoogleTest)

set(SRC_LIST
    src/tensor.cpp
    src/attributes.cpp
    src/node.cpp
    src/graph.cpp
    ../../../lib/structure/Tensor.cpp
    ../../../lib/structure/Graph.cpp
    ../../../lib/structure/Node.cpp
)

add_executable(structure ${SRC_LIST})

target_include_directories(structure PRIVATE ${CMAKE_BINARY_DIR}/onnx_generated)

target_link_libraries(structure
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        tensor_compiler::headers
        onnx_proto
)

gtest_discover_tests(structure
    PROPERTIES LABELS "unit"
)
