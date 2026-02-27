find_package(Threads REQUIRED)
find_package(GTest REQUIRED)
include(GoogleTest)

set(SRC_LIST
    src/tensor.cpp
    src/attributes.cpp
)

add_executable(structure ${SRC_LIST})

target_link_libraries(structure
    PRIVATE 
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        tensor_compiler::headers
        ${PROTOBUF_LIBRARIES}
)

gtest_discover_tests(structure
    PROPERTIES LABELS "unit"
)  