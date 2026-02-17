cmake_minimum_required(VERSION 3.16)

project(tensor_compiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(ONNX_PROTO "${CMAKE_CURRENT_SOURCE_DIR}/onnx/onnx.proto")
if(NOT EXISTS ${ONNX_PROTO})
    message(FATAL_ERROR "ONNX proto file not found: ${ONNX_PROTO}")
endif()

find_package(Protobuf REQUIRED)

set(PROTO_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/onnx_generated")

file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

protobuf_generate_cpp(ONNX_GENERATED_SRC ONNX_GENERATED_HDR
    ${ONNX_PROTO}
    GENERATE_EXTENSIONS .pb.h .pb.cc
    PROTOC_OUT_DIR ${PROTO_GEN_DIR}
)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/driver.cpp
    ${ONNX_GENERATED_SRC}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROTO_GEN_DIR}
    ${PROTOBUF_INCLUDE_DIRS}
    include
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${PROTOBUF_LIBRARIES}
)

# enable_testing()
# add_subdirectory(tests)
