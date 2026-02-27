cmake_minimum_required(VERSION 3.16)

project(tensor_compiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------------------------------------------------
# Dependencies
# -------------------------------------------------------------------
find_package(Protobuf REQUIRED)
find_package(Threads REQUIRED)

# -------------------------------------------------------------------
# Generate ONNX protobuf
# -------------------------------------------------------------------
set(ONNX_PROTO "${CMAKE_CURRENT_SOURCE_DIR}/onnx/onnx.proto")

if(NOT EXISTS "${ONNX_PROTO}")
    message(FATAL_ERROR "ONNX proto file not found: ${ONNX_PROTO}")
endif()

set(PROTO_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/onnx_generated")

file(MAKE_DIRECTORY "${PROTO_GEN_DIR}")

protobuf_generate_cpp(
    ONNX_GENERATED_SRC 
    ONNX_GENERATED_HDR
    ${ONNX_PROTO}
    PROTOC_OUT_DIR ${PROTO_GEN_DIR}
    GENERATE_EXTENSIONS .pb.h .pb.cc
)

add_library(onnx_proto STATIC ${ONNX_GENERATED_SRC})
target_include_directories(onnx_proto 
    PUBLIC 
        "${PROTO_GEN_DIR}"
        ${PROTOBUF_INCLUDE_DIRS}
)
target_link_libraries(onnx_proto 
    PUBLIC 
        ${PROTOBUF_LIBRARIES}
)

# -------------------------------------------------------------------
# Project headers (interface library)
# -------------------------------------------------------------------
add_library(tensor_compiler_headers INTERFACE)
target_include_directories(tensor_compiler_headers
    INTERFACE 
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/structure
)
add_library(tensor_compiler::headers ALIAS tensor_compiler_headers)

# -------------------------------------------------------------------
# Main executable
# -------------------------------------------------------------------
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/driver.cpp
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        tensor_compiler::headers
        onnx_proto
        Threads::Threads
)

# -------------------------------------------------------------------
# Graph dump
# -------------------------------------------------------------------
option(GRAPH_DUMP "Enable Graphviz dump" OFF)
if (GRAPH_DUMP)
    target_compile_definitions(${PROJECT_NAME} PRIVATE GRAPH_DUMP)
endif()

# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
