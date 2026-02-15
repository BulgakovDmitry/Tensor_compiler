cmake_minimum_required(VERSION 3.16)

project(tensor_compiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Protobuf REQUIRED)

set(ONNX_GENERATED_SRC ${CMAKE_CURRENT_SOURCE_DIR}/onnx/onnx.pb.cc)
set(ONNX_GENERATED_HDR ${CMAKE_CURRENT_SOURCE_DIR}/onnx/onnx.pb.h)

add_executable(${PROJECT_NAME}
    src/main.cpp
    ${ONNX_GENERATED_SRC}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROTOBUF_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${PROTOBUF_LIBRARIES}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    include
)

# enable_testing()
# add_subdirectory(tests)
