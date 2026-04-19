cmake_minimum_required(VERSION 3.16)

project(tensor-compiler LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------------------------------------------------
# Dependencies
# -------------------------------------------------------------------
find_package(Protobuf REQUIRED)
find_package(Threads REQUIRED)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

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
        ${CMAKE_SOURCE_DIR}/include/Structure
)
add_library(tensor_compiler::headers ALIAS tensor_compiler_headers)

# -------------------------------------------------------------------
# Tensor Compiler executable
# -------------------------------------------------------------------
add_executable(${PROJECT_NAME}
    tools/tensor-compiler.cpp
    lib/Driver.cpp
    lib/Structure/Tensor.cpp
    lib/Structure/Graph.cpp
    lib/Structure/Node.cpp
    lib/Codegen/Codegen.cpp
    lib/Lowering/MLIRToLLVM.cpp
    lib/Lowering/LLVMToASM.cpp
    lib/Lowering/LLVMToLLVMIR.cpp
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        tensor_compiler::headers
        onnx_proto
        Threads::Threads

        # Core MLIR
        MLIRIR
        MLIRSupport

        # Dialects
        MLIRFuncDialect
        MLIRArithDialect
        MLIRTensorDialect
        MLIRLinalgDialect
        MLIRSCFDialect
        MLIRMemRefDialect
        MLIRMathDialect
        MLIRControlFlowDialect
        MLIRLLVMDialect
        MLIRBufferizationDialect
        MLIRBufferizationTransforms
        MLIRBufferizationToMemRef

        # Conversions
        MLIRPass
        MLIRTransforms
        MLIRFuncToLLVM
        MLIRArithToLLVM
        MLIRMemRefToLLVM
        MLIRControlFlowToLLVM
        MLIRMathToLLVM
        MLIRLinalgTransforms
        MLIRSCFToControlFlow
        MLIRConvertToLLVMPass

        # LLVM → LLVM IR
        MLIRTargetLLVMIRExport
        MLIRBuiltinToLLVMIRTranslation
        MLIRLLVMToLLVMIRTranslation

        # LLVM libs
        LLVMCore
        LLVMTarget
        LLVMOption
        LLVMX86Info
        LLVMX86Desc
        LLVMX86AsmParser
        LLVMX86CodeGen
        LLVMMC
        LLVMMCParser
        LLVMAnalysis
        LLVMTransformUtils
        LLVMSupport
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    ${LLVM_DEFINITIONS}
)

# -------------------------------------------------------------------
# Tensor Driver library
# -------------------------------------------------------------------
add_library(tensor_driver SHARED tensorDriver/lib/TensorDrvImpl.cpp)

target_include_directories(tensor_driver PUBLIC tensorDriver/include)

# -------------------------------------------------------------------
# Inference App executable
# -------------------------------------------------------------------
add_executable(inference-app tools/inference-app.cpp)

target_link_libraries(inference-app PRIVATE tensor_driver dl)

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
