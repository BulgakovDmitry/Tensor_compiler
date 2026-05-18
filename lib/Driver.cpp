#include "Driver.h"
#include "Codegen/Codegen.h"
#include "GraphDump/DumpPathGen.h"
#include "GraphDump/GraphvizDumper.h"
#include "Lowering/MLIRToLLVM.h"
#include "Lowering/LLVMToASM.h"
#include "Lowering/LLVMToLLVMIR.h"
#include "onnx.pb.h"
#include "Structure/Graph.h"
#include <cstring>
#include <fstream>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <iostream>
#include <string>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

// CLI arguments for controlling compilation
namespace {

llvm::cl::opt<std::string> inputFile(
    llvm::cl::Positional,
    llvm::cl::desc("<ONNX model file>"),
    llvm::cl::Required);

llvm::cl::opt<std::string> emitTarget(
    "emit",
    llvm::cl::desc("Compilation stage: mlir, llvm, or asm"),
    llvm::cl::init("asm")
);

constexpr const char *outputFilename = "model.s";

llvm::cl::opt<std::string> targetTriple(
    "mtriple",
    llvm::cl::desc("Target triple for codegen (default: x86_64-pc-linux-gnu)"),
    llvm::cl::init("x86_64-pc-linux-gnu")
);

llvm::cl::opt<unsigned> optLevel(
    "O",
    llvm::cl::desc("Optimization level (0-3)"),
    llvm::cl::init(2)
);

} // anonymous namespace

namespace tensor_compiler {

int driver(int argc, char *argv[]) {
    llvm::cl::ParseCommandLineOptions(argc, argv, "Tensor Compiler\n");

    onnx::ModelProto model;
    std::fstream input(inputFile, std::ios::in | std::ios::binary);
    if (!input.good())
        throw std::runtime_error(
            "Failed to open ONNX model file: " + inputFile + "\n");

    if (!model.ParseFromIstream(&input))
        throw std::runtime_error("Failed to parse ONNX model.\n");

    Graph compute_graph{model.graph()};

#ifdef GRAPH_DUMP
    // ____________GRAPH DUMP___________ //
    const auto paths = tensor_compiler::makeDumpPaths();
    const std::string gv_file = paths.gv.string();
    const std::string svg_file = paths.svg.string();
    // dot dump/dump.gv -Tsvg -o dump/dump.svg

    std::ofstream gv(gv_file);
    if (!gv)
        throw std::runtime_error("unable to open gv file\n");

    GraphvizDumper::dump(compute_graph, gv);
#endif

    mlir::MLIRContext context;

    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::math::MathDialect>();
    registry.insert<mlir::cf::ControlFlowDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::bufferization::BufferizationDialect>();
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
        registry);
    mlir::cf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
    mlir::registerBuiltinDialectTranslation(context);
    mlir::registerLLVMDialectTranslation(context);

    tensor_compiler::Codegen codegen{context};
    auto mlirModule = codegen.generate(compute_graph);
    if (!mlirModule) {
        llvm::errs() << "Error: Codegen returned null module\n";
        return 1;
    }

    using namespace mlir;

    auto moduleOp = mlirModule->getOperation();

    // Target triple for System V AMD64 ABI
    moduleOp->setAttr("llvm.target_triple",
        StringAttr::get(&context, targetTriple));

    if (targetTriple == "x86_64-pc-linux-gnu" ||
        targetTriple.starts_with("x86_64-")) {
        moduleOp->setAttr("llvm.data_layout",
            StringAttr::get(&context,
                "e-m:e-p270:32:32-p271:32:32-p272:64:64-"
                "i64:64-f80:128-n8:16:32:64-S128"));
    }

    if (emitTarget == "mlir") {
        mlirModule->print(llvm::outs());
        llvm::outs() << "\n";
        return 0;
    }

    if (mlir::failed(MLIRToLLVM(context, mlirModule))) {
        llvm::errs() << "Error: MLIR to LLVM lowering failed\n";
        return 1;
    }

    llvm::LLVMContext llvmCtx;
    auto llvmModule = LLVMToLLVMIR(llvmCtx, mlirModule);
    if (!llvmModule) {
        llvm::errs() << "Error: Failed to export LLVM IR\n";
        return 1;
    }

    if (emitTarget == "llvm") {
        llvmModule->print(llvm::outs(), nullptr);
        return 0;
    }

    if (emitTarget == "asm") {
        std::error_code ec;
        llvm::raw_fd_ostream asmStream(
            outputFilename,
            ec,
            llvm::sys::fs::OF_None);

        if (mlir::failed(generateAssembly(
                llvmModule.get(), targetTriple, optLevel, asmStream))) {
            llvm::errs() << "Error: Assembly generation failed\n";
            return 1;
        }
        return 0;
    }

    llvm::errs() << "Unknown emit target: " << emitTarget << "\n";
    return 1;
}

} // namespace tensor_compiler
