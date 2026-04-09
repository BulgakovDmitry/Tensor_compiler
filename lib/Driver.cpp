#include "Driver.h"
#include "codegen/Codegen.h"
#include "graph_dump/DumpPathGen.h"
#include "graph_dump/GraphvizDumper.h"
#include "lowering/MLIRToLLVMLowering.h"
#include "lowering/LLVMToASMLowering.h"
#include "onnx.pb.h"
#include "structure/Graph.h"
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

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
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
    llvm::cl::init("llvm")
);

llvm::cl::opt<std::string> outputFilename(
    "o",
    llvm::cl::desc("Output filename (for assembly)"),
    llvm::cl::value_desc("filename"),
    llvm::cl::init("-")  // "-" = stdout
);

llvm::cl::opt<std::string> targetTriple(
    "mtriple",
    llvm::cl::desc("Target triple for code generation"),
    llvm::cl::init("")
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
    const auto paths = tensor_compiler::make_dump_paths();
    const std::string gv_file = paths.gv.string();
    const std::string svg_file = paths.svg.string();
    // dot dump/dump.gv -Tsvg -o dump/dump.svg

    std::ofstream gv(gv_file);
    if (!gv)
        throw std::runtime_error("unable to open gv file\n");

    Graphviz_dumper::dump(compute_graph, gv);
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
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
    mlir::registerBuiltinDialectTranslation(context);
    mlir::registerLLVMDialectTranslation(context);

    MLIRToLLVMLowering lowering(context);

    tensor_compiler::Codegen codegen{context};
    auto mlirModule = codegen.generate(compute_graph);
    if (!mlirModule) {
        llvm::errs() << "Error: Codegen returned null module\n";
        return 1;
    }

    if (emitTarget == "mlir") {
        mlirModule->print(llvm::outs());
        llvm::outs() << "\n";
        return 0;
    }

    if (mlir::failed(lowering.lower(std::move(mlirModule)))) {
        llvm::errs() << "Error: MLIR to LLVM lowering failed\n";
        return 1;
    }

    auto llvmModule = lowering.exportToLLVM();
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
            outputFilename == "-" ? "" : outputFilename.getValue(),
            ec,
            llvm::sys::fs::OF_None);

        if (ec) {
            llvm::errs() << "Error: Could not open output file '"
                        << outputFilename.getValue() << "': " << ec.message() << "\n";
            return 1;
        }

        llvm::raw_pwrite_stream &outStream =
            (outputFilename == "-") ? llvm::outs() : asmStream;

        if (mlir::failed(generateAssembly(
                llvmModule.get(), targetTriple, optLevel, outStream))) {
            llvm::errs() << "Error: Assembly generation failed\n";
            return 1;
        }

        if (outputFilename != "-") {
            llvm::outs() << "Assembly written to " << outputFilename << "\n";
        }
        return 0;
    }

    llvm::errs() << "Unknown emit target: " << emitTarget << "\n";
    return 1;
}

} // namespace tensor_compiler
