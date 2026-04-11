#include "Lowering/LLVMToASMLowering.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/CodeGen/CommandFlags.h"

#include <memory>
#include <string>

using namespace mlir;

namespace tensor_compiler {

LogicalResult generateAssembly(llvm::Module *llvmModule,
                                      const std::string &triple,
                                      unsigned optLevel,
                                      llvm::raw_pwrite_stream &os) {
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  std::string error;
  auto targetTriple = triple.empty() ? llvmModule->getTargetTriple() : triple;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(
      targetTriple, error);
  if (!target) {
    llvm::errs() << "Error: " << error << "\n";
    return failure();
  }

  llvm::TargetOptions opt;
  auto RM = std::optional<llvm::Reloc::Model>(llvm::Reloc::PIC_);

  std::unique_ptr<llvm::TargetMachine> TM(
      target->createTargetMachine(targetTriple,
                                /*CPU=*/"",
                                /*Features=*/"",
                                opt,
                                RM));
  if (!TM) {
    llvm::errs() << "Error: Could not create TargetMachine\n";
    return failure();
  }

  llvmModule->setDataLayout(TM->createDataLayout());
  llvmModule->setTargetTriple(TM->getTargetTriple().str());

  llvm::legacy::PassManager PM;
  llvm::CodeGenFileType fileType = llvm::CodeGenFileType::AssemblyFile;

  if (TM->addPassesToEmitFile(PM, os, nullptr, fileType)) {
    llvm::errs() << "Error: Target does not support assembly emission\n";
    return failure();
  }

  PM.run(*llvmModule);
  return success();
}

} // namespace tensor_compiler
