#include <random>


#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
//#include "mlir/Dialect/Index/IR/IndexEnums.h.inc"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
// #include "mlir/Dialect/ArmSVE/ArmSVEDialect.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
// #include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <map>


enum value_type
{
    SCALAR,
    TENSOR,
    VECTOR,
    MEMREF
};
struct CheckSum {

    std::map<std::string, std::map<value_type, std::vector<mlir::Value>>> checksumValuePool; // example: checksumValuePool["i16"][SCALAR] = %1 (value)

    std::map<value_type, std::map<std::string, mlir::func::FuncOp>> funcOpPool; // example: funcOpPool[TENSOR]["i16"] = tensor_i16(funcOp)

    static mlir::Operation *mainfunc, *idx0op, *idx1op;

    void addValue(mlir::Value val);

    mlir::Value calCheckSumAll(mlir::Location loc, mlir::OpBuilder &builder);

    mlir::Value calCheckSumByType(mlir::Location loc, mlir::OpBuilder &builder, mlir::Type type);

    mlir::Value calTensorCheckSumByType(mlir::Location loc, mlir::OpBuilder &builder,
                                       mlir::Type type, std::string typestr,mlir::Value lstval);

    mlir::Value calMemrefCheckSumByType(mlir::Location loc, mlir::OpBuilder &builder,
                                       mlir::Type type, std::string typestr,mlir::Value lstval);

    mlir::Value calVectorCheckSumByType(mlir::Location loc, mlir::OpBuilder &builder,
                                       mlir::Type type, std::string typestr,mlir::Value lstval);

    mlir::func::FuncOp calCheckSumTensor(mlir::Type type);

    mlir::func::FuncOp calCheckSumMemref(mlir::Type type);

    inline static std::string getValueTypeStr(mlir::Value v);
    
    inline static std::string getTypeStr(mlir::Type type);

};

