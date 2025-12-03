#include "MutationUtil.h"
#include "NodeGenerator.h"
#include "TypeBase.h"
#include <algorithm>
#include <random>

// Operator: tosa::AbsOp
// AbsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
// ::mlir::Type output, ::mlir::Value input1)
OpGenerator tosaAbsGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src = operandCandidates[rollIdx(operandCandidates.size())];
    auto output = src.getType();
    auto input1 = src;
    mlir::Value res_tensor =
        mlir::tosa::AbsOp::create(builder, loc, output, input1);
    mb->add2Pool(res_tensor);
    return true;
  };
}

// Operator: mlir::tosa::AddOp
// Documentation:
// input1	tensor of number values
// input2	tensor of number values
// output	tensor of number values
// Builder Definition File: ./include/mlir/Dialect/Tosa/IR/TosaOps.td
// Selected Builder Function:
// void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState
// &odsState,
//                   ::mlir::Type output, ::mlir::Value input1, ::mlir::Value
//                   input2);
OpGenerator tosaAddGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(src2Candidates.begin(),
                            mb->valuePool.pool[src1TyStr].begin(),
                            mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(generateRankedTensor(
          builder, loc, mlir::dyn_cast<RankedTensorType>(src1.getType()), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto input1 = src1;
    auto input2 = src2;
    auto output = src1.getType();

    mlir::Value res_tensor =
        mlir::tosa::AddOp::create(builder, loc, output, input1, input2);
    mb->add2Pool(res_tensor);
    return true;
  };
}

// Operator: mlir::tosa::ApplyScaleOp
// Documentation:
// double_round	::mlir::BoolAttr	bool attribute
// value	signless-integer-like
// multiplier	signless-integer-like
// shift	signless-integer-8-bit-like
// output	signless-integer-like
// Builder Definition File: ./include/mlir/Dialect/Tosa/IR/TosaOps.td
// Selected Builder Function:
// void ApplyScaleOp::build(::mlir::OpBuilder &odsBuilder,
// ::mlir::OperationState &odsState,
//                          ::mlir::Type output, ::mlir::Value value,
//                          ::mlir::Value multiplier, ::mlir::Value shift, bool
//                          double_round)
OpGenerator tosaApplyScaleGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (candidates.empty()) {
      candidates.push_back(generateInteger(builder, loc, builder.getI32Type()));
    }
    auto value_vt = candidates[rollIdx(candidates.size())];
    auto multiplier_vt = candidates[rollIdx(candidates.size())];
    auto shift_vt = generateInteger(builder, loc, builder.getI8Type());
    auto value = value_vt;
    auto multiplier = multiplier_vt;
    auto shift = shift_vt;
    bool double_round = rollIdx(2);
    auto output = mlir::dyn_cast<IntegerType>(value_vt.getType());

    auto mode_idx = rollIdx(3);
    tosa::RoundingMode mode;
    if (mode_idx == 0) {
      mode = tosa::RoundingMode::SINGLE_ROUND;
    } else if (mode_idx == 1) {
      mode = tosa::RoundingMode::INEXACT_ROUND;
    } else if (mode_idx == 2) {
      mode = tosa::RoundingMode::DOUBLE_ROUND;
    } else {
      assert(false);
    }
    tosa::RoundingModeAttr attr =
        tosa::RoundingModeAttr::get(builder.getContext(), mode);

    mlir::Value res = mlir::tosa::ApplyScaleOp::create(
        builder, loc, output, value, multiplier, shift, attr);
    mb->add2Pool(res);
    return true;
  };
}

// Operator:tosa.argmax (mlir::tosa::ArgMaxOp)
// export PATH=/data/hqy/mlirsmith-dev/build/tools/llvm-symbolizer:$PATH

// axis	::mlir::IntegerAttr	64-bit signless integer attribute
// input	tensor of number values
// output	tensor of number values

// include/mlir/Dialect/Tosa/IR/TosaOps.td
// void ArgMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState
// &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) void
// ArgMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState
// &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr
// axis)
OpGenerator tosaArgMaxGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    const int dim_ub = 32;

    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
        [&](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          if (auto t = mlir::dyn_cast<RankedTensorType>(v.getType())) {
            bool rankCond = t.getRank() > 0 && t.getRank() <= 4;
            bool elementTyCond = t.getElementType().isIntOrIndex();
            return rankCond && elementTyCond;
          }
          return false;
        },
        &operandCandidates);
    if (operandCandidates.empty()) {
      SmallVector<int64_t> new_shape;
      new_shape.push_back(1);
      for (int i = 0; i < rollIdx(3); i++) {
        new_shape.push_back(rollIdx(dim_ub) + 1);
      }
      operandCandidates.push_back(generateRankedTensor(
          builder, loc,
          RankedTensorType::get(new_shape, randomIntType(builder.getContext())),
          mb));
    }

    auto src = operandCandidates[rollIdx(operandCandidates.size())];
    auto input = src;
    uint64_t axis =
        rollIdx(mlir::dyn_cast<RankedTensorType>(input.getType()).getRank());

    SmallVector<int64_t> shape;
    auto srcTy = mlir::dyn_cast<RankedTensorType>(src.getType());

    if (srcTy.hasStaticShape()) {
      for (uint64_t i = 0; i < srcTy.getRank(); ++i) {
        if (i == axis)
          continue;
        shape.push_back(ShapedType::kDynamic);
      }
    } else {
      for (uint64_t i = 0; i < srcTy.getRank(); ++i) {
        if (i == axis)
          continue;
        if (srcTy.isDynamicDim(i)) {
          shape.push_back(rollIdx(dim_ub) + 1);
        } else {
          shape.push_back(srcTy.getDimSize(i));
        }
      }
    }

    auto output = RankedTensorType::get(shape, srcTy.getElementType());

    mlir::Value res =
        mlir::tosa::ArgMaxOp::create(builder, loc, output, input, axis);
    mb->add2Pool(res);
    return true;
  };
}

/*
tosa.arithmetic_right_shift (mlir::tosa::ArithmeticRightShiftOp)
Elementwise arithmetic right shift of input1 by the amount specified in input2.
Axis of size 1 will be broadcast, as necessary.

round	::mlir::BoolAttr	bool attribute
input1	tensor of number values
input2	tensor of number values
output	tensor of number values
void ArithmeticRightShiftOp::build(::mlir::OpBuilder &odsBuilder,
::mlir::OperationState &odsState,
::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2, bool round)
*/
OpGenerator tosaArithmeticRightShiftGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(src2Candidates.begin(),
                            mb->valuePool.pool[src1TyStr].begin(),
                            mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(generateRankedTensor(
          builder, loc, mlir::dyn_cast<RankedTensorType>(src1.getType()), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto input1 = src1;
    auto input2 = src2;
    auto output = mlir::dyn_cast<RankedTensorType>(src1.getType());
    bool round = rollIdx(2);

    mlir::Value res = mlir::tosa::ArithmeticRightShiftOp::create(
        builder, loc, output, input1, input2, round);
    mb->add2Pool(res);
    return true;
  };
}

/*tosa.avg_pool2d (mlir::tosa::AvgPool2dOp) ¶
Performs max pooling on the input.
This performs an average pooling over the given input tensor. A sliding window
of size given by is passed over the input tensor, with the mean value being
placed in the output tensor.

kernel	::mlir::DenseI64ArrayAttr	i64 dense array attribute with exactly 2
elements stride	::mlir::DenseI64ArrayAttr	i64 dense array attribute with
exactly 2 elements pad	::mlir::DenseI64ArrayAttr	i64 dense array
attribute with exactly 4 elements acc_type	::mlir::TypeAttr	type
attribute of 32-bit signless integer or 32-bit signed integer or 16-bit float or
32-bit float quantization_info	mlir::tosa::UnaryOpQuantizationAttr
Attribute for UnaryOp quantization information. input	unranked tensor of
number values or 4D tensor of number values output	unranked tensor of
number values or 4D tensor of number values void
AvgPool2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState
&odsState, ::mlir::Type outputType, ::mlir::Value input,
::mlir::DenseI64ArrayAttr kernel, ::mlir::DenseI64ArrayAttr s tride,
::mlir::DenseI64ArrayAttr pad)
*/

/*tosa.bitwise_and (mlir::tosa::BitwiseAndOp)
Bitwise AND operator
Elementwise bitwise AND of input1 and input2. Axis of size 1 will be broadcast
as necessary.

input1	tensor of number values
input2	tensor of number values
output	tensor of number values

void BitwiseAndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState
&odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2)
*/
OpGenerator tosaBitwiseAndGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(src2Candidates.begin(),
                            mb->valuePool.pool[src1TyStr].begin(),
                            mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(generateRankedTensor(
          builder, loc, mlir::dyn_cast<RankedTensorType>(src1.getType()), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto input1 = src1;
    auto input2 = src2;
    auto output = mlir::dyn_cast<RankedTensorType>(src1.getType());
    mlir::Value res =
        mlir::tosa::BitwiseAndOp::create(builder, loc, output, input1, input2);
    mb->add2Pool(res);
    return true;
  };
}

/*tosa.bitwise_not (mlir::tosa::BitwiseNotOp)

input1	tensor of number values
output	tensor of number values

*/
OpGenerator tosaBitwiseNotGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src = operandCandidates[rollIdx(operandCandidates.size())];
    auto input = src;
    auto output = mlir::dyn_cast<RankedTensorType>(src.getType());
    mlir::Value res =
        mlir::tosa::BitwiseNotOp::create(builder, loc, output, input);
    mb->add2Pool(res);
    return true;
  };
}

/*tosa.bitwise_or (mlir::tosa::BitwiseOrOp)
input1	tensor of number values
input2	tensor of number values
output	tensor of number values

void BitwiseOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState
&odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2)
*/
OpGenerator tosaBitwiseOrGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(src2Candidates.begin(),
                            mb->valuePool.pool[src1TyStr].begin(),
                            mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(generateRankedTensor(
          builder, loc, mlir::dyn_cast<RankedTensorType>(src1.getType()), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto input1 = src1;
    auto input2 = src2;
    auto output = mlir::dyn_cast<RankedTensorType>(src1.getType());
    mlir::Value res =
        mlir::tosa::BitwiseOrOp::create(builder, loc, output, input1, input2);
    mb->add2Pool(res);
    return true;
  };
}

/*tosa.bitwise_xor (mlir::tosa::BitwiseXorOp)
input1	tensor of number values
input2	tensor of number values
output	tensor of number values
void BitwiseXorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState
&odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2)
*/
OpGenerator tosaBitwiseXorGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(src2Candidates.begin(),
                            mb->valuePool.pool[src1TyStr].begin(),
                            mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(generateRankedTensor(
          builder, loc, mlir::dyn_cast<RankedTensorType>(src1.getType()), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto input1 = src1;
    auto input2 = src2;
    auto output = mlir::dyn_cast<RankedTensorType>(src1.getType());
    mlir::Value res =
        mlir::tosa::BitwiseXorOp::create(builder, loc, output, input1, input2);
    mb->add2Pool(res);
    return true;
  };
}

/*tosa.cast (mlir::tosa::CastOp)
void CastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState
&odsState, ::mlir::Type output, ::mlir::Value input) Performs a set of
permissible cast operations

input:tensor of number_plus_f64 values
output	tensor of number_plus_f64 values
void CastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState
&odsState, ::mlir::Type output, ::mlir::Value input)
*/

OpGenerator tosaCastGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    const int dim_ub = 32;

    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
        [&](std::string s, mlir::Value v) -> bool {
          if (auto t = mlir::dyn_cast<TensorType>(v.getType())) {
            return t.getElementType() == builder.getI8Type();
          }
          return false;
        },
        &operandCandidates);
    if (operandCandidates.empty()) {
      SmallVector<int64_t> shape;

      for (int i = 0; i < rollIdx(dim_ub) + 1; i++) {
        shape.push_back(rollIdx(dim_ub) + 1);
      }
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, RankedTensorType::get(shape, builder.getI8Type()), mb));
    }
    auto src = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src;
    auto shape = mlir::dyn_cast<RankedTensorType>(src.getType()).getShape();
    auto output = RankedTensorType::get(shape, builder.getI16Type());

    mlir::Value res = mlir::tosa::CastOp::create(builder, loc, output, input);
    mb->add2Pool(res);
    return true;
  };
}

/*tosa.ceil (mlir::tosa::CeilOp)
:Elementwise ceiling operation
input1	tensor of number values
output	tensor of number values
void CeilOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState
&odsState, ::mlir::Type output, ::mlir::Value input1)
*/
OpGenerator tosaCeilGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src = operandCandidates[rollIdx(operandCandidates.size())];
    auto input = src;
    auto output = mlir::dyn_cast<RankedTensorType>(src.getType());
    mlir::Value res = mlir::tosa::CeilOp::create(builder, loc, output, input);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.clamp (mlir::tosa::ClampOp)
// void ClampOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState
// &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t min_int,
// uint64_t max_int, ::llvm::APFloat min _fp, ::llvm::APFloat max_fp)
//  min_int	::mlir::IntegerAttr	64-bit signless integer attribute
//  max_int	::mlir::IntegerAttr	64-bit signless integer attribute
//  min_fp	::mlir::FloatAttr	32-bit float attribute
//  max_fp	::mlir::FloatAttr	32-bit float attribute

// OpGenerator tosaClampGenerator() {
//   return [](OpBuilder &builder, Location loc, OpRegion &region) {
//     auto operandCandidates = region.pool.searchCandidatesFrom(
//       {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
//       emptyFilter());
//     if (operandCandidates.empty()) {
//         operandCandidates.push_back(
//           region.pool.generateRankedTensor(builder, loc,
//           randomRankedTensorType(builder.getContext())));
//     }
//     auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.clamp");

//     auto input = src1.val;
//     auto output = src1.type;
//     uint64_t max_int = ((unsigned long long)rand() << 32) | rand();
//     uint64_t min_int = (unsigned long long)UR(max_int+1);
//     // 创建一个随机数生成器
//     std::random_device rd;
//     std::mt19937 gen(rd());

//     // 定义随机数的分布范围
//     std::uniform_real_distribution<float> dist(0.0f, 1.0f);

//     // 生成随机数
//     ::llvm::APFloat min_fp = APFloat(::llvm::APFloat::IEEEsingle, dist(gen));
//     ::llvm::APFloat max_fp = APFloat(::llvm::APFloat::IEEEsingle, dist(gen));

//     auto res = mlir::tosa::ClampOp::create(builder, loc, output,
//     input,min_int, max_int, min_fp, max_fp); region.pool.addRankedTensor(
//       TypeValue(output.dyn_cast<RankedTensorType>(), res),
//       "tosa.clamp");
//     return res.getOperation();
//   };
// }

//  ::mlir::Type output, ::mlir::Value input1) {
// tosa.identity (mlir::tosa::IdentityOp)
OpGenerator tosaIdentityOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input1 = src1;
    auto output = src1.getType();

    mlir::Value res =
        mlir::tosa::IdentityOp::create(builder, loc, output, input1);
    mb->add2Pool(res);
    return true;
  };
}
// ::mlir::TypeRange output, ::mlir::Value cond, ::mlir::ValueRange inputs) {
// tosa.cond_if (mlir::tosa::IfOp)

// ::mlir::Type output, ::mlir::Value input1) {
// tosa.log (mlir::tosa::LogOp)
OpGenerator tosaLogOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input1 = src1;
    auto output = src1.getType();

    mlir::Value res = mlir::tosa::LogOp::create(builder, loc, output, input1);
    mb->add2Pool(res);
    return true;
  };
}

// ::mlir::Type z, ::mlir::Value input1, ::mlir::Value input2) {
// tosa.logical_and (mlir::tosa::LogicalAndOp)
OpGenerator tosaLogicalAndOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
        [&](std::string s, mlir::Value v) -> bool {
          if (!(ranked_tensor_filter(s, v)))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType())
                     .getElementType() == builder.getI1Type();
        },
        &operandCandidates);
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      auto predTy = RankedTensorType::get(newShape, builder.getI1Type());
      auto dest = generateRankedTensor(builder, loc, predTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(src2Candidates.begin(),
                            mb->valuePool.pool[src1TyStr].begin(),
                            mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(generateRankedTensor(
          builder, loc, mlir::dyn_cast<RankedTensorType>(src1.getType()), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto input1 = src1;
    auto input2 = src2;
    auto z = src1.getType();

    mlir::Value res_tensor =
        mlir::tosa::LogicalAndOp::create(builder, loc, z, input1, input2);
    mb->add2Pool(res_tensor);
    return true;
  };
}

// ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
// tosa.logical_left_shift (mlir::tosa::LogicalLeftShiftOp)
OpGenerator tosaLogicalLeftShiftOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(src2Candidates.begin(),
                            mb->valuePool.pool[src1TyStr].begin(),
                            mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(generateRankedTensor(
          builder, loc, mlir::dyn_cast<RankedTensorType>(src1.getType()), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto input1 = src1;
    auto input2 = src2;
    auto output = src1.getType();

    mlir::Value res_tensor = mlir::tosa::LogicalLeftShiftOp::create(
        builder, loc, output, input1, input2);
    mb->add2Pool(res_tensor);
    return true;
  };
}
// ::mlir::Type output, ::mlir::Value input1) {
// tosa.logical_not (mlir::tosa::LogicalNotOp)
OpGenerator tosaLogicalNotOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
        [&](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType())
                     .getElementType() == builder.getI1Type();
        },
        &operandCandidates);
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      auto predTy = RankedTensorType::get(newShape, builder.getI1Type());
      auto dest = generateRankedTensor(builder, loc, predTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input1 = src1;
    auto output = src1.getType();

    mlir::Value res =
        mlir::tosa::LogicalNotOp::create(builder, loc, output, input1);
    mb->add2Pool(res);
    return true;
  };
}

// ::mlir::Type z, ::mlir::Value input1, ::mlir::Value input2) {
// tosa.logical_or (mlir::tosa::LogicalOrOp)
OpGenerator tosaLogicalOrOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
        [&](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType())
                     .getElementType() == builder.getI1Type();
        },
        &operandCandidates);
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      auto predTy = RankedTensorType::get(newShape, builder.getI1Type());
      auto dest = generateRankedTensor(builder, loc, predTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(src2Candidates.begin(),
                            mb->valuePool.pool[src1TyStr].begin(),
                            mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(generateRankedTensor(
          builder, loc, mlir::dyn_cast<RankedTensorType>(src1.getType()), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto input1 = src1;
    auto input2 = src2;
    auto z = src1.getType();

    mlir::Value res_tensor =
        mlir::tosa::LogicalOrOp::create(builder, loc, z, input1, input2);

    mb->add2Pool(res_tensor);
    return true;
  };
}
// ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
// tosa.logical_right_shift (mlir::tosa::LogicalRightShiftOp)
OpGenerator tosaLogicalRightShiftOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(src2Candidates.begin(),
                            mb->valuePool.pool[src1TyStr].begin(),
                            mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(generateRankedTensor(
          builder, loc, mlir::dyn_cast<RankedTensorType>(src1.getType()), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto input1 = src1;
    auto input2 = src2;
    auto output = src1.getType();

    mlir::Value res_tensor = mlir::tosa::LogicalRightShiftOp::create(
        builder, loc, output, input1, input2);
    mb->add2Pool(res_tensor);
    return true;
  };
}
// ::mlir::Type z, ::mlir::Value input1, ::mlir::Value input2) {
// tosa.logical_xor (mlir::tosa::LogicalXorOp)
OpGenerator tosaLogicalXorOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
        [&](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType())
                     .getElementType() == builder.getI1Type();
        },
        &operandCandidates);
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      auto predTy = RankedTensorType::get(newShape, builder.getI1Type());
      auto dest = generateRankedTensor(builder, loc, predTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(src2Candidates.begin(),
                            mb->valuePool.pool[src1TyStr].begin(),
                            mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(generateRankedTensor(
          builder, loc, mlir::dyn_cast<RankedTensorType>(src1.getType()), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto input1 = src1;
    auto input2 = src2;
    auto z = src1.getType();

    mlir::Value res_tensor =
        mlir::tosa::LogicalXorOp::create(builder, loc, z, input1, input2);
    mb->add2Pool(res_tensor);
    return true;
  };
}

// tosa.matmul (mlir::tosa::MatMulOp)
// Type outputType, Value a, Value b) {
OpGenerator tosaMatMulOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
        [](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() == 3;
        },
        &operandCandidates);
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for (int i = 0; i < 3; i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(src2Candidates.begin(),
                            mb->valuePool.pool[src1TyStr].begin(),
                            mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(generateRankedTensor(
          builder, loc, mlir::dyn_cast<RankedTensorType>(src1.getType()), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    Type outputType = src1.getType();
    Value a = src1;
    Value b = src2;

    mlir::Value res =
        mlir::tosa::MatMulOp::create(builder, loc, outputType, a, b);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.max_pool2d (mlir::tosa::MaxPool2dOp)
// ::mlir::Type output, ::mlir::Value input,
// ::mlir::DenseI64ArrayAttr kernel, ::mlir::DenseI64ArrayAttr stride,
// ::mlir::DenseI64ArrayAttr pad) {
OpGenerator tosaMaxPool2dOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
        [](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() == 4;
        },
        &operandCandidates);
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for (int i = 0; i < 4; i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src1;
    auto output = src1.getType();

    SmallVector<int64_t> a, b, c;
    for (int i = 0; i < 2; i++) {
      a.push_back(1);
    }
    for (int i = 0; i < 2; i++) {
      b.push_back(1);
    }
    for (int i = 0; i < 4; i++) {
      c.push_back(1);
    }
    auto kernel = builder.getDenseI64ArrayAttr(a);
    auto stride = builder.getDenseI64ArrayAttr(b);
    auto pad = builder.getDenseI64ArrayAttr(c);

    mlir::Value res = mlir::tosa::MaxPool2dOp::create(
        builder, loc, output, input, kernel, stride, pad);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.maximum (mlir::tosa::MaximumOp)
// ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
OpGenerator tosaMaximumOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(src2Candidates.begin(),
                            mb->valuePool.pool[src1TyStr].begin(),
                            mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(generateRankedTensor(
          builder, loc, mlir::dyn_cast<RankedTensorType>(src1.getType()), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto input1 = src1;
    auto input2 = src2;
    auto output = src1.getType();

    mlir::Value res_tensor =
        mlir::tosa::MaximumOp::create(builder, loc, output, input1, input2);
    mb->add2Pool(res_tensor);
    return true;
  };
}

// tosa.minimum (mlir::tosa::MinimumOp)
OpGenerator tosaMinimumOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(src2Candidates.begin(),
                            mb->valuePool.pool[src1TyStr].begin(),
                            mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(generateRankedTensor(
          builder, loc, mlir::dyn_cast<RankedTensorType>(src1.getType()), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto input1 = src1;
    auto input2 = src2;
    auto output = src1.getType();

    mlir::Value res_tensor =
        mlir::tosa::MinimumOp::create(builder, loc, output, input1, input2);
    mb->add2Pool(res_tensor);
    return true;
  };
}

// tosa.mul (mlir::tosa::MulOp)
// ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2,
// ::mlir::IntegerAttr shift) {
OpGenerator tosaMulOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(src2Candidates.begin(),
                            mb->valuePool.pool[src1TyStr].begin(),
                            mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(generateRankedTensor(
          builder, loc, mlir::dyn_cast<RankedTensorType>(src1.getType()), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto output = src1.getType();
    auto input1 = src1;
    auto input2 = src2;
    auto shift = builder.getIntegerAttr(builder.getI8Type(),
                                        ((long long)rand() << 8) | rand());

    auto shiftElementType = IntegerType::get(builder.getContext(), 8);
    auto shiftType = RankedTensorType::get({1}, shiftElementType);
    auto shiftZeroAttr = DenseElementsAttr::get(shiftType, shift);
    Value shift_val =
        tosa::ConstOp::create(builder, loc, shiftType, shiftZeroAttr);

    mlir::Value res = mlir::tosa::MulOp::create(builder, loc, output, input1,
                                                input2, shift_val);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.negate (mlir::tosa::NegateOp)
// Type outputType, Value input) {
OpGenerator tosaNegateOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    Type outputType = src1.getType();
    Value input = src1;

    mlir::Value res =
        mlir::tosa::NegateOp::create(builder, loc, outputType, input);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.pad (mlir::tosa::PadOp)
// Type outputType, Value input, Value paddings) {
OpGenerator tosaPadOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> paddingsCandidates = std::vector<mlir::Value>();
    mb->search(
        [&](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType())
                         .getElementType() == builder.getI64Type() ||
                 mlir::dyn_cast<RankedTensorType>(v.getType())
                         .getElementType() == builder.getI32Type();
        },
        &paddingsCandidates);
    if (paddingsCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for (int i = 0; i < rollIdx(5); i++) {
        newShape.push_back(1);
      }
      paddingsCandidates.push_back(generateRankedTensor(
          builder, loc, RankedTensorType::get(newShape, builder.getI64Type()),
          mb));
    }
    auto src2 = paddingsCandidates[rollIdx(paddingsCandidates.size())];

    Type outputType = src1.getType();
    Value input = src1;
    Value paddings = src2;

    mlir::Value res =
        mlir::tosa::PadOp::create(builder, loc, outputType, input, paddings);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.pow (mlir::tosa::PowOp)
// ::mlir::Type z, ::mlir::Value input1, ::mlir::Value input2) {
OpGenerator tosaPowOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(src2Candidates.begin(),
                            mb->valuePool.pool[src1TyStr].begin(),
                            mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(generateRankedTensor(
          builder, loc, mlir::dyn_cast<RankedTensorType>(src1.getType()), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto input1 = src1;
    auto input2 = src2;
    auto z = src1.getType();

    mlir::Value res_tensor =
        mlir::tosa::PowOp::create(builder, loc, z, input1, input2);
    mb->add2Pool(res_tensor);
    return true;
  };
}

// ::mlir::Type output_real, ::mlir::Type output_imag, ::mlir::Value input) {
// tosa.rfft2d (mlir::tosa::RFFT2dOp)
OpGenerator tosaRFFT2dOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
        [](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() == 3;
        },
        &operandCandidates);
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for (int i = 0; i < 3; i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(src2Candidates.begin(),
                            mb->valuePool.pool[src1TyStr].begin(),
                            mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(generateRankedTensor(
          builder, loc, mlir::dyn_cast<RankedTensorType>(src1.getType()), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto output_real = src1.getType();
    auto output_imag = src2.getType();
    auto input = src2;

    auto res_tensor = mlir::tosa::RFFT2dOp::create(builder, loc, output_real,
                                                   output_imag, input);

    // mb->add2Pool(res_tensor);
    return true;
  };
}

// ::mlir::Type output, ::mlir::Value input1) {
// tosa.reciprocal (mlir::tosa::ReciprocalOp)
OpGenerator tosaReciprocalOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input1 = src1;
    auto output = src1.getType();

    mlir::Value res =
        mlir::tosa::ReciprocalOp::create(builder, loc, output, input1);
    mb->add2Pool(res);
    return true;
  };
}

//  ::mlir::Type output, ::mlir::Value input, uint64_t axis)
// tosa.reduce_all (mlir::tosa::ReduceAllOp)
OpGenerator tosaReduceAllOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
        [](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() > 0 &&
                 mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() <= 4;
        },
        &operandCandidates);
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      for (int i = 0; i < rollIdx(3); i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src1;
    auto output = src1.getType();

    uint64_t axis =
        rollIdx(mlir::cast<RankedTensorType>(src1.getType()).getRank());
    mlir::Value res =
        mlir::tosa::ReduceAllOp::create(builder, loc, output, input, axis);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.reduce_any (mlir::tosa::ReduceAnyOp)
// ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
OpGenerator tosaReduceAnyOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
        [](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() > 0 &&
                 mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() <= 4;
        },
        &operandCandidates);
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      for (int i = 0; i < rollIdx(3); i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src1;
    auto output = src1.getType();
    uint64_t axis =
        rollIdx(mlir::cast<RankedTensorType>(src1.getType()).getRank());
    mlir::Value res =
        mlir::tosa::ReduceAnyOp::create(builder, loc, output, input, axis);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.reduce_max (mlir::tosa::ReduceMaxOp)
//  ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
OpGenerator tosaReduceMaxOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
        [](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() > 0 &&
                 mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() <= 4;
        },
        &operandCandidates);
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      for (int i = 0; i < rollIdx(3); i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src1;
    auto output = src1.getType();
    uint64_t axis =
        rollIdx(mlir::dyn_cast<RankedTensorType>(input.getType()).getRank());
    mlir::Value res =
        mlir::tosa::ReduceMaxOp::create(builder, loc, output, input, axis);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.reduce_min (mlir::tosa::ReduceMinOp)
// operand #0 must be unranked.tensor of number values or 1D/2D/3D/4D tensor of
// number values, but got 'tensor<f16>'
OpGenerator tosaReduceMinOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
        [](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() > 0 &&
                 mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() <= 4;
        },
        &operandCandidates);
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      for (int i = 0; i < rollIdx(3); i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src1;
    auto output = src1.getType();
    uint64_t axis =
        rollIdx(mlir::dyn_cast<RankedTensorType>(input.getType()).getRank());
    mlir::Value res =
        mlir::tosa::ReduceMinOp::create(builder, loc, output, input, axis);
    mb->add2Pool(res);
    return true;
  };
}
// tosa.reduce_prod (mlir::tosa::ReduceProdOp)
OpGenerator tosaReduceProdOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
        [](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() > 0 &&
                 mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() <= 4;
        },
        &operandCandidates);
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      for (int i = 0; i < rollIdx(4); i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src1;
    auto output = src1.getType();
    uint64_t axis =
        rollIdx(mlir::cast<RankedTensorType>(src1.getType()).getRank());
    mlir::Value res =
        mlir::tosa::ReduceProductOp::create(builder, loc, output, input, axis);
    mb->add2Pool(res);
    return true;
  };
}
// tosa.reduce_sum (mlir::tosa::ReduceSumOp)
OpGenerator tosaReduceSumOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
        [](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() > 0 &&
                 mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() <= 4;
        },
        &operandCandidates);
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      for (int i = 0; i < rollIdx(4); i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src1;
    auto output = src1.getType();
    uint64_t axis =
        rollIdx(mlir::cast<RankedTensorType>(src1.getType()).getRank());
    mlir::Value res =
        mlir::tosa::ReduceSumOp::create(builder, loc, output, input, axis);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.rescale (mlir::tosa::RescaleOp)
// ::mlir::Type output, ::mlir::Value input,
//  ::mlir::IntegerAttr input_zp, ::mlir::IntegerAttr output_zp, :
//  ::mlir::DenseI32ArrayAttr multiplier, ::mlir::DenseI32ArrayAttr shift,
//  ::mlir::BoolAttr scale32, ::mlir::BoolAttr double_round, ::mlir::BoolAttr
//  per_channel
OpGenerator tosaRescaleOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];
    auto input = src1;
    auto output = src1.getType();

    auto input_zp = builder.getIntegerAttr(builder.getI32Type(),
                                           ((long long)rand() << 32) | rand());
    auto output_zp = builder.getIntegerAttr(builder.getI32Type(),
                                            ((long long)rand() << 32) | rand());
    auto i32ty = IntegerType::get(builder.getContext(), 32);
    auto i32_tensor_ty = RankedTensorType::get({1}, i32ty);
    auto output_zp_attr = DenseElementsAttr::get(i32_tensor_ty, output_zp);
    auto input_zp_attr = DenseElementsAttr::get(i32_tensor_ty, input_zp);
    Value input_zp_tensor =
        tosa::ConstOp::create(builder, loc, i32_tensor_ty, input_zp_attr);
    Value output_zp_tensor =
        tosa::ConstOp::create(builder, loc, i32_tensor_ty, output_zp_attr);

    // mlir::DenseI32ArrayAttr multiplier_attr =
    // builder.getDenseI32ArrayAttr(a); mlir::DenseI8ArrayAttr shift_attr =
    // builder.getDenseI8ArrayAttr(b);
    auto multiplier = builder.getIntegerAttr(
        builder.getI32Type(), ((long long)rand() << 32) | rand());
    auto shift = builder.getIntegerAttr(builder.getI32Type(),
                                        ((long long)rand() << 8) | rand());
    auto i8ty = IntegerType::get(builder.getContext(), 8);
    auto i8_tensor_ty = RankedTensorType::get({1}, i8ty);
    auto multipiler_attr = DenseElementsAttr::get(i32_tensor_ty, multiplier);
    auto shift_attr = DenseElementsAttr::get(i8_tensor_ty, shift);
    Value multiplier_tensor =
        tosa::ConstOp::create(builder, loc, i32_tensor_ty, multipiler_attr);
    Value shift_tensor =
        tosa::ConstOp::create(builder, loc, i8_tensor_ty, shift_attr);

    auto scale32 = builder.getBoolAttr(true);
    // auto double_round = builder.getBoolAttr(true);
    auto per_channel = builder.getBoolAttr(true);
    auto input_unsigned = builder.getBoolAttr(true);
    auto output_unsigned = builder.getBoolAttr(true);

    auto mode_idx = rollIdx(3);
    tosa::RoundingMode mode;
    if (mode_idx == 0) {
      mode = tosa::RoundingMode::SINGLE_ROUND;
    } else if (mode_idx == 1) {
      mode = tosa::RoundingMode::INEXACT_ROUND;
    } else if (mode_idx == 2) {
      mode = tosa::RoundingMode::DOUBLE_ROUND;
    } else {
      assert(false);
    }
    tosa::RoundingModeAttr rounding_mode_attr =
        tosa::RoundingModeAttr::get(builder.getContext(), mode);

    mlir::Value res = mlir::tosa::RescaleOp::create(
        builder, loc, output, input, multiplier_tensor, shift_tensor,
        input_zp_tensor, output_zp_tensor, scale32, rounding_mode_attr,
        per_channel, input_unsigned, output_unsigned);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.reshape (mlir::tosa::ReshapeOp)
// ::mlir::Type output, ::mlir::Value input1, ::mlir::DenseI64ArrayAttr
// new_shape) {
OpGenerator tosaReshapeOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input1 = src1;
    auto new_shape = operandCandidates[rollIdx(operandCandidates.size())];
    auto output = new_shape.getType();

    mlir::Value res =
        mlir::tosa::ReshapeOp::create(builder, loc, output, input1, new_shape);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.resize (mlir::tosa::ResizeOp)
// ::mlir::Type output, ::mlir::Value input,
// ::mlir::DenseI64ArrayAttr scale, ::mlir::DenseI64ArrayAttr offset,
// ::mlir::DenseI64ArrayAttr border,
// ::mlir::StringAttr mode) {
OpGenerator tosaResizeOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
        [](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() == 4;
        },
        &operandCandidates);
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for (int i = 0; i < 4; i++) {
        newShape.push_back(rollIdx(100) + 10);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];
    auto input = src1;
    auto output = src1.getType();

    SmallVector<mlir::Value> a, b, c;
    for (int i = 0; i < 4; i++) {
      a.push_back(mlir::index::ConstantOp::create(
          builder, loc, builder.getIndexAttr(rollIdx(10))));
    }
    for (int i = 0; i < 2; i++) {
      b.push_back(mlir::index::ConstantOp::create(
          builder, loc, builder.getIndexAttr(rollIdx(10))));
    }
    for (int i = 0; i < 2; i++) {
      c.push_back(mlir::index::ConstantOp::create(
          builder, loc, builder.getIndexAttr(rollIdx(10))));
    }
    auto scale =
        mlir::tensor::FromElementsOp::create(builder, loc, ValueRange(a));
    auto offset =
        mlir::tensor::FromElementsOp::create(builder, loc, ValueRange(b));
    auto border =
        mlir::tensor::FromElementsOp::create(builder, loc, ValueRange(c));

    // 'tosa.resize' op attribute 'mode' failed to satisfy constraint: Supported
    // resize/upsampling strategies

    // NearestNeighbor: 最近邻插值,简单快速但产生锯齿。
    // Linear: 双线性插值,较近邻产生更平滑结果但计算复杂度更高.
    // Area: 用平均值填充像素,保持像素总和不变但模糊原图。
    // Cubic: 三次样条插值,产生更光滑结果但计算开销最大。
    // Lanczos: 兰索斯窗口函数插值,在光滑度和计算效率上取得平衡。
    // MitchelNetravali: 一种二次Lanczos插值的变种算法。
    // std::vector<std::string> modes = {
    //     "NearestNeighbor", "Linear",  "Area",
    //     "Cubic",           "Lanczos", "MitchelNetravali"};

    mlir::tosa::ResizeModeAttr mode;
    if (rollIdx(2) == 0) {
      mode = mlir::tosa::ResizeModeAttr::get(builder.getContext(),
                                             mlir::tosa::ResizeMode::BILINEAR);
    } else {
      mode = mlir::tosa::ResizeModeAttr::get(
          builder.getContext(), mlir::tosa::ResizeMode::NEAREST_NEIGHBOR);
    }

    mlir::Value res = mlir::tosa::ResizeOp::create(builder, loc, output, input,
                                                   scale, offset, border, mode);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.reverse (mlir::tosa::ReverseOp)
// ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
OpGenerator tosaReverseOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
        [](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() > 0 &&
                 mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() <= 4;
        },
        &operandCandidates);
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      for (int i = 0; i < rollIdx(3); i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src1;
    auto output = src1.getType();

    uint64_t axis =
        rollIdx(mlir::dyn_cast<RankedTensorType>(src1.getType()).getRank());
    mlir::Value res =
        mlir::tosa::ReverseOp::create(builder, loc, output, input, axis);
    mb->add2Pool(res);
    return true;
  };
}

// ::mlir::Type output, ::mlir::Value input1) {
// tosa.rsqrt (mlir::tosa::RsqrtOp)
OpGenerator tosaRsqrtOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input1 = src1;
    auto output = src1.getType();

    mlir::Value res = mlir::tosa::RsqrtOp::create(builder, loc, output, input1);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.scatter (mlir::tosa::ScatterOp)
// ::mlir::Type values_out, ::mlir::Value values_in, ::mlir::Value indices,
// ::mlir::Value input) {
OpGenerator tosaScatterOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> _3DCandidates = std::vector<mlir::Value>();
    mb->search(
        [](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() == 3;
        },
        &_3DCandidates);
    if (_3DCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for (int i = 0; i < 3; i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      _3DCandidates.push_back(dest);
    }
    auto src0 = _3DCandidates[rollIdx(_3DCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src0);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(src2Candidates.begin(),
                            mb->valuePool.pool[src1TyStr].begin(),
                            mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(generateRankedTensor(
          builder, loc, mlir::dyn_cast<RankedTensorType>(src0.getType()), mb));
    }
    auto src1 = src2Candidates[rollIdx(src2Candidates.size())];

    std::vector<mlir::Value> _2D32bitCandidates = std::vector<mlir::Value>();
    mb->search(
        [&](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() == 2 &&
                 mlir::dyn_cast<RankedTensorType>(v.getType())
                         .getElementType() == builder.getI32Type();
        },
        &_2D32bitCandidates);
    if (_2D32bitCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for (int i = 0; i < 2; i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(newShape, builder.getI32Type());
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      _2D32bitCandidates.push_back(dest);
    }
    auto src2 = _2D32bitCandidates[rollIdx(_2D32bitCandidates.size())];

    auto values_out = src0.getType();
    auto values_in = src0;
    auto indices = src2;
    auto input = src1;

    mlir::Value res = mlir::tosa::ScatterOp::create(builder, loc, values_out,
                                                    values_in, indices, input);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.select (mlir::tosa::SelectOp)
// ::mlir::Type output, ::mlir::Value pred, ::mlir::Value on_true, ::mlir::Value
// on_false) {
OpGenerator tosaSelectOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> predCandidates = std::vector<mlir::Value>();
    mb->search(
        [&](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType())
                     .getElementType() == builder.getI1Type();
        },
        &predCandidates);
    if (predCandidates.empty()) {
      SmallVector<int64_t> newShape;
      auto predTy = RankedTensorType::get(newShape, builder.getI1Type());
      auto dest = generateRankedTensor(builder, loc, predTy, mb);
      predCandidates.push_back(dest);
    }
    auto src0 = predCandidates[rollIdx(predCandidates.size())];

    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(src2Candidates.begin(),
                            mb->valuePool.pool[src1TyStr].begin(),
                            mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(generateRankedTensor(
          builder, loc, mlir::dyn_cast<RankedTensorType>(src1.getType()), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto output = src1.getType();
    auto pred = src0;
    auto on_true = src1;
    auto on_false = src2;

    mlir::Value res_tensor = mlir::tosa::SelectOp::create(
        builder, loc, output, pred, on_true, on_false);
    mb->add2Pool(res_tensor);
    return true;
  };
}

// tosa.sigmoid (mlir::tosa::SigmoidOp)
//  ::mlir::Type output, ::mlir::Value input) {
OpGenerator tosaSigmoidOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src1;
    auto output = src1.getType();

    mlir::Value res =
        mlir::tosa::SigmoidOp::create(builder, loc, output, input);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.slice (mlir::tosa::SliceOp)
// ::mlir::Type output, ::mlir::Value input, ::mlir::DenseI64ArrayAttr start,
// ::mlir::DenseI64ArrayAttr size) {
OpGenerator tosaSliceOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
        [](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() > 0 &&
                 mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() <= 6;
        },
        &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src1;
    auto output = src1.getType();

    // SmallVector<int64_t> a, b;
    // for (int i = 0; i <
    // mlir::cast<RankedTensorType>(input.getType()).getRank();
    //      i++) {
    //   a.push_back(rollIdx(100));
    //   b.push_back(rollIdx(100));
    // }
    // auto start = builder.getDenseI64ArrayAttr(a);
    // auto size = builder.getDenseI64ArrayAttr(b);

    SmallVector<mlir::Value> starts;
    SmallVector<mlir::Value> sizes;
    for (int i = 0;
         i < mlir::dyn_cast<RankedTensorType>(input.getType()).getRank(); i++) {
      mlir::Value idx = mlir::index::ConstantOp::create(
          builder, loc, builder.getIndexAttr(i));
      mlir::Value dim_size =
          mlir::tensor::DimOp::create(builder, loc, input, idx);
      mlir::Value ori_start = mlir::index::ConstantOp::create(
          builder, loc, builder.getIndexAttr(rollIdx(100)));
      mlir::Value adjusted_start =
          mlir::index::RemSOp::create(builder, loc, ori_start, dim_size);
      mlir::Value size =
          mlir::index::SubOp::create(builder, loc, dim_size, adjusted_start);
      starts.push_back(adjusted_start);
      sizes.push_back(size);
    }
    mlir::Value start =
        mlir::tensor::FromElementsOp::create(builder, loc, ValueRange(starts));
    mlir::Value size =
        mlir::tensor::FromElementsOp::create(builder, loc, ValueRange(size));

    mlir::Value res =
        mlir::tosa::SliceOp::create(builder, loc, output, input, start, size);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.sub (mlir::tosa::SubOp)
// ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
OpGenerator tosaSubOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(src2Candidates.begin(),
                            mb->valuePool.pool[src1TyStr].begin(),
                            mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(generateRankedTensor(
          builder, loc, mlir::dyn_cast<RankedTensorType>(src1.getType()), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto input1 = src1;
    auto input2 = src2;
    auto output = src1.getType();

    mlir::Value res_tensor =
        mlir::tosa::SubOp::create(builder, loc, output, input1, input2);
    mb->add2Pool(res_tensor);
    return true;
  };
}

// tosa.table (mlir::tosa::TableOp)
// ::mlir::Type output, ::mlir::Value input, ::mlir::Value table) {
OpGenerator tosaTableOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      for (int i = 0; i < rollIdx(3); i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> tableCandidates = std::vector<mlir::Value>();
    mb->search(
        [](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() == 1;
        },
        &tableCandidates);
    if (tableCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      auto tableTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, tableTy, mb);
      tableCandidates.push_back(dest);
    }
    auto src2 = tableCandidates[rollIdx(tableCandidates.size())];

    auto output = src1.getType();
    auto input = src1;
    auto table = src2;

    mlir::Value res =
        mlir::tosa::TableOp::create(builder, loc, output, input, table);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.tanh (mlir::tosa::TanhOp)
OpGenerator tosaTanhOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src1;
    auto output = src1.getType();

    mlir::Value res = mlir::tosa::TanhOp::create(builder, loc, output, input);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.tile (mlir::tosa::TileOp)
OpGenerator tosaTileOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
        [](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() > 0 &&
                 mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() <= 4;
        },
        &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src1;
    auto output = src1.getType();

    // SmallVector<int64_t> a;
    // for (int i = 0; i < input.getType().cast<RankedTensorType>().getRank();
    // i++) {
    //   a.push_back(1);
    // }
    // auto multiples = builder.getDenseI64ArrayAttr(a);

    mlir::Value multiples = arith::ConstantOp::create(
        builder, loc,
        builder.getIntegerAttr(IntegerType::get(builder.getContext(), 32), 1));

    mlir::Value res =
        mlir::tosa::TileOp::create(builder, loc, output, input, multiples);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.transpose_conv2d (mlir::tosa::TransposeConv2DOp)
// ::mlir::Type output, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value
// bias,
// ::mlir::DenseI64ArrayAttr out_pad, ::mlir::DenseI64ArrayAttr stride,
// ::mlir::DenseI64ArrayAttr out_shape,
// /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
OpGenerator tosaTransposeConv2DOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> inputCandidates = std::vector<mlir::Value>();
    mb->search(
        [](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() == 4;
        },
        &inputCandidates);
    if (inputCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for (int i = 0; i < 4; i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      inputCandidates.push_back(dest);
    }
    auto src1 = inputCandidates[rollIdx(inputCandidates.size())];

    auto input = src1;
    auto output = src1.getType();

    std::vector<mlir::Value> biasCandidates = std::vector<mlir::Value>();
    mb->search(
        [](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() == 1;
        },
        &biasCandidates);
    if (biasCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      biasCandidates.push_back(dest);
    }
    auto src2 = biasCandidates[rollIdx(biasCandidates.size())];
    auto bias = src2;

    SmallVector<Type> supported_types;
    // supported_types.push_back(builder.getI4Type());
    supported_types.push_back(builder.getI8Type());
    supported_types.push_back(builder.getF16Type());
    supported_types.push_back(builder.getF32Type());

    std::vector<mlir::Value> filterCandidates = std::vector<mlir::Value>();
    mb->search(
        [&](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() == 4 &&
                 (
                     // tval.type.dyn_cast<RankedTensorType>().getElementType()
                     // == builder.getI4Type() ||
                     mlir::dyn_cast<RankedTensorType>(v.getType())
                             .getElementType() == builder.getI8Type() ||
                     mlir::dyn_cast<RankedTensorType>(v.getType())
                             .getElementType() == builder.getF32Type() ||
                     mlir::dyn_cast<RankedTensorType>(v.getType())
                             .getElementType() == builder.getF16Type());
        },
        &filterCandidates);
    if (filterCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for (int i = 0; i < 4; i++) {
        newShape.push_back(rollIdx(20));
      }
      auto destTy =
          RankedTensorType::get(newShape, supported_types[rollIdx(3)]);
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      filterCandidates.push_back(dest);
    }
    auto src3 = filterCandidates[rollIdx(filterCandidates.size())];
    auto filter = src3;

    SmallVector<int64_t> a, b, c;
    for (int i = 0; i < 4; i++) {
      a.push_back(1);
    }
    for (int i = 0; i < 2; i++) {
      b.push_back(1);
    }
    for (int i = 0; i < 4; i++) {
      c.push_back(1);
    }
    auto out_pad = builder.getDenseI64ArrayAttr(a);
    auto stride = builder.getDenseI64ArrayAttr(b);
    auto out_shape = builder.getDenseI64ArrayAttr(c);

    mlir::IntegerType int32Type =
        mlir::IntegerType::get(builder.getContext(), 32);
    mlir::TypeAttr typeAttr = mlir::TypeAttr::get(int32Type);

    // mlir::Value res = mlir::tosa::TransposeConv2DOp::create(
    //     builder, loc, output, input, filter, bias, out_pad, stride,
    //     out_shape, typeAttr);
    mlir::Value res = mlir::tosa::TransposeConv2DOp::create(
        builder, loc, output, input, filter, bias, out_pad, stride, typeAttr);
    mb->add2Pool(res);
    return true;
  };
}

// ::mlir::Type output, ::mlir::Value input1, ::mlir::Value perms) {
// tosa.transpose (mlir::tosa::TransposeOp)

OpGenerator tosaTransposeOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock *mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
        [](std::string s, mlir::Value v) -> bool {
          if (!ranked_tensor_filter(s, v))
            return false;
          return mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() > 0 &&
                 mlir::dyn_cast<RankedTensorType>(v.getType()).getRank() <= 6;
        },
        &operandCandidates);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    // std::vector<mlir::Value> permsCandidates = std::vector<mlir::Value>();
    // mb->search(
    //     [&](std::string s, mlir::Value v) -> bool {
    //       if (!ranked_tensor_filter(s, v))
    //         return false;
    //       return mlir::dyn_cast<RankedTensorType>(v.getType())
    //                      .getElementType() == builder.getI32Type() ||
    //              mlir::dyn_cast<RankedTensorType>(v.getType())
    //                      .getElementType() == builder.getI64Type();
    //     },
    //     &permsCandidates);
    // if (permsCandidates.empty()) {
    //   SmallVector<int64_t> newShape;
    //   auto destTy = RankedTensorType::get(newShape, builder.getI64Type());
    //   if (rollIdx(2) == 1) {
    //     auto destTy = RankedTensorType::get(newShape, builder.getI32Type());
    //   }
    //   auto dest = generateRankedTensor(builder, loc, destTy, mb);
    //   permsCandidates.push_back(dest);
    // }
    // auto src2 = permsCandidates[rollIdx(permsCandidates.size())];

    auto output = src1.getType();
    auto input1 = src1;
    // auto perms = src2;

    std::random_device rd;
    std::mt19937 g(rd());
    SmallVector<int32_t> ranks;
    for (int i = 0; i < mlir::dyn_cast<RankedTensorType>(src1.getType()).getRank(); i++) {
      ranks.push_back(i);
    }
    std::shuffle(ranks.begin(), ranks.end(), g);

    auto perms = builder.getDenseI32ArrayAttr(ranks);
    mlir::Value res_tensor = mlir::tosa::TransposeOp::create(builder, loc, output, input1, perms);
    mb->add2Pool(res_tensor);
    return true;
  };
}
