//
// Created by Stan Wang on 2022/10/28.
//

#include "smith/TypeGeneration.h"
#include "smith/generators/OpGeneration.h"

/*------------ math generator ------------*/

OpGenerator absFGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return mlir::isa<FloatType>(tval.type) ||
                 (mlir::isa<ShapedType>(tval.type) &&
                  mlir::isa<FloatType>(
                      mlir::dyn_cast<ShapedType>(tval.type).getElementType()));
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.absf");
    auto value = math::AbsFOp::create(builder, loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.absf");
    return value.getOperation();
  };
}

OpGenerator absIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return mlir::isa<IntegerType>(tval.type) ||
                 (mlir::isa<ShapedType>(tval.type) &&
                  mlir::isa<IntegerType>(
                      mlir::dyn_cast<ShapedType>(tval.type).getElementType()));
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateInteger(builder, loc, builder.getI32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.absi");
    auto value = math::AbsIOp::create(builder, loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.absi");
    return value.getOperation();
  };
}

OpGenerator atanGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return mlir::isa<FloatType>(tval.type) ||
                 (mlir::isa<ShapedType>(tval.type) &&
                  mlir::isa<FloatType>(
                      mlir::dyn_cast<ShapedType>(tval.type).getElementType()));
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.atan");
    auto value = math::AtanOp::create(builder, loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.atan");
    return value.getOperation();
  };
}

OpGenerator atan2Generator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto elemTy = builder.getF32Type();
    auto candidates =
        typedValuePool.getCandidatesFromIntOrFloats(builder, loc, elemTy);
    auto tensorCandidates = typedValuePool.getCandidatesFromStaticShapedTensor(
        builder, loc, randomStaticShapedTensorType(elemTy));
    candidates.insert(candidates.end(), tensorCandidates.begin(),
                      tensorCandidates.end());
    auto operands =
        searchFloatBinaryOperandsFrom(builder, loc, candidates, "math.atan2");
    auto value =
        math::Atan2Op::create(builder, loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "math.atan2");
    return value.getOperation();
  };
}

OpGenerator ceilGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return mlir::isa<FloatType>(tval.type) ||
                 (mlir::isa<ShapedType>(tval.type) &&
                  mlir::isa<FloatType>(
                      mlir::dyn_cast<ShapedType>(tval.type).getElementType()));
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.ceil");
    auto value = math::CeilOp::create(builder, loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.ceil");
    return value.getOperation();
  };
}

OpGenerator copySignGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto elemTy = builder.getF32Type();
    auto candidates =
        typedValuePool.getCandidatesFromIntOrFloats(builder, loc, elemTy);
    auto tensorCandidates = typedValuePool.getCandidatesFromStaticShapedTensor(
        builder, loc, randomStaticShapedTensorType(elemTy));
    candidates.insert(candidates.end(), tensorCandidates.begin(),
                      tensorCandidates.end());
    auto operands = searchFloatBinaryOperandsFrom(builder, loc, candidates,
                                                  "math.copySign");
    auto value = math::CopySignOp::create(builder, loc, operands[0].val,
                                          operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "math.copySign");
    return value.getOperation();
  };
} // f b

OpGenerator cosGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return mlir::isa<FloatType>(tval.type) ||
                 (mlir::isa<ShapedType>(tval.type) &&
                  mlir::isa<FloatType>(
                      mlir::dyn_cast<ShapedType>(tval.type).getElementType()));
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.cos");
    auto value = math::CosOp::create(builder, loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.cos");
    return value.getOperation();
  };
} // f u

OpGenerator sinGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return mlir::isa<FloatType>(tval.type) ||
                 (mlir::isa<ShapedType>(tval.type) &&
                  mlir::isa<FloatType>(
                      mlir::dyn_cast<ShapedType>(tval.type).getElementType()));
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.sin");
    auto value = math::SinOp::create(builder, loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.sin");
    return value.getOperation();
  };
}

OpGenerator ctlzGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return mlir::isa<IntegerType>(tval.type) ||
                 (mlir::isa<ShapedType>(tval.type) &&
                  mlir::isa<IntegerType>(
                      mlir::dyn_cast<ShapedType>(tval.type).getElementType()));
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateInteger(builder, loc, builder.getI32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.ctzl");
    auto value = math::CountLeadingZerosOp::create(builder, loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.ctzl");
    return value.getOperation();
  };
}

OpGenerator cttzGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return mlir::isa<IntegerType>(tval.type) ||
                 (mlir::isa<ShapedType>(tval.type) &&
                  mlir::isa<IntegerType>(
                      mlir::dyn_cast<ShapedType>(tval.type).getElementType()));
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateInteger(builder, loc, builder.getI32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.cttz");
    auto value = math::CountTrailingZerosOp::create(builder, loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.cttz");
    return value.getOperation();
  };
}

OpGenerator ctpopGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return mlir::isa<IntegerType>(tval.type) ||
                 (mlir::isa<ShapedType>(tval.type) &&
                  mlir::isa<IntegerType>(
                      mlir::dyn_cast<ShapedType>(tval.type).getElementType()));
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateInteger(builder, loc, builder.getI32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.ctpop");
    auto value = math::CtPopOp::create(builder, loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.ctpop");
    return value.getOperation();
  };
}

// OpGenerator erfGenerator(); //f u // What's the semantic of this op?

OpGenerator expGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return mlir::isa<FloatType>(tval.type) ||
                 (mlir::isa<ShapedType>(tval.type) &&
                  mlir::isa<FloatType>(
                      mlir::dyn_cast<ShapedType>(tval.type).getElementType()));
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.exp");
    auto value = math::ExpOp::create(builder, loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.exp");
    return value.getOperation();
  };
}

OpGenerator exp2Generator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return mlir::isa<FloatType>(tval.type) ||
                 (mlir::isa<ShapedType>(tval.type) &&
                  mlir::isa<FloatType>(
                      mlir::dyn_cast<ShapedType>(tval.type).getElementType()));
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.exp2");
    auto value = math::Exp2Op::create(builder, loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.exp2");
    return value.getOperation();
  };
}

OpGenerator expm1Generator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return mlir::isa<FloatType>(tval.type) ||
                 (mlir::isa<ShapedType>(tval.type) &&
                  mlir::isa<FloatType>(
                      mlir::dyn_cast<ShapedType>(tval.type).getElementType()));
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.expm1");
    auto value = math::ExpM1Op::create(builder, loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.expm1");
    return value.getOperation();
  };
}

OpGenerator floorGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return mlir::isa<FloatType>(tval.type) ||
                 (mlir::isa<ShapedType>(tval.type) &&
                  mlir::isa<FloatType>(
                      mlir::dyn_cast<ShapedType>(tval.type).getElementType()));
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.floor");
    auto value = math::FloorOp::create(builder, loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.floor");
    return value.getOperation();
  };
}

OpGenerator fmaGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto elemTy = builder.getF32Type();
    auto candidates =
        typedValuePool.getCandidatesFromIntOrFloats(builder, loc, elemTy);
    auto tensorCandidates = typedValuePool.getCandidatesFromStaticShapedTensor(
        builder, loc, randomStaticShapedTensorType(elemTy));
    candidates.insert(candidates.end(), tensorCandidates.begin(),
                      tensorCandidates.end());
    auto operands =
        searchFloatTernaryOperandsFrom(builder, loc, candidates, "math.fma");
    auto value = math::FmaOp::create(builder, loc, operands[0].val,
                                     operands[1].val, operands[2].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "math.fma");
    return value.getOperation();
  };
} // f t

OpGenerator ipowiGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto elemTy = builder.getI32Type();
    auto candidates =
        typedValuePool.getCandidatesFromIntOrFloats(builder, loc, elemTy);
    auto tensorCandidates = typedValuePool.getCandidatesFromStaticShapedTensor(
        builder, loc, randomStaticShapedTensorType(elemTy));
    candidates.insert(candidates.end(), tensorCandidates.begin(),
                      tensorCandidates.end());
    auto operands =
        searchIntegerBinaryOperandsFrom(builder, loc, candidates, "math.ipowi");
    auto value =
        math::IPowIOp::create(builder, loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "math.ipowi");
    return value.getOperation();
  };
} // i b

OpGenerator logGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return mlir::isa<FloatType>(tval.type) ||
                 (mlir::isa<ShapedType>(tval.type) &&
                  mlir::isa<FloatType>(
                      mlir::dyn_cast<ShapedType>(tval.type).getElementType()));
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.log");
    auto value = math::LogOp::create(builder, loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.log");
    return value.getOperation();
  };
} // f u
OpGenerator log10Generator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return mlir::isa<FloatType>(tval.type) ||
                 (mlir::isa<ShapedType>(tval.type) &&
                  mlir::isa<FloatType>(
                      mlir::dyn_cast<ShapedType>(tval.type).getElementType()));
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.log10");
    auto value = math::Log10Op::create(builder, loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.log10");
    return value.getOperation();
  };
}
OpGenerator log1pGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return mlir::isa<FloatType>(tval.type) ||
                 (mlir::isa<ShapedType>(tval.type) &&
                  mlir::isa<FloatType>(
                      mlir::dyn_cast<ShapedType>(tval.type).getElementType()));
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.log1p");
    auto value = math::Log1pOp::create(builder, loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.log1p");
    return value.getOperation();
  };
}
OpGenerator log2Generator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return mlir::isa<FloatType>(tval.type) ||
                 (mlir::isa<ShapedType>(tval.type) &&
                  mlir::isa<FloatType>(
                      mlir::dyn_cast<ShapedType>(tval.type).getElementType()));
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.log2");
    auto value = math::Log2Op::create(builder, loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.log2");
    return value.getOperation();
  };
}
OpGenerator powfGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto elemTy = builder.getF32Type();
    auto candidates =
        typedValuePool.getCandidatesFromIntOrFloats(builder, loc, elemTy);
    auto tensorCandidates = typedValuePool.getCandidatesFromStaticShapedTensor(
        builder, loc, randomStaticShapedTensorType(elemTy));
    candidates.insert(candidates.end(), tensorCandidates.begin(),
                      tensorCandidates.end());
    auto operands =
        searchFloatBinaryOperandsFrom(builder, loc, candidates, "math.powF");
    auto value =
        math::PowFOp::create(builder, loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "math.powF");
    return value.getOperation();
  };
} // f b

OpGenerator rsqrtGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return mlir::isa<FloatType>(tval.type) ||
                 (mlir::isa<ShapedType>(tval.type) &&
                  mlir::isa<FloatType>(
                      mlir::dyn_cast<ShapedType>(tval.type).getElementType()));
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.rqst");
    auto value = math::RsqrtOp::create(builder, loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.rsqrt");
    return value.getOperation();
  };
} // f u
OpGenerator sqrtGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return mlir::isa<FloatType>(tval.type) ||
                 (mlir::isa<ShapedType>(tval.type) &&
                  mlir::isa<FloatType>(
                      mlir::dyn_cast<ShapedType>(tval.type).getElementType()));
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.sqrt");
    auto value = math::SqrtOp::create(builder, loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.sqrt");
    return value.getOperation();
  };
}
OpGenerator tanGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return mlir::isa<FloatType>(tval.type) ||
                 (mlir::isa<ShapedType>(tval.type) &&
                  mlir::isa<FloatType>(
                      mlir::dyn_cast<ShapedType>(tval.type).getElementType()));
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.absf");
    auto value = math::TanOp::create(builder, loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.tan");
    return value.getOperation();
  };
}
OpGenerator tanhGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return mlir::isa<FloatType>(tval.type) ||
                 (mlir::isa<ShapedType>(tval.type) &&
                  mlir::isa<FloatType>(
                      mlir::dyn_cast<ShapedType>(tval.type).getElementType()));
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.tanh");
    auto value = math::TanhOp::create(builder, loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.tanh");
    return value.getOperation();
  };
}
OpGenerator roundEvenGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return mlir::isa<FloatType>(tval.type) ||
                 (mlir::isa<ShapedType>(tval.type) &&
                  mlir::isa<FloatType>(
                      mlir::dyn_cast<ShapedType>(tval.type).getElementType()));
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.roundeven");
    auto value = math::RoundEvenOp::create(builder, loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.roundeven");
    return value.getOperation();
  };
}

OpGenerator roundGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto elemTy = builder.getF32Type();

    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return mlir::isa<FloatType>(tval.type) ||
                 (mlir::isa<ShapedType>(tval.type) &&
                  mlir::isa<FloatType>(
                      mlir::dyn_cast<ShapedType>(tval.type).getElementType()));
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.round");
    auto value = math::RoundOp::create(builder, loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.round");
    return value.getOperation();
  };
}

OpGenerator truncGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return mlir::isa<FloatType>(tval.type) ||
                 (mlir::isa<ShapedType>(tval.type) &&
                  mlir::isa<FloatType>(
                      mlir::dyn_cast<ShapedType>(tval.type).getElementType()));
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.trunc");
    auto value = math::RoundOp::create(builder, loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.trunc");
    return value.getOperation();
  };
}

OpGenerator fpowiGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto elemTy = builder.getF32Type();
    auto candidates =
        typedValuePool.getCandidatesFromIntOrFloats(builder, loc, elemTy);
    auto tensorCandidates = typedValuePool.getCandidatesFromStaticShapedTensor(
        builder, loc, randomStaticShapedTensorType(elemTy));
    candidates.insert(candidates.end(), tensorCandidates.begin(),
                      tensorCandidates.end());
    auto operand0 =
        searchFloatUnaryOperandFrom(builder, loc, candidates, "math.fpowi");
    auto ctx = builder.getContext();
    auto i32 = IntegerType::get(ctx, 32);
    std::vector<TypeValue> candidates2;

    if (mlir::dyn_cast<FloatType>(operand0.type)) {
      for (auto ele : typedValuePool.getCandidatesFromIntOrFloats(
               builder, loc, builder.getI32Type())) {
        if (ele.type == i32) {
          candidates2.push_back(ele);
        }
      }
    } else { // otherwise it should be Tensor Type
      auto t = RankedTensorType::get(
          mlir::dyn_cast<ShapedType>(operand0.type).getShape(), i32);
      candidates2 = searchShapedInputFrom(
          t,
          typedValuePool.getCandidatesFromStaticShapedTensor(builder, loc, t));
    }
    if (candidates2.empty()) {
      Type ty;
      if (mlir::isa<FloatType>(operand0.type)) {
        ty = builder.getI32Type();
      } else {
        ty = RankedTensorType::get(
            mlir::dyn_cast<ShapedType>(operand0.type).getShape(), i32);
      }
      candidates2.push_back(region.pool.generateTypedValue(builder, loc, ty));
    }
    auto operand1 = sampleTypedValueFrom(candidates2, "math.fpowi");
    auto value =
        math::FPowIOp::create(builder, loc, operand0.val, operand1.val);
    auto tVal = TypeValue(operand0.type, value);
    typedValuePool.addTypeValue(tVal, "math.fpowi");
    return value.getOperation();
  };
} // f,i
