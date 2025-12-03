//
// Created by Stan Wang on 2022/10/28.
//
#include "smith/Util.h"
#include "smith/TypeGeneration.h"
#include "smith/generators/OpGeneration.h"

/*---------------------arith operations-------------------*/

std::vector<arith::CmpFPredicate> cmpFPredicates = {
    arith::CmpFPredicate::AlwaysFalse, arith::CmpFPredicate::AlwaysTrue,
    arith::CmpFPredicate::OEQ,         arith::CmpFPredicate::OGE,
    arith::CmpFPredicate::OGT,         arith::CmpFPredicate::OLE,
    arith::CmpFPredicate::OLT,         arith::CmpFPredicate::ONE,
    arith::CmpFPredicate::ORD,         arith::CmpFPredicate::UEQ,
    arith::CmpFPredicate::UGT,         arith::CmpFPredicate::UGE,
    arith::CmpFPredicate::ULT,         arith::CmpFPredicate::ULE,
    arith::CmpFPredicate::UNE,         arith::CmpFPredicate::UNO};

std::vector<arith::CmpIPredicate> cmpIPredicates = {
    arith::CmpIPredicate::eq,  arith::CmpIPredicate::ne,
    arith::CmpIPredicate::slt, arith::CmpIPredicate::sle,
    arith::CmpIPredicate::sgt, arith::CmpIPredicate::sge,
    arith::CmpIPredicate::ult, arith::CmpIPredicate::ule,
    arith::CmpIPredicate::ugt, arith::CmpIPredicate::uge};

OpGenerator addFGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getF32Type());
    auto operands =
        searchFloatBinaryOperandsFrom(builder, loc, candidates, "arith.addF");
    auto value =
        arith::AddFOp::create(builder, loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.addF");
    return value.getOperation();
  };
}

OpGenerator addIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;

    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());

    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands = searchIntegerBinaryOperandsFrom(builder, loc, candidates,
                                                    "arith.addI");
    auto value =
        arith::AddIOp::create(builder, loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.addI");
    return value.getOperation();
  };
}

OpGenerator andIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    auto operands =
        searchIntegerBinaryOperandsFrom(builder, loc, candidates, "arith.andI");
    auto value =
        arith::AndIOp::create(builder, loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.andI");
    return value.getOperation();
  };
}

OpGenerator ceilDivSIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    auto operands = searchIntegerBinaryOperandsFrom(builder, loc, candidates,
                                                    "arith.ceilDivSI");
    auto value = arith::CeilDivSIOp::create(builder, loc, operands[0].val,
                                                    operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.ceilDivSI");
    return value.getOperation();
  };
}

OpGenerator ceilDivUIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands = searchIntegerBinaryOperandsFrom(builder, loc, candidates,
                                                    "arith.ceilDivUI");
    auto value = arith::CeilDivUIOp::create(builder, loc, operands[0].val,
                                                    operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.ceilDivUI");
    return value.getOperation();
  };
}

OpGenerator cmpFGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getF32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands =
        searchFloatBinaryOperandsFrom(builder, loc, candidates, "arith.cmpF");
    auto predicate = cmpFPredicates[UR(cmpFPredicates.size())];
    auto value = arith::CmpFOp::create(builder, loc, predicate, operands[0].val,
                                               operands[1].val);
    if (mlir::dyn_cast<FloatType>(operands[0].type)) {
      auto tVal = TypeValue(builder.getI1Type(), value);
      typedValuePool.addIntOrFloat(tVal, "");
    } else if (mlir::dyn_cast<RankedTensorType>(operands[0].type)) {
      auto tVal =
          TypeValue(RankedTensorType::get(
                        mlir::dyn_cast<ShapedType>(operands[0].type).getShape(),
                        builder.getI1Type()),
                    value);
      typedValuePool.addTypeValue(tVal);
    }
    return value.getOperation();
  };
}

OpGenerator cmpIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands =
        searchIntegerBinaryOperandsFrom(builder, loc, candidates, "arith.cmpI");
    auto predicate = cmpIPredicates[UR(cmpIPredicates.size())];
    auto value = arith::CmpIOp::create(builder, loc, predicate, operands[0].val,
                                               operands[1].val);
    if (mlir::dyn_cast<IntegerType>(operands[0].type)) {
      auto tVal = TypeValue(builder.getI1Type(), value);
      typedValuePool.addIntOrFloat(tVal,"");
    } else if (mlir::dyn_cast<RankedTensorType>(operands[0].type)) {
      auto tVal =
          TypeValue(RankedTensorType::get(
                        mlir::dyn_cast<ShapedType>(operands[0].type).getShape(),
                        builder.getI1Type()),
                    value);
      typedValuePool.addTypeValue(tVal);
    }
    return value.getOperation();
  };
}

OpGenerator constantGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    std::vector<mlir::Type> supportedElementaryTypes =
        getSupportedIntOrFloatTypes(builder.getContext());
    auto elementType =
        supportedElementaryTypes[UR(supportedElementaryTypes.size())];
    IntegerType iType = mlir::dyn_cast<IntegerType>(elementType);
    assert(elementType.isIntOrFloat());
    if (iType) {
      // TODO- support value width greater than 32.
      unsigned width = min(iType.getWidth(), 32);
      long long val = UR(((long long)1) << width);
      Value cons = arith::ConstantOp::create(builder, 
          loc, builder.getIntegerAttr(elementType, val));
      auto tVal = TypeValue(iType, cons);
      typedValuePool.addIntOrFloat(tVal, "arith.constant");
      return cons.getDefiningOp();
    } else {
      FloatType fType = mlir::dyn_cast<FloatType>(elementType);
      unsigned width = min(fType.getWidth(), 32);
      long long valf = UR(((long long)1) << width);
      double val = static_cast<double>(valf);
      Value cons = arith::ConstantOp::create(builder, 
          loc, builder.getFloatAttr(elementType, val));
      auto tVal = TypeValue(fType, cons);
      typedValuePool.addIntOrFloat(tVal, "arith.constant");
      return cons.getDefiningOp();
    }
  };
}

OpGenerator divFGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getF32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands =
        searchFloatBinaryOperandsFrom(builder, loc, candidates, "arith.divF");
    auto value =
        arith::DivFOp::create(builder, loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.divF");
    return value.getOperation();
  };
}

OpGenerator divSIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands = searchIntegerBinaryOperandsFrom(builder, loc, candidates,
                                                    "arith.divSI");
    auto value =
        arith::DivSIOp::create(builder, loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.divSI");
    return value.getOperation();
  };
}

OpGenerator divUIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands = searchIntegerBinaryOperandsFrom(builder, loc, candidates,
                                                    "arith.divUI");
    auto value =
        arith::DivUIOp::create(builder, loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.divUI");
    return value.getOperation();
  };
}

OpGenerator floorDivSIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands = searchIntegerBinaryOperandsFrom(builder, loc, candidates,
                                                    "arith.floorDivSI");
    auto value = arith::FloorDivSIOp::create(builder, loc, operands[0].val,
                                                     operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.floorDivSI");
    return value.getOperation();
  };
}

OpGenerator minSIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands = searchIntegerBinaryOperandsFrom(builder, loc, candidates,
                                                    "arith.minSI");
    auto value =
        arith::MinSIOp::create(builder, loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.minSI");
    return value.getOperation();
  };
}

OpGenerator minUIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands = searchIntegerBinaryOperandsFrom(builder, loc, candidates,
                                                    "arith.minUI");
    auto value =
        arith::MinUIOp::create(builder, loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.minUI");
    return value.getOperation();
  };
}

OpGenerator mulFGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getF32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands =
        searchFloatBinaryOperandsFrom(builder, loc, candidates, "arith.mulF");
    auto value =
        arith::MulFOp::create(builder, loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.mulF");
    return value.getOperation();
  };
}

OpGenerator mulIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands =
        searchIntegerBinaryOperandsFrom(builder, loc, candidates, "arith.mulI");
    auto value =
        arith::MulIOp::create(builder, loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.mulI");
    return value.getOperation();
  };
}

OpGenerator negFGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getF32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operand0 =
        searchFloatUnaryOperandFrom(builder, loc, candidates, "arith.negF");
    auto value = arith::NegFOp::create(builder, loc, operand0.val);
    auto tVal = TypeValue(operand0.type, value);
    typedValuePool.addTypeValue(tVal, "arith.negF");
    return value.getOperation();
  };
}

OpGenerator orIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands =
        searchIntegerBinaryOperandsFrom(builder, loc, candidates, "arith.orI");
    auto value =
        arith::OrIOp::create(builder, loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.orI");
    return value.getOperation();
  };
}

OpGenerator remFGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getF32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands =
        searchFloatBinaryOperandsFrom(builder, loc, candidates, "arith.remF");
    auto value =
        arith::RemFOp::create(builder, loc, operands[0].val, operands[1].val);

    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.remF");
    return value.getOperation();
  };
}

OpGenerator remSIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands = searchIntegerBinaryOperandsFrom(builder, loc, candidates,
                                                    "arith.remSI");
    auto value =
        arith::RemSIOp::create(builder, loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.remSI");
    return value.getOperation();
  };
}

OpGenerator remUIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands = searchIntegerBinaryOperandsFrom(builder, loc, candidates,
                                                    "arith.remUI");
    auto value =
        arith::RemUIOp::create(builder, loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.remUI");
    return value.getOperation();
  };
}

OpGenerator shlIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands =
        searchIntegerBinaryOperandsFrom(builder, loc, candidates, "arith.shlI");
    auto value =
        arith::ShLIOp::create(builder, loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.shlI");
    return value.getOperation();
  };
}

OpGenerator shrSIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    auto operands = searchIntegerBinaryOperandsFrom(builder, loc, candidates,
                                                    "arith.shrSI");
    auto value =
        arith::ShRSIOp::create(builder, loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.shrSI");
    return value.getOperation();
  };
}

OpGenerator shrUIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    auto operands = searchIntegerBinaryOperandsFrom(builder, loc, candidates,
                                                    "arith.shrUI");
    auto value =
        arith::ShRUIOp::create(builder, loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.shrUI");
    return value.getOperation();
  };
}

OpGenerator subFGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getF32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands =
        searchFloatBinaryOperandsFrom(builder, loc, candidates, "arith.subF");
    auto value =
        arith::RemFOp::create(builder, loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.subF");
    return value.getOperation();
  };
}

OpGenerator subIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands =
        searchIntegerBinaryOperandsFrom(builder, loc, candidates, "arith.subI");
    auto value =
        arith::SubIOp::create(builder, loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.subI");
    return value.getOperation();
  };
}

OpGenerator xorIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands =
        searchIntegerBinaryOperandsFrom(builder, loc, candidates, "arith.xorI");
    auto value =
        arith::XOrIOp::create(builder, loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.xorI");
    return value.getOperation();
  };
}