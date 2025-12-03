//
// Created by Stan Wang on 2022/12/27.
//
#include "../../include/smith/generators/OpGeneration.h"

using namespace mlir;

OpGenerator indexAddGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.getCandidatesFromIndex(builder, loc);
    auto operand0 = sampleTypedValueFrom(candidates, "index.add");
    auto operand1 = sampleTypedValueFrom(candidates, "index.add");
    auto res = index::AddOp::create(builder, loc, operand0.val, operand1.val);
    region.pool.addIndex(TypeValue(res.getType(), res), "index.add");
    return res.getOperation();
  };
}

OpGenerator indexAndGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.getCandidatesFromIndex(builder, loc);
    auto operand0 = sampleTypedValueFrom(candidates, "index.and");
    auto operand1 = sampleTypedValueFrom(candidates, "index.and");
    auto res = index::AndOp::create(builder, loc, operand0.val, operand1.val);
    region.pool.addIndex(TypeValue(res.getType(), res), "index.and");
    return res.getOperation();
  };
}
OpGenerator indexBoolConstantGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto res = index::BoolConstantOp::create(builder, loc, UR(2));
    region.pool.addIntOrFloat(TypeValue(res.getType(), res),
                              "index.bool.constant");
    return res.getOperation();
  };
}

OpGenerator indexCastSGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    if (UR(2)) {
      auto intCandidates = region.pool.searchCandidatesFrom(
          {PoolType::IntOrFloat},
          [&](TypeValue tVal) { return mlir::isa<IntegerType>(tVal.type); });
      if (intCandidates.empty()) {
        intCandidates.push_back(
            region.pool.generateInteger(builder, loc, builder.getI32Type()));
      }
      auto operand = sampleTypedValueFrom(intCandidates, "index.casts");
      auto res = index::CastSOp::create(builder, loc, builder.getIndexType(),
                                                operand.val);
      region.pool.addIndex(TypeValue(res.getType(), res), "index.casts");
      return res.getOperation();
    } else {
      auto idxCandidates = region.pool.getCandidatesFromIndex(builder, loc);
      auto operand = sampleTypedValueFrom(idxCandidates, "index.casts");
      auto res = index::CastSOp::create(builder, loc, builder.getI32Type(),
                                                operand.val);
      region.pool.addIndex(TypeValue(res.getType(), res), "index.casts");
      return res.getOperation();
    }
  };
}

OpGenerator indexCastUGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    if (UR(2)) {
      auto intCandidates = region.pool.searchCandidatesFrom(
          {PoolType::IntOrFloat},
          [&](TypeValue tVal) { return mlir::isa<IntegerType>(tVal.type); });
      if (intCandidates.empty()) {
        intCandidates.push_back(
            region.pool.generateInteger(builder, loc, builder.getI32Type()));
      }
      auto operand = sampleTypedValueFrom(intCandidates, "index.castu");
      auto res = index::CastUOp::create(builder, loc, builder.getIndexType(),
                                                operand.val);
      region.pool.addIndex(TypeValue(res.getType(), res), "index.castu");
      return res.getOperation();
    } else {
      auto idxCandidates = region.pool.getCandidatesFromIndex(builder, loc);
      auto operand = sampleTypedValueFrom(idxCandidates, "index.castu");
      auto res = index::CastUOp::create(builder, loc, builder.getI32Type(),
                                                operand.val);
      region.pool.addIndex(TypeValue(res.getType(), res), "index.castu");
      return res.getOperation();
    }
  };
}

OpGenerator indexCeilDivSGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.getCandidatesFromIndex(builder, loc);
    auto operand0 = sampleTypedValueFrom(candidates, "index.ceildivs");
    auto operand1 = sampleTypedValueFrom(candidates, "index.ceildivs");
    auto res =
        index::CeilDivSOp::create(builder, loc, operand0.val, operand1.val);
    region.pool.addIndex(TypeValue(res.getType(), res), "index.ceildivs");
    return res.getOperation();
  };
}

OpGenerator indexCeilDivUGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.getCandidatesFromIndex(builder, loc);
    auto operand0 = sampleTypedValueFrom(candidates, "index.ceildivu");
    auto operand1 = sampleTypedValueFrom(candidates, "index.ceildivu");
    auto res =
        index::CeilDivUOp::create(builder, loc, operand0.val, operand1.val);
    region.pool.addIndex(TypeValue(res.getType(), res), "index.ceildivu");
    return res.getOperation();
  };
}

// OpGenerator indexCmpGenerator() {
//
//   return [](OpBuilder &builder, Location loc, OpRegion &region) {
//     IndexCmpPredicate::
//     auto candidates = region.pool.getCandidatesFromIndex(builder, loc);
//     auto operand0 = sampleTypedValueFrom(candidates);
//     auto operand1 = sampleTypedValueFrom(candidates);
//     auto res = index::CmpOp::create(builder, loc, operand0.val, operand1.val);
//     region.pool.addIndex(TypeValue(res.getType(), res), "index.ceildivu");
//   };
// }

OpGenerator indexConstantGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto res = index::ConstantOp::create(builder, loc, UR(8));
    region.pool.addIndex(TypeValue(res.getType(), res), "index.constant");
    return res.getOperation();
  };
}

OpGenerator indexDivSGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.getCandidatesFromIndex(builder, loc);
    auto operand0 = sampleTypedValueFrom(candidates, "index.divs");
    auto operand1 = sampleTypedValueFrom(candidates, "index.divs");
    auto res = index::DivSOp::create(builder, loc, operand0.val, operand1.val);
    region.pool.addIndex(TypeValue(res.getType(), res), "index.divs");
    return res.getOperation();
  };
}

OpGenerator indexDivUGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.getCandidatesFromIndex(builder, loc);
    auto operand0 = sampleTypedValueFrom(candidates, "index.divu");
    auto operand1 = sampleTypedValueFrom(candidates, "index.divu");
    auto res = index::DivUOp::create(builder, loc, operand0.val, operand1.val);
    region.pool.addIndex(TypeValue(res.getType(), res), "index.divu");
    return res.getOperation();
  };
}

OpGenerator indexFloorDivSGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.getCandidatesFromIndex(builder, loc);
    auto operand0 = sampleTypedValueFrom(candidates, "index.floordivs");
    auto operand1 = sampleTypedValueFrom(candidates, "index.floordivs");
    auto res =
        index::FloorDivSOp::create(builder, loc, operand0.val, operand1.val);
    region.pool.addIndex(TypeValue(res.getType(), res), "index.floordivs");
    return res.getOperation();
  };
}

OpGenerator indexMaxSGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.getCandidatesFromIndex(builder, loc);
    auto operand0 = sampleTypedValueFrom(candidates, "index.maxs");
    auto operand1 = sampleTypedValueFrom(candidates, "index.maxs");
    auto res = index::MaxSOp::create(builder, loc, operand0.val, operand1.val);
    region.pool.addIndex(TypeValue(res.getType(), res), "index.maxs");
    return res.getOperation();
  };
}

OpGenerator indexMaxUGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.getCandidatesFromIndex(builder, loc);
    auto operand0 = sampleTypedValueFrom(candidates, "index.maxu");
    auto operand1 = sampleTypedValueFrom(candidates, "index.maxu");
    auto res = index::MaxUOp::create(builder, loc, operand0.val, operand1.val);
    region.pool.addIndex(TypeValue(res.getType(), res), "index.maxu");
    return res.getOperation();
  };
}

OpGenerator indexMulGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.getCandidatesFromIndex(builder, loc);
    auto operand0 = sampleTypedValueFrom(candidates, "index.mul");
    auto operand1 = sampleTypedValueFrom(candidates, "index.mul");
    auto res = index::MulOp::create(builder, loc, operand0.val, operand1.val);
    region.pool.addIndex(TypeValue(res.getType(), res), "index.mul");
    return res.getOperation();
  };
}

OpGenerator indexOrGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.getCandidatesFromIndex(builder, loc);
    auto operand0 = sampleTypedValueFrom(candidates, "index.or");
    auto operand1 = sampleTypedValueFrom(candidates, "index.or");
    auto res = index::OrOp::create(builder, loc, operand0.val, operand1.val);
    region.pool.addIndex(TypeValue(res.getType(), res), "index.or");
    return res.getOperation();
  };
}

OpGenerator indexRemSGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.getCandidatesFromIndex(builder, loc);
    auto operand0 = sampleTypedValueFrom(candidates, "index.rems");
    auto operand1 = sampleTypedValueFrom(candidates, "index.rems");
    auto res = index::RemSOp::create(builder, loc, operand0.val, operand1.val);
    region.pool.addIndex(TypeValue(res.getType(), res), "index.rems");
    return res.getOperation();
  };
}

OpGenerator indexRemUGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.getCandidatesFromIndex(builder, loc);
    auto operand0 = sampleTypedValueFrom(candidates, "index.remu");
    auto operand1 = sampleTypedValueFrom(candidates, "index.remu");
    auto res = index::RemUOp::create(builder, loc, operand0.val, operand1.val);
    region.pool.addIndex(TypeValue(res.getType(), res), "index.remu");
    return res.getOperation();
  };
}

OpGenerator indexShLGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.getCandidatesFromIndex(builder, loc);
    auto operand0 = sampleTypedValueFrom(candidates, "index.shl");
    auto operand1 = sampleTypedValueFrom(candidates, "index.shl");
    auto res = index::ShlOp::create(builder, loc, operand0.val, operand1.val);
    region.pool.addIndex(TypeValue(res.getType(), res), "index.shl");
    return res.getOperation();
  };
}

OpGenerator indexShrSGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.getCandidatesFromIndex(builder, loc);
    auto operand0 = sampleTypedValueFrom(candidates, "index.shrs");
    auto operand1 = sampleTypedValueFrom(candidates, "index.shrs");
    auto res = index::ShrSOp::create(builder, loc, operand0.val, operand1.val);
    region.pool.addIndex(TypeValue(res.getType(), res), "index.shrs");
    return res.getOperation();
  };
}

OpGenerator indexShrUGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.getCandidatesFromIndex(builder, loc);
    auto operand0 = sampleTypedValueFrom(candidates, "index.shru");
    auto operand1 = sampleTypedValueFrom(candidates, "index.shru");
    auto res = index::ShrUOp::create(builder, loc, operand0.val, operand1.val);
    region.pool.addIndex(TypeValue(res.getType(), res), "index.shru");
    return res.getOperation();
  };
}

OpGenerator indexSizeOfGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.getCandidatesFromIndex(builder, loc);
    auto res = index::SizeOfOp::create(builder, loc);
    region.pool.addIndex(TypeValue(res.getType(), res), "index.sizeof");
    return res.getOperation();
  };
}

OpGenerator indexSubGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.getCandidatesFromIndex(builder, loc);
    auto operand0 = sampleTypedValueFrom(candidates, "index.sub");
    auto operand1 = sampleTypedValueFrom(candidates, "index.sub");
    auto res = index::SubOp::create(builder, loc, operand0.val, operand1.val);
    region.pool.addIndex(TypeValue(res.getType(), res), "index.sub");
    return res.getOperation();
  };
}

OpGenerator indexXorGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.getCandidatesFromIndex(builder, loc);
    auto operand0 = sampleTypedValueFrom(candidates, "index.xor");
    auto operand1 = sampleTypedValueFrom(candidates, "index.xor");
    auto res = index::XOrOp::create(builder, loc, operand0.val, operand1.val);
    region.pool.addIndex(TypeValue(res.getType(), res), "index.xor");
    return res.getOperation();
  };
}