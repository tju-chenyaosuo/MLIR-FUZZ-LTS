
#include "NodeGenerator.h"

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
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // auto typedValuePool = region.pool;
    // auto candidates = typedValuePool.getCandidatesFromIntOrFloats(builder, loc, builder.getF32Type());  // 取 INT FLOAT 或者 INDEX
    // auto operands = searchFloatBinaryOperandsFrom(builder, loc, candidates, "arith.addF");  // 取两个相同的operand

    // Get first operand 
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(float_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");

    mlir::Value value = arith::AddFOp::create(builder, loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator addIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    mlir::Value value = arith::AddIOp::create(builder, loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator andIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create the Operation
    mlir::Value value = arith::AndIOp::create(builder, loc, first, second);

    mb->add2Pool(value);
    return true;
  };
}

OpGenerator ceilDivSIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Generate operation
    mlir::Value value = arith::CeilDivSIOp::create(builder, loc, first, second);

    mb->add2Pool(value);
    return true;
  };
}

OpGenerator ceilDivUIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Generate Operation
    mlir::Value value = arith::CeilDivUIOp::create(builder, loc, first, second);

    mb->add2Pool(value);
    return true;
  };
}

OpGenerator cmpFGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(float_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Select Predicate
    arith::CmpFPredicate predicate = cmpFPredicates[rollIdx(cmpFPredicates.size())];
    // Generate peration
    mlir::Value value = arith::CmpFOp::create(builder, loc, predicate, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator cmpIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Select Predicate
    arith::CmpIPredicate predicate = cmpIPredicates[rollIdx(cmpIPredicates.size())];
    // create operation
    mlir::Value value = arith::CmpIOp::create(builder, loc, predicate, first, second);

    mb->add2Pool(value);
    return true;
  };
}

OpGenerator constantGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Type> supportedElementaryTypes = getSupportedIntOrFloatTypes(builder.getContext());
    auto elementType = supportedElementaryTypes[rollIdx(supportedElementaryTypes.size())];
    IntegerType iType = elementType.dyn_cast<IntegerType>();
    assert(elementType.isIntOrFloat());
    // Generation Logic
    if (iType) {
      unsigned width = min(iType.getWidth(), 32);
      long long val = rollIdx(((long long)1) << width);
      mlir::Value cons = arith::ConstantOp::create(builder, loc, builder.getIntegerAttr(elementType, val));
      mb->add2Pool(cons);
    } else {
      FloatType fType = elementType.dyn_cast<FloatType>();
      unsigned width = min(fType.getWidth(), 32);
      long long valf = rollIdx(((long long)1) << width);
      double val = static_cast<double>(valf);
      mlir::Value cons = arith::ConstantOp::create(builder, loc, builder.getFloatAttr(elementType, val));
      mb->add2Pool(cons);
    }
    return true;
  };
}

OpGenerator divFGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(float_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create the Operation
    mlir::Value value = arith::DivFOp::create(builder, loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator divSIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation
    mlir::Value value = arith::DivSIOp::create(builder, loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator divUIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation    
    mlir::Value value = arith::DivUIOp::create(builder, loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator floorDivSIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation
    mlir::Value value = arith::FloorDivSIOp::create(builder, loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

// OpGenerator maxFGenerator() {
//   return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
//     // Get first operand
//     std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
//     mb->search(float_filter, &candidates);
//     if (!candidates.size())
//       return false;
//     mlir::Value first = candidates[rollIdx(candidates.size())];
//     // Get second operand
//     std::string ty = getValueTypeStr(first);
//     if (!mb->valuePool.pool.count(ty))
//       return false;
//     mlir::Value second = mb->search(ty, "Unknown");
//     // Create Operation
//     auto value = arith::MaxFOp::create(builder, loc, first, second);
//     return true;
//   };
// }

OpGenerator maxSIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation    
    mlir::Value value = arith::MaxSIOp::create(builder, loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator maxUIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation
    mlir::Value value = arith::MaxUIOp::create(builder, loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

// OpGenerator minFGenerator() {
//   return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
//     // Get first operand
//     std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
//     mb->search(float_filter, &candidates);
//     if (!candidates.size())
//       return false;
//     mlir::Value first = candidates[rollIdx(candidates.size())];
//     // Get second operand
//     std::string ty = getValueTypeStr(first);
//     if (!mb->valuePool.pool.count(ty))
//       return false;
//     mlir::Value second = mb->search(ty, "Unknown");
//     // Create Opeartion
//     auto value = arith::MinFOp::create(builder, loc, first, second);
//     return true;
//   };
// }

OpGenerator minSIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation
    mlir::Value value = arith::MinSIOp::create(builder, loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator minUIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation    
    mlir::Value value = arith::MinUIOp::create(builder, loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator mulFGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(float_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation
    mlir::Value value = arith::MulFOp::create(builder, loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator mulIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Opeartion
    mlir::Value value = arith::MulIOp::create(builder, loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator negFGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(float_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Create Operation
    mlir::Value value = arith::NegFOp::create(builder, loc, first);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator orIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation
    mlir::Value value = arith::OrIOp::create(builder, loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator remFGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(float_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation
    mlir::Value value = arith::RemFOp::create(builder, loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator remSIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation
    mlir::Value value = arith::RemSIOp::create(builder, loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator remUIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Opeartion    
    mlir::Value value = arith::RemUIOp::create(builder, loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator shlIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Opeartion    
    mlir::Value value = arith::ShLIOp::create(builder, loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator shrSIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Opeartion 
    mlir::Value value = arith::ShRSIOp::create(builder, loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator shrUIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation
    mlir::Value value = arith::ShRUIOp::create(builder, loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator subFGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(float_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation
    mlir::Value value = arith::RemFOp::create(builder, loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator subIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation
    mlir::Value value = arith::SubIOp::create(builder, loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator xorIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation
    mlir::Value value = arith::XOrIOp::create(builder, loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}