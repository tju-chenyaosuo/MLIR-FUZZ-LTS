//
// Created by Stan Wang on 2022/12/1.
//
#include "include/smith/RegionGeneration.h"
#include "include/smith/TypeGeneration.h"
#include "include/smith/generators/OpGeneration.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

OpGenerator vectorBroadcastGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.getCandidatesFromIntOrFloats(
        builder, loc, randomIntOrFloatType(builder.getContext()));
    auto operand = sampleTypedValueFrom(candidates, "vector.broadcast");
    auto vectorTy = randomVectorType(operand.type);
    auto vector =
        vector::BroadcastOp::create(builder, loc, vectorTy, operand.val);
    region.pool.addVector(TypeValue(vectorTy, vector), "vector.broadcast");
    return vector.getOperation();
  };
}

OpGenerator vectorBitCastGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector}, [&](TypeValue tval) {
          return mlir::dyn_cast<VectorType>(tval.type).getRank() > 0;
        });
    if (candidates.empty()) {
      SmallVector<int64_t> shape;
      shape.push_back(1);
      auto elemTy = randomIntOrFloatType(builder.getContext());
      auto vecTy = VectorType::get(shape, elemTy);
      candidates.push_back(region.pool.generateVector(builder, loc, vecTy));
    }
    auto operand = sampleTypedValueFrom(candidates, "vector.bitcast");
    auto srcShapeTy = mlir::dyn_cast<ShapedType>(operand.type);
    auto srcShape = srcShapeTy.getShape();
    auto srcElemTy = srcShapeTy.getElementType();
    auto srcElemTyWidth = srcElemTy.getIntOrFloatBitWidth();
    auto totalBitWidth = srcElemTyWidth;
    for (auto dimSize : srcShape) {
      totalBitWidth *= dimSize;
    }
    auto destElemTy = randomIntOrFloatType(builder.getContext());
    auto destElemTyWidth = destElemTy.getIntOrFloatBitWidth();

    SmallVector<int64_t> destShape;
    for (auto dimSize : srcShape) {
      destShape.push_back(dimSize);
    }
    if (destElemTyWidth != srcElemTyWidth) {
      destElemTy = srcElemTy;
    }
    auto resultTy = VectorType::get(destShape, destElemTy);
    auto vector =
        vector::BitCastOp::create(builder, loc, resultTy, operand.val);
    region.pool.addVector(TypeValue(resultTy, vector), "vector.bitcast");
    return vector.getOperation();
  };
}

OpGenerator vectorCompressStoreGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto memrefCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::StaticShapedMemref, PoolType::DynamicShapedMemref},
        emptyFilter());
    if (memrefCandidates.empty()) {
      memrefCandidates.push_back(typedValuePool.generateTypedValue(
          builder, loc, randomRankedMemrefType(builder.getContext())));
    }
    auto memref =
        sampleTypedValueFrom(memrefCandidates, "vector.compress_store");
    auto memrefTy = mlir::dyn_cast<ShapedType>(memref.type);
    auto elemTy = memrefTy.getElementType();

    auto vectorCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::Vector}, [&](TypeValue tval) {
          auto vType = mlir::dyn_cast<VectorType>(tval.type);
          return vType.getRank() == 1 && vType.getElementType() == elemTy;
        });
    if (vectorCandidates.empty()) {
      auto vectorTy = random1DVectorType(elemTy);
      vectorCandidates.push_back(typedValuePool.generateVector(
          builder, loc, mlir::dyn_cast<VectorType>(vectorTy)));
    }
    auto vectorOperand =
        sampleTypedValueFrom(vectorCandidates, "vector.compress_store");
    auto vecTy = mlir::dyn_cast<VectorType>(vectorOperand.type);

    auto maskCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::Vector}, [&](TypeValue tVal) {
          auto vType = mlir::dyn_cast<VectorType>(tVal.type);
          return vType.getElementType() == builder.getI1Type() &&
                 vType.getRank() == 1 &&
                 vType.getDimSize(0) == vecTy.getDimSize(0);
        });
    if (maskCandidates.empty()) {
      auto vectorTy = VectorType::get(
          mlir::dyn_cast<ShapedType>(vectorOperand.type).getShape(),
          builder.getI1Type());
      maskCandidates.push_back(
          typedValuePool.generateVector(builder, loc, vectorTy));
    }
    auto maskOperand =
        sampleTypedValueFrom(maskCandidates, "vector.compressstore");

    auto indices =
        region.pool.randomIndicesForShapedType(memrefTy, builder, loc);

    /**
     * It is strange that this function has reversed order of mask and vector...
     * I just change the order of these two parameters.
     */
    return vector::CompressStoreOp::create(builder, loc, memref.val, indices,
                                           vectorOperand.val, maskOperand.val /*vectorOperand.val */ )
        .getOperation();
  };
}

OpGenerator vectorConstantMaskGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto vectorTy = randomVectorType(builder.getI1Type());
    auto shape = mlir::dyn_cast<VectorType>(vectorTy).getShape();
    SmallVector<Attribute> maskDimSizes;
    for (auto dimSize : shape) {
      maskDimSizes.push_back(builder.getI64IntegerAttr(UR(dimSize) + 1));
    }
    // auto arrAttr = ArrayAttr::get(builder.getContext(), maskDimSizes);

    auto attr = mlir::vector::ConstantMaskKind::AllTrue;
    if (UR(2))
      attr = mlir::vector::ConstantMaskKind::AllFalse;
    auto val = vector::ConstantMaskOp::create(builder, loc, vectorTy, attr);
    region.pool.addVector(TypeValue(vectorTy, val), "vector.constant_mask");
    return val.getOperation();
  };
}

OpGenerator vectorCreateMaskGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto vectorTy = randomVectorType(builder.getI1Type());
    auto shape = mlir::dyn_cast<VectorType>(vectorTy).getShape();
    SmallVector<Attribute> maskDimSizes;
    auto indexCandidates = region.pool.getCandidatesFromIndex(builder, loc);
    SmallVector<Value> indices;
    for (size_t i = 0; i < shape.size(); ++i) {
      indices.push_back(
          sampleTypedValueFrom(indexCandidates, "vector.create_mask").val);
    }
    auto val = vector::CreateMaskOp::create(builder, loc, vectorTy, indices);
    region.pool.addVector(TypeValue(vectorTy, val), "vector.create_mask");
    return val.getOperation();
  };
}

std::vector<vector::CombiningKind> intKinds = {
    vector::CombiningKind::ADD,   vector::CombiningKind::MUL,
    vector::CombiningKind::MINUI, vector::CombiningKind::MINSI,
    vector::CombiningKind::MAXUI, vector::CombiningKind::MAXSI,
    vector::CombiningKind::AND,   vector::CombiningKind::OR,
    vector::CombiningKind::XOR};
std::vector<vector::CombiningKind> floatKinds = {
    vector::CombiningKind::ADD, vector::CombiningKind::MUL,
    vector::CombiningKind::MINNUMF, vector::CombiningKind::MAXNUMF};

OpGenerator vectorContractGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto pool = region.pool;
    auto lhsCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector}, [&](TypeValue tval) {
          return mlir::dyn_cast<VectorType>(tval.type).getRank() > 0;
        });
    if (lhsCandidates.empty()) {
      SmallVector<int64_t> shape;
      shape.push_back(1);
      auto elemTy = randomIntOrFloatType(builder.getContext());
      auto vecTy = VectorType::get(shape, elemTy);
      lhsCandidates.push_back(region.pool.generateVector(builder, loc, vecTy));
    }
    auto vectorLhs = sampleTypedValueFrom(lhsCandidates, "vector.contract");
    auto lhsVTy = mlir::dyn_cast<VectorType>(vectorLhs.type);
    auto rhsCandidates =
        pool.searchCandidatesFrom({PoolType::Vector}, [&](TypeValue tval) {
          if (mlir::dyn_cast<VectorType>(tval.type).getElementType() !=
              lhsVTy.getElementType()) {
            return false;
          }
          auto rhsVTy = mlir::dyn_cast<VectorType>(tval.type);
          for (int i = 0; i < lhsVTy.getRank(); i++) {
            for (int j = 0; j < rhsVTy.getRank(); ++j) {
              auto dimSizeLhs = lhsVTy.getDimSize(i);
              auto dimSizeRhs = rhsVTy.getDimSize(j);
              if (dimSizeLhs == dimSizeRhs) {
                return true;
              }
            }
          }
          return false;
        });
    if (rhsCandidates.empty()) {
      rhsCandidates.push_back(pool.generateVector(builder, loc, lhsVTy));
    }

    auto vectorRhs = sampleTypedValueFrom(rhsCandidates, "vector.contract");
    auto rhsVTy = mlir::dyn_cast<VectorType>(vectorRhs.type);

    SmallVector<AffineMap> indexingMaps;
    SmallVector<Attribute> iteratorTypes;

    int lhsReducePos;
    int rhsReducePos;
    // here we consider the most simple situation -- one dim is reduced.
    for (lhsReducePos = 0; lhsReducePos < lhsVTy.getRank(); lhsReducePos++) {
      bool flag = false;
      for (rhsReducePos = 0; rhsReducePos < rhsVTy.getRank(); rhsReducePos++) {
        auto dimSizeLhs = lhsVTy.getDimSize(lhsReducePos);
        auto dimSizeRhs = rhsVTy.getDimSize(rhsReducePos);
        if (dimSizeLhs == dimSizeRhs) {
          flag = true;
          break;
        }
      }
      if (flag) {
        break;
      }
    }

    SmallVector<int64_t> accShape;
    SmallVector<AffineExpr> leftExprs;
    SmallVector<AffineExpr> rightExprs;
    SmallVector<AffineExpr> accExprs;
    auto dimCount = lhsVTy.getRank() + rhsVTy.getRank();
    for (int i = 0; i < lhsReducePos; ++i) {
      auto exprI = builder.getAffineDimExpr(i);
      leftExprs.push_back(exprI);
      accExprs.push_back(exprI);
      accShape.push_back(lhsVTy.getDimSize(i));
      iteratorTypes.push_back(vector::IteratorTypeAttr::get(
          builder.getContext(), vector::IteratorType::parallel));
    }
    // The reduced dim
    leftExprs.push_back(builder.getAffineDimExpr(dimCount - 2));
    for (int i = lhsReducePos + 1; i < lhsVTy.getRank(); ++i) {
      // skip the reduced dim
      auto exprI = builder.getAffineDimExpr(i - 1);
      leftExprs.push_back(exprI);
      accExprs.push_back(exprI);
      accShape.push_back(lhsVTy.getDimSize(i));
      iteratorTypes.push_back(vector::IteratorTypeAttr::get(
          builder.getContext(), vector::IteratorType::parallel));
    }

    for (int i = 0; i < rhsReducePos; ++i) {
      auto exprI = builder.getAffineDimExpr(i + lhsVTy.getRank() - 1);
      rightExprs.push_back(exprI);
      accExprs.push_back(exprI);
      accShape.push_back(rhsVTy.getDimSize(i));
      iteratorTypes.push_back(vector::IteratorTypeAttr::get(
          builder.getContext(), vector::IteratorType::parallel));
    }
    // The reduced dim
    rightExprs.push_back(builder.getAffineDimExpr(dimCount - 2));
    for (int i = rhsReducePos + 1; i < rhsVTy.getRank(); ++i) {
      auto exprI = builder.getAffineDimExpr(i + lhsVTy.getRank() - 2);
      rightExprs.push_back(exprI);
      accExprs.push_back(exprI);
      accShape.push_back(rhsVTy.getDimSize(i));
      iteratorTypes.push_back(vector::IteratorTypeAttr::get(
          builder.getContext(), vector::IteratorType::parallel));
    }
    iteratorTypes.push_back(vector::IteratorTypeAttr::get(
        builder.getContext(), vector::IteratorType::reduction));

    indexingMaps.push_back(
        AffineMap::get(dimCount - 1, 0, leftExprs, builder.getContext()));
    indexingMaps.push_back(
        AffineMap::get(dimCount - 1, 0, rightExprs, builder.getContext()));
    indexingMaps.push_back(
        AffineMap::get(dimCount - 1, 0, accExprs, builder.getContext()));
    Type accTy;
    // For K = 0 (no free or batch dimensions), the accumulator and output are
    // a scalar.
    if (accExprs.empty()) {
      accTy = lhsVTy.getElementType();
    } else {
      accTy = VectorType::get(accShape, lhsVTy.getElementType());
    }

    auto accCandidates = pool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::Vector},
        [&](TypeValue tVal) { return tVal.type == accTy; });
    if (accCandidates.empty()) {
      accCandidates.push_back(pool.generateTypedValue(builder, loc, accTy));
    }
    auto accOperand = sampleTypedValueFrom(accCandidates, "vector.contract");

    std::vector<vector::CombiningKind> kinds =
        mlir::dyn_cast<IntegerType>(lhsVTy.getElementType()) ? intKinds
                                                             : floatKinds;
    auto kind = kinds[UR(kinds.size())];
    auto indexingMapAttr = builder.getAffineMapArrayAttr(indexingMaps);
    auto iteratorTypesAttr = builder.getArrayAttr(iteratorTypes);

    return vector::ContractionOp::create(
               builder, loc, vectorLhs.val, vectorRhs.val, accOperand.val,
               indexingMapAttr, iteratorTypesAttr, kind)
        .getOperation();
  };
}

OpGenerator vectorExpandLoadGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto memrefCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedMemref, PoolType::DynamicShapedMemref});
    if (memrefCandidates.empty()) {
      memrefCandidates.push_back(region.pool.generateRankedMemref(
          builder, loc, randomRankedMemrefType(builder.getContext())));
    }
    auto base = sampleTypedValueFrom(memrefCandidates, "vector.expand_load");
    auto elemTy = mlir::dyn_cast<MemRefType>(base.type).getElementType();
    auto passThruCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector}, [&](TypeValue tval) {
          return mlir::dyn_cast<VectorType>(tval.type).getRank() == 1 &&
                 mlir::dyn_cast<VectorType>(tval.type).getElementType() ==
                     mlir::dyn_cast<MemRefType>(base.type).getElementType();
        });
    if (passThruCandidates.empty()) {
      auto passThruTy = random1DVectorType(elemTy);
      passThruCandidates.push_back(
          region.pool.generateVector(builder, loc, passThruTy));
    }
    auto passThruVector =
        sampleTypedValueFrom(passThruCandidates, "vector.expand_load");
    auto resultTy = passThruVector.type;
    auto maskTy = VectorType::get(
        mlir::dyn_cast<VectorType>(resultTy).getShape(), builder.getI1Type());
    auto maskCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector},
        [&](TypeValue tval) { return tval.type == maskTy; });
    if (maskCandidates.empty()) {
      maskCandidates.push_back(
          region.pool.generateVector(builder, loc, maskTy));
    }
    auto maskVector =
        sampleTypedValueFrom(maskCandidates, "vector.expand_load");

    SmallVector<Value> indices;
    auto indexCandidates = region.pool.getCandidatesFromIndex(builder, loc);
    for (int i = 0; i < mlir::dyn_cast<MemRefType>(base.type).getRank(); ++i) {
      indices.push_back(
          sampleTypedValueFrom(indexCandidates, "vector.expand_load").val);
    }
    auto val = vector::ExpandLoadOp::create(
        builder, loc, mlir::dyn_cast<mlir::VectorType>(resultTy), base.val,
        mlir::ValueRange(indices), maskVector.val, passThruVector.val);
    auto tVal = TypeValue(resultTy, val);
    region.pool.addVector(tVal, "vector.expand_load");
    return val.getOperation();
  };
}

OpGenerator vectorExtractGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto vectorCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector}, [&](TypeValue tval) {
          return mlir::dyn_cast<VectorType>(tval.type).getRank() > 0;
        });
    if (vectorCandidates.empty()) {
      SmallVector<int64_t> shape;
      shape.push_back(1);
      auto elemTy = randomIntOrFloatType(builder.getContext());
      auto vecTy = VectorType::get(shape, elemTy);

      vectorCandidates.push_back(
          region.pool.generateVector(builder, loc, vecTy));
    }
    auto vectorOperand =
        sampleTypedValueFrom(vectorCandidates, "vector.extract");
    auto vectorTy = mlir::dyn_cast<VectorType>(vectorOperand.type);
    SmallVector<int64_t> indices;
    auto indicesSize =
        max(1, UR(vectorTy.getRank())); // at least extract 1 dim.
    for (int i = 0; i < indicesSize; ++i) {
      auto idx = UR(vectorTy.getDimSize(i));
      indices.push_back(idx);
    }
    auto res =
        vector::ExtractOp::create(builder, loc, vectorOperand.val, indices);
    auto tVal = TypeValue(res.getType(), res);
    region.pool.addVector(tVal, "vector.extract");
    return res.getOperation();
  };
}

OpGenerator vectorExtractElementGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto vectorCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector}, [&](TypeValue tval) {
          return mlir::dyn_cast<VectorType>(tval.type).getRank() <= 1;
        });
    if (vectorCandidates.empty()) {
      SmallVector<int64_t> shape;
      if (UR(2)) {
        shape.push_back(dimPool[UR(dimPool.size())]);
      }
      region.pool.generateVector(
          builder, loc,
          VectorType::get(shape, randomIntOrFloatType(builder.getContext())));
    }
    auto vector =
        sampleTypedValueFrom(vectorCandidates, "vector.extract_element");
    auto vectorTy = mlir::dyn_cast<VectorType>(vector.type);
    Value res;
    if (vectorTy.getRank() == 0) {
      // res = vector::ExtractElementOp::create(builder, loc, vector.val);
      res = vector::ExtractOp::create(builder, loc, vector.val);
    } else {
      auto idxCandidates = region.pool.getCandidatesFromIndex(builder, loc);
      auto position =
          sampleTypedValueFrom(idxCandidates, "vector.extract_element");
      // res = vector::ExtractElementOp::create(builder, loc, vector.val,
      // position.val);
      res = vector::ExtractOp::create(builder, loc, vector.val, position.val);
    }
    auto tVal = TypeValue(vectorTy.getElementType(), res);
    region.pool.addElement(tVal, "vector.extract_element");
    return res.getDefiningOp();
  };
}

// Inference works as follows:
//   1. Add 'sizes' from prefix of dims in 'offsets'.
//   2. Add sizes from 'vectorType' for remaining dims.
static Type inferStridedSliceOpResultType(VectorType vectorType,
                                          SmallVector<int64_t> offsets,
                                          SmallVector<int64_t> sizes,
                                          SmallVector<int64_t> strides) {
  assert(offsets.size() == sizes.size() && offsets.size() == strides.size());
  SmallVector<int64_t, 4> shape;
  shape.reserve(vectorType.getRank());
  unsigned idx = 0;
  for (unsigned e = offsets.size(); idx < e; ++idx)
    shape.push_back(sizes[idx]);
  for (unsigned e = vectorType.getShape().size(); idx < e; ++idx)
    shape.push_back(vectorType.getShape()[idx]);

  return VectorType::get(shape, vectorType.getElementType());
}

OpGenerator vectorExtractStridedSliceGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto vectorCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector}, [&](TypeValue tval) {
          return mlir::dyn_cast<VectorType>(tval.type).getRank() > 0;
        });
    if (vectorCandidates.empty()) {
      SmallVector<int64_t> shape;
      shape.push_back(1);
      auto elemTy = randomIntOrFloatType(builder.getContext());
      auto vecTy = VectorType::get(shape, elemTy);
      vectorCandidates.push_back(
          region.pool.generateVector(builder, loc, vecTy));
    }
    auto srcVector =
        sampleTypedValueFrom(vectorCandidates, "vector.extract_strided_slice");
    auto vectorTy = mlir::dyn_cast<VectorType>(srcVector.type);
    auto sliceDimSize = max(1, UR(vectorTy.getRank()));
    SmallVector<int64_t> offsets;
    SmallVector<int64_t> sizes;
    SmallVector<int64_t> strides;
    for (int i = 0; i < sliceDimSize; ++i) {
      int64_t offset = UR(max(vectorTy.getDimSize(i) - 1, 1));
      offsets.push_back(offset);
      int64_t size = UR(vectorTy.getDimSize(i) - offset) + 1;
      sizes.push_back(size);
      strides.push_back(1);
    }
    auto resultTy =
        inferStridedSliceOpResultType(vectorTy, offsets, sizes, strides);
    auto res = vector::ExtractStridedSliceOp::create(
        builder, loc, srcVector.val, offsets, sizes, strides);
    auto tVal = TypeValue(resultTy, res);
    region.pool.addVector(tVal, "vector.extract_strided_slice");
    return res.getOperation();
  };
}

OpGenerator vectorFMAGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto vectorCandidates = region.pool.getCandidatesFromVector(
        builder, loc, randomVectorType(builder.getContext()));
    auto lhsCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector},
        [&](TypeValue tval) { return mlir::isa<FloatType>(tval.type); });
    if (lhsCandidates.empty()) {
      lhsCandidates.push_back(region.pool.generateVector(
          builder, loc,
          randomVectorType(mlir::Float32Type::get(builder.getContext()))));
    }
    auto lhsOperand = sampleTypedValueFrom(lhsCandidates, "vector.fma");
    auto lhsVTy = mlir::dyn_cast<VectorType>(lhsOperand.type);
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector},
        [&](TypeValue typeValue) { return lhsVTy == typeValue.type; });
    auto rhsOperand = sampleTypedValueFrom(candidates, "vector.fma");
    auto accOperand = sampleTypedValueFrom(candidates, "vector.fma");
    auto res = vector::FMAOp::create(builder, loc, lhsOperand.val,
                                     rhsOperand.val, accOperand.val);
    auto tVal = TypeValue(lhsVTy, res);
    region.pool.addVector(tVal, "vector.fma");
    return res.getOperation();
  };
}

// This operation has been removed to the LLVM Dialect.
OpGenerator vectorFlatTransposeGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto vectorCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector}, [&](TypeValue tVal) {
          return mlir::dyn_cast<VectorType>(tVal.type).getRank() == 1;
        });
    if (vectorCandidates.empty()) {
      vectorCandidates.push_back(region.pool.generateTypedValue(
          builder, loc, random1DVectorType(builder.getContext())));
    }
    auto source = sampleTypedValueFrom(
        vectorCandidates,
        "llvm.intr.matrix.transpose"); // vector.flat_transpose
    auto vectorTy = mlir::dyn_cast<VectorType>(source.type);
    auto size = vectorTy.getDimSize(0);
    int mid = (int)ceil(sqrt(size));
    int factor;
    for (factor = mid; factor > 0; factor--) {
      if (size % factor == 0) {
        break;
      }
    }
    auto rows = IntegerAttr::get(builder.getI32Type(), factor);
    auto columns = IntegerAttr::get(builder.getI32Type(), size / factor);
    auto val = mlir::LLVM::MatrixTransposeOp::create(builder, loc, vectorTy,
                                                     source.val, rows, columns);
    region.pool.addVector(
        TypeValue(vectorTy, val),
        "llvm.intr.matrix.transpose"); // vector.flat_transpose
    return val.getOperation();
  };
}

OpGenerator vectorGatherGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto baseTy = mlir::dyn_cast<ShapedType>(
        randomMemrefOrRankedTensorType(builder.getContext()));
    auto baseElemTy = baseTy.getElementType();
    auto baseCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::StaticShapedMemref},
        [&](TypeValue tval) { return tval.type == baseTy; });
    if (baseCandidates.empty()) {
      baseCandidates.push_back(
          region.pool.generateTypedValue(builder, loc, baseTy));
    }

    auto base = sampleTypedValueFrom(baseCandidates, "vector.gather");

    auto resTy = randomVectorType(baseElemTy);
    auto passThruTy = resTy;

    auto passThruCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector},
        [&](TypeValue tval) { return tval.type == passThruTy; });
    if (passThruCandidates.empty()) {
      passThruCandidates.push_back(
          region.pool.generateVector(builder, loc, passThruTy));
    }
    auto passThru = sampleTypedValueFrom(passThruCandidates, "vector.gather");

    auto maskTy = VectorType::get(passThruTy.getShape(), builder.getI1Type());
    auto maskCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector},
        [&](TypeValue tval) { return tval.type == maskTy; });
    if (maskCandidates.empty()) {
      maskCandidates.push_back(
          region.pool.generateVector(builder, loc, maskTy));
    }
    auto mask = sampleTypedValueFrom(maskCandidates, "vector.gather");

    auto indexVecTy =
        VectorType::get(passThruTy.getShape(), builder.getI32Type());
    auto indexVecCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector},
        [&](TypeValue tval) { return tval.type == indexVecTy; });
    if (indexVecCandidates.empty()) {
      indexVecCandidates.push_back(
          region.pool.generateVector(builder, loc, indexVecTy));
    }
    auto indexVec = sampleTypedValueFrom(indexVecCandidates, "vector.gather");

    auto idxCandidates = region.pool.getCandidatesFromIndex(builder, loc);
    SmallVector<Value> indices;
    for (int i = 0; i < baseTy.getRank(); ++i) {
      indices.push_back(
          sampleTypedValueFrom(idxCandidates, "vector.gather").val);
    }
    auto res = vector::GatherOp::create(builder, loc, resTy, base.val, indices,
                                        indexVec.val, mask.val, passThru.val);
    auto tval = TypeValue(resTy, res);
    region.pool.addVector(tval, "vector.gather");
    return res.getOperation();
  };
}

OpGenerator vectorInsertElementGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto vectorCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector}, [&](TypeValue tval) {
          return mlir::dyn_cast<VectorType>(tval.type).getRank() <= 1;
        });
    if (vectorCandidates.empty()) {
      SmallVector<int64_t> shape;
      if (UR(2)) {
        shape.push_back(dimPool[UR(dimPool.size())]);
      }
      vectorCandidates.push_back(region.pool.generateVector(
          builder, loc,
          VectorType::get(shape, randomIntOrFloatType(builder.getContext()))));
    }
    auto vector =
        sampleTypedValueFrom(vectorCandidates, "vector.insert_element");
    auto vectorTy = mlir::dyn_cast<VectorType>(vector.type);

    auto elemCandidates = region.pool.searchCandidatesFrom(
        {PoolType::IntOrFloat},
        [&](TypeValue tval) { return tval.type == vectorTy.getElementType(); });
    if (elemCandidates.empty()) {
      elemCandidates.push_back(
          region.pool.generateElement(builder, loc, vectorTy.getElementType()));
    }
    auto source = sampleTypedValueFrom(elemCandidates, "vector.insert_element");
    if (vectorTy.getRank() == 0) {
      // return vector::InsertElementOp::create(builder, loc,
      // source.val,vector.val).getOperation();
      return vector::InsertOp::create(builder, loc, source.val, vector.val)
          .getOperation();
    } else {
      auto idxCandidates = region.pool.getCandidatesFromIndex(builder, loc);
      auto position =
          sampleTypedValueFrom(idxCandidates, "vector.insert_element");
      // return vector::InsertElementOp::create(builder, loc,
      // source.val,vector.val, position.val).getOperation();
      return vector::InsertOp::create(builder, loc, source.val, vector.val,
                                      position.val)
          .getOperation();
    }
  };
}

OpGenerator vectorInsertGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto vectorCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector}, [&](TypeValue tval) {
          return mlir::dyn_cast<VectorType>(tval.type).getRank() > 0;
        });
    if (vectorCandidates.empty()) {
      SmallVector<int64_t> shape;
      shape.push_back(1);
      auto elemTy = randomIntOrFloatType(builder.getContext());
      auto vecTy = VectorType::get(shape, elemTy);
      vectorCandidates.push_back(
          region.pool.generateVector(builder, loc, vecTy));
    }
    auto vectorOperand =
        sampleTypedValueFrom(vectorCandidates, "vector.insert");
    auto vectorTy = mlir::dyn_cast<VectorType>(vectorOperand.type);
    SmallVector<int64_t> indices;
    auto indicesSize =
        max(UR(vectorTy.getRank()), 1); // at least extract 1 dim.
    for (int i = 0; i < indicesSize; ++i) {
      indices.push_back(UR(vectorTy.getDimSize(i)));
    }
    Type sourceTy;
    if ((int)indices.size() == vectorTy.getRank()) {
      sourceTy = vectorTy.getElementType();
    } else {
      auto n = std::min<size_t>(indicesSize, vectorTy.getRank() - 1);
      sourceTy = VectorType::get(vectorTy.getShape().drop_front(n),
                                 vectorTy.getElementType());
    }

    auto srcCandidates = region.pool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::Vector},
        [&](TypeValue tval) { return tval.type == sourceTy; });
    if (srcCandidates.empty()) {
      srcCandidates.push_back(
          region.pool.generateTypedValue(builder, loc, sourceTy));
    }
    auto src = sampleTypedValueFrom(srcCandidates, "vector.insert");
    return vector::InsertOp::create(builder, loc, src.val, vectorOperand.val,
                                    indices)
        .getOperation();
  };
}

// TODO
// 'vector.insert_strided_slice' op expected sum(offsets, source vector shape)
// dimension 1 to be confined to [1, 11)
//  toy.gen
//  ^
// src.mlir:2:3: note: see current operation: %3894 =
// "vector.insert_strided_slice"(%3893, %3889) {offsets = [1, 0], strides = [1]}
// : (vector<12xi1>, vector<15x10xi1>) -> vector<15x10xi1>
OpGenerator vectorInsertStridedSliceGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto vectorCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector}, [&](TypeValue tval) {
          return mlir::dyn_cast<VectorType>(tval.type).getRank() > 0;
        });
    if (vectorCandidates.empty()) {
      SmallVector<int64_t> shape;
      shape.push_back(1);
      auto elemTy = randomIntOrFloatType(builder.getContext());
      auto vecTy = VectorType::get(shape, elemTy);
      vectorCandidates.push_back(
          region.pool.generateVector(builder, loc, vecTy));
    }
    auto destVector =
        sampleTypedValueFrom(vectorCandidates, "vector.insert_strided_slice");
    auto vectorTy = mlir::dyn_cast<VectorType>(destVector.type);
    auto sliceDimSize = UR(vectorTy.getRank()) + 1;
    SmallVector<int64_t> offsets;
    SmallVector<int64_t> sizes;
    SmallVector<int64_t> strides;
    for (int i = 0; i < sliceDimSize; ++i) {
      int64_t offset = UR(max(vectorTy.getDimSize(i) - 1, 1));
      offsets.push_back(offset);
      int64_t size = UR(vectorTy.getDimSize(i) - offset) + 1;
      sizes.push_back(size);
      strides.push_back(1);
    }
    for (int i = sliceDimSize; i < vectorTy.getRank(); ++i) {
      offsets.push_back(1);
    }
    auto srcTy = VectorType::get(sizes, vectorTy.getElementType());
    auto srcCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector}, [&](TypeValue tval) { return tval.type == srcTy; });
    if (srcCandidates.empty()) {
      srcCandidates.push_back(region.pool.generateVector(
          builder, loc, mlir::dyn_cast<VectorType>(srcTy)));
    }
    auto srcVector =
        sampleTypedValueFrom(srcCandidates, "vector.insert_strided_slice");
    return vector::InsertStridedSliceOp::create(
               builder, loc, srcVector.val, destVector.val, offsets, strides)
        .getOperation();
  };
}

OpGenerator vectorLoadGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto memrefCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedMemref, PoolType::DynamicShapedMemref});
    if (memrefCandidates.empty()) {
      memrefCandidates.push_back(region.pool.generateRankedMemref(
          builder, loc, randomRankedMemrefType(builder.getContext())));
    }
    auto memref = sampleTypedValueFrom(memrefCandidates, "vector.load");
    auto memrefTy = mlir::dyn_cast<ShapedType>(memref.type);
    auto indices =
        region.pool.randomIndicesForShapedType(memrefTy, builder, loc);
    
    /**
     * To fix this:
     * error: 'vector.load' op destination memref has lower rank than the result vector
     */
    // auto vectorTy = randomVectorType(memrefTy.getElementType());
    auto vectorTy = littleOrEqualRankVectorType(memrefTy);
    auto res =
        vector::LoadOp::create(builder, loc, vectorTy, memref.val, indices);
    region.pool.addVector(TypeValue(vectorTy, res), "vector.load");
    return res.getOperation();
  };
}

OpGenerator vectorMaskGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    OpRegion maskOpRegion("vector.mask", region.depth + 1);

    // search operator for maskable op
    auto opFilter = OpNameFilter(maskableOps);
    auto vectorCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector}, [&](TypeValue tval) {
          return mlir::dyn_cast<VectorType>(tval.type).getRank() > 0;
        });
    if (vectorCandidates.empty()) {
      SmallVector<int64_t> shape;
      shape.push_back(1);
      auto elemTy = randomIntOrFloatType(builder.getContext());
      auto vecTy = VectorType::get(shape, elemTy);
      vectorCandidates.push_back(
          region.pool.generateVector(builder, loc, vecTy));
    }
    auto operand = sampleTypedValueFrom(vectorCandidates, "vector.mask");
    Operation *maskableOp;
    if (true) {
      auto vTy = mlir::dyn_cast<VectorType>(operand.type);
      SmallVector<bool> reductionMask;
      SmallVector<int64_t> newShape;
      reductionMask.push_back(false);
      newShape.push_back(vTy.getDimSize(0));
      for (int i = 1; i < vTy.getRank(); ++i) {
        if (UR(2)) {
          reductionMask.push_back(true);
        } else {
          reductionMask.push_back(false);
          newShape.push_back(vTy.getDimSize(i));
        }
      }

      auto accType =
          newShape.empty()
              ? vTy.getElementType()
              : (Type)VectorType::get(newShape, vTy.getElementType());
      auto accCandidates = region.pool.searchCandidatesFrom(
          {PoolType::IntOrFloat, PoolType::Vector},
          [&](TypeValue tVal) { return tVal.type == accType; });
      if (accCandidates.empty()) {
        accCandidates.push_back(
            region.pool.generateTypedValue(builder, loc, accType));
      }
      auto acc = sampleTypedValueFrom(accCandidates, "vector.mask");

      std::vector<vector::CombiningKind> kinds =
          mlir::dyn_cast<IntegerType>(vTy.getElementType()) ? intKinds
                                                            : floatKinds;
      auto kind = kinds[UR(kinds.size())];

      maskableOp = vector::MultiDimReductionOp::create(
                       builder, loc, operand.val, acc.val, reductionMask, kind)
                       .getOperation();
    }
    auto regionBuilder = [](OpBuilder &b, Operation *maskableOp) {
      vector::createMaskOpRegion(b, maskableOp);
    };
    auto maskTy =
        VectorType::get(mlir::dyn_cast<ShapedType>(operand.type).getShape(),
                        builder.getI1Type());
    auto maskCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector},
        [&](TypeValue tval) { return tval.type == maskTy; });
    if (maskCandidates.empty()) {
      maskCandidates.push_back(
          region.pool.generateVector(builder, loc, maskTy));
    }
    auto mask = sampleTypedValueFrom(maskCandidates, "vector.mask");
    Value res =
        vector::MaskOp::create(builder, loc,
                               TypeRange({maskableOp->getResult(0).getType()}),
                               mask.val, maskableOp, regionBuilder)
            .getResult(0);

    return res.getDefiningOp();
  };
}

OpGenerator vectorMaskedLoadGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto memrefCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedMemref, PoolType::DynamicShapedMemref});
    if (memrefCandidates.empty()) {
      memrefCandidates.push_back(region.pool.generateRankedMemref(
          builder, loc, randomRankedMemrefType(builder.getContext())));
    }
    auto memref = sampleTypedValueFrom(memrefCandidates, "vector.masked_load");
    auto memrefTy = mlir::dyn_cast<ShapedType>(memref.type);
    auto indices =
        region.pool.randomIndicesForShapedType(memrefTy, builder, loc);

    auto vectorTy = random1DVectorType(memrefTy.getElementType());

    auto passThruCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector},
        [&](TypeValue tVal) { return tVal.type == vectorTy; });
    if (passThruCandidates.empty()) {
      passThruCandidates.push_back(
          region.pool.generateVector(builder, loc, vectorTy));
    }
    auto passThru =
        sampleTypedValueFrom(passThruCandidates, "vector.masked_load");

    auto maskTy = VectorType::get(vectorTy.getShape(), builder.getI1Type());
    auto maskCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector},
        [&](TypeValue tVal) { return tVal.type == maskTy; });
    if (maskCandidates.empty()) {
      maskCandidates.push_back(
          region.pool.generateVector(builder, loc, maskTy));
    }
    auto mask = sampleTypedValueFrom(maskCandidates, "vector.masked_load");

    auto res = vector::MaskedLoadOp::create(builder, loc, vectorTy, memref.val,
                                            indices, mask.val, passThru.val);
    region.pool.addVector(TypeValue(vectorTy, res), "vector.masked_load");

    return res.getOperation();
  };
}

OpGenerator vectorStoreGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto memrefCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedMemref, PoolType::DynamicShapedMemref});
    if (memrefCandidates.empty()) {
      memrefCandidates.push_back(region.pool.generateRankedMemref(
          builder, loc, randomRankedMemrefType(builder.getContext())));
    }
    auto memref = sampleTypedValueFrom(memrefCandidates, "vector.store");
    auto memrefTy = mlir::dyn_cast<ShapedType>(memref.type);
    auto indices =
        region.pool.randomIndicesForShapedType(memrefTy, builder, loc);
    auto vectorTy = randomVectorType(memrefTy.getElementType());

    auto srcCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector},
        [&](TypeValue tVal) { return tVal.type == vectorTy; });
    if (srcCandidates.empty()) {
      srcCandidates.push_back(
          region.pool.generateVector(builder, loc, vectorTy));
    }
    auto src = sampleTypedValueFrom(srcCandidates, "vector.store");
    return vector::StoreOp::create(builder, loc, src.val, memref.val, indices)
        .getOperation();
  };
}

OpGenerator vectorMaskedStoreGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto memrefCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedMemref, PoolType::DynamicShapedMemref});
    if (memrefCandidates.empty()) {
      memrefCandidates.push_back(region.pool.generateRankedMemref(
          builder, loc, randomRankedMemrefType(builder.getContext())));
    }
    auto memref = sampleTypedValueFrom(memrefCandidates, "vector.masked_store");
    auto memrefTy = mlir::dyn_cast<ShapedType>(memref.type);
    auto indices =
        region.pool.randomIndicesForShapedType(memrefTy, builder, loc);
    auto vectorTy = random1DVectorType(memrefTy.getElementType());

    auto srcCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector},
        [&](TypeValue tVal) { return tVal.type == vectorTy; });
    if (srcCandidates.empty()) {
      srcCandidates.push_back(
          region.pool.generateVector(builder, loc, vectorTy));
    }
    auto src = sampleTypedValueFrom(srcCandidates, "vector.masked_store");

    auto maskTy = VectorType::get(vectorTy.getShape(), builder.getI1Type());
    auto maskCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector},
        [&](TypeValue tVal) { return tVal.type == maskTy; });
    if (maskCandidates.empty()) {
      maskCandidates.push_back(
          region.pool.generateVector(builder, loc, maskTy));
    }
    auto mask = sampleTypedValueFrom(maskCandidates, "vector.masked_store");
    return vector::MaskedStoreOp::create(builder, loc, memref.val, indices,
                                         mask.val, src.val)
        .getOperation();
  };
}

int gcd(int a, int b) {
  int m = a > b ? a : b;
  int n = a < b ? a : b;

  int r = m % n;
  while (r != 0) {
    m = n;
    n = r;
    r = m % n;
  }
  return n;
}

// Has Been Removed to LLVM Dialect
OpGenerator vectorMatrixMultiplyGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto lhsCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector}, [&](TypeValue tval) {
          return mlir::dyn_cast<VectorType>(tval.type).getRank() == 1;
        });
    if (lhsCandidates.empty()) {
      lhsCandidates.push_back(region.pool.generateVector(
          builder, loc, random1DVectorType(builder.getContext())));
    }
    // auto lhs = sampleTypedValueFrom(lhsCandidates, "vector.matmul");
    auto lhs = sampleTypedValueFrom(lhsCandidates, "llvm.intr.matrix.multiply");
    auto lhsVTy = mlir::dyn_cast<VectorType>(lhs.type);

    auto rhsCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector}, [&](TypeValue tVal) {
          auto vTy = mlir::dyn_cast<VectorType>(tVal.type);
          return vTy.getRank() == 1 &&
                 vTy.getElementType() == lhsVTy.getElementType();
        });
    if (rhsCandidates.empty()) {
      rhsCandidates.push_back(region.pool.generateVector(
          builder, loc, random1DVectorType(lhsVTy.getElementType())));
    }
    // auto rhs = sampleTypedValueFrom(rhsCandidates, "vector.matmul");
    auto rhs = sampleTypedValueFrom(rhsCandidates, "llvm.intr.matrix.multiply");
    auto rhsVTy = mlir::dyn_cast<VectorType>(rhs.type);

    auto reducedDimSize = gcd(lhsVTy.getDimSize(0), rhsVTy.getDimSize(0));

    SmallVector<int64_t> resShape;
    resShape.push_back(lhsVTy.getDimSize(0) / reducedDimSize *
                       rhsVTy.getDimSize(0) / reducedDimSize);

    auto resVTy = VectorType::get(resShape, lhsVTy.getElementType());
    // auto val = mlir::vector::MatmulOp::create(
    //     builder, loc, lhs.val, rhs.val, lhsVTy.getDimSize(0) /
    //     reducedDimSize, reducedDimSize, rhsVTy.getDimSize(0) /
    //     reducedDimSize);

    auto val = mlir::LLVM::MatrixMultiplyOp::create(
        builder, loc, resVTy, lhs.val, rhs.val, lhsVTy.getDimSize(0) / reducedDimSize,
        reducedDimSize, rhsVTy.getDimSize(0) / reducedDimSize);
    // region.pool.addVector(TypeValue(resVTy, val), "vector.matmul");
    region.pool.addVector(TypeValue(resVTy, val), "llvm.intr.matrix.multiply");

    return val.getOperation();
  };
}

OpGenerator vectorMultiReductionGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto vectorCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector}, [&](TypeValue tval) {
          return mlir::dyn_cast<VectorType>(tval.type).getRank() > 0;
        });
    if (vectorCandidates.empty()) {
      SmallVector<int64_t> shape;
      shape.push_back(1);
      auto elemTy = randomIntOrFloatType(builder.getContext());
      auto vecTy = VectorType::get(shape, elemTy);
      vectorCandidates.push_back(
          region.pool.generateVector(builder, loc, vecTy));
    }
    auto operand =
        sampleTypedValueFrom(vectorCandidates, "vector.multi_reduction");
    auto vTy = mlir::dyn_cast<VectorType>(operand.type);
    SmallVector<bool> reductionMask;
    SmallVector<int64_t> newShape;
    for (int i = 0; i < vTy.getRank(); ++i) {
      if (UR(2)) {
        reductionMask.push_back(true);
      } else {
        reductionMask.push_back(false);
        newShape.push_back(vTy.getDimSize(i));
      }
    }

    auto accType = newShape.empty()
                       ? vTy.getElementType()
                       : (Type)VectorType::get(newShape, vTy.getElementType());
    auto accCandidates = region.pool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::Vector},
        [&](TypeValue tVal) { return tVal.type == accType; });
    if (accCandidates.empty()) {
      accCandidates.push_back(
          region.pool.generateTypedValue(builder, loc, accType));
    }
    auto acc = sampleTypedValueFrom(accCandidates, "vector.multi_reduction");

    std::vector<vector::CombiningKind> kinds =
        mlir::dyn_cast<IntegerType>(vTy.getElementType()) ? intKinds
                                                          : floatKinds;
    auto kind = kinds[UR(kinds.size())];

    auto val = vector::MultiDimReductionOp::create(
        builder, loc, operand.val, acc.val, reductionMask, kind);
    region.pool.addTypeValue(TypeValue(accType, val), "vector.multi_reduction");
    return val.getOperation();
  };
}

OpGenerator vectorOuterProductGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto pool = region.pool;
    auto lhsCandidates =
        pool.searchCandidatesFrom({PoolType::Vector}, [&](TypeValue tval) {
          return mlir::dyn_cast<VectorType>(tval.type).getRank() == 1;
        });
    if (lhsCandidates.empty()) {
      lhsCandidates.push_back(pool.generateVector(
          builder, loc, random1DVectorType(builder.getContext())));
    }
    auto lhs = sampleTypedValueFrom(lhsCandidates, "vector.outer_product");
    auto lhsElemTy = mlir::dyn_cast<VectorType>(lhs.type).getElementType();

    auto rhsCandidates =
        pool.searchCandidatesFrom({PoolType::Vector}, [&](TypeValue tval) {
          return tval.type == lhs.type;
        });
    if (rhsCandidates.empty()) {
      rhsCandidates.push_back(
          pool.generateVector(builder, loc, random1DVectorType(lhsElemTy)));
    }
    auto rhs = sampleTypedValueFrom(rhsCandidates, "vector.outer_product");

    SmallVector<int64_t> accShape;
    accShape.push_back(mlir::dyn_cast<VectorType>(lhs.type).getDimSize(0));
    accShape.push_back(mlir::dyn_cast<VectorType>(rhs.type).getDimSize(0));
    auto accTy = VectorType::get(accShape, lhsElemTy);
    auto accCandidates = pool.searchCandidatesFrom(
        {PoolType::Vector}, [&](TypeValue tval) { return tval.type == accTy; });
    if (accCandidates.empty()) {
      accCandidates.push_back(pool.generateVector(builder, loc, accTy));
    }
    auto acc = sampleTypedValueFrom(accCandidates, "vector.outer_product");

    std::vector<vector::CombiningKind> kinds =
        mlir::dyn_cast<IntegerType>(lhsElemTy) ? intKinds : floatKinds;
    auto kind = kinds[UR(kinds.size())];

    auto res = vector::OuterProductOp::create(builder, loc, acc.type, lhs.val,
                                              rhs.val, acc.val, kind);
    pool.addVector(TypeValue(accTy, res), "vector.outer_product");
    return res.getOperation();
  };
}

OpGenerator vectorPrintGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto vectorCandidates = region.pool.getCandidatesFromVector(
        builder, loc, randomVectorType(builder.getContext()));
    auto operand = sampleTypedValueFrom(vectorCandidates, "vector.print");
    return vector::PrintOp::create(builder, loc, operand.val).getOperation();
  };
}

OpGenerator vectorReductionGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto pool = region.pool;
    auto candidates =
        pool.searchCandidatesFrom({PoolType::Vector}, [&](TypeValue tval) {
          return mlir::dyn_cast<VectorType>(tval.type).getRank() == 1;
        });
    if (candidates.empty()) {
      candidates.push_back(pool.generateVector(
          builder, loc, random1DVectorType(builder.getContext())));
    }
    auto src = sampleTypedValueFrom(candidates, "vector.reduction");

    std::vector<vector::CombiningKind> kinds =
        mlir::dyn_cast<IntegerType>(
            mlir::dyn_cast<VectorType>(src.type).getElementType())
            ? intKinds
            : floatKinds;
    auto kind = kinds[UR(kinds.size())];

    auto resTy = mlir::dyn_cast<VectorType>(src.type).getElementType();
    auto res = vector::ReductionOp::create(builder, loc, kind, src.val);
    pool.addIntOrFloat(TypeValue(resTy, res), "vector.reduction");
    return res.getOperation();
  };
}

OpGenerator vectorScanGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto pool = region.pool;

    auto srcCandidates =
        pool.searchCandidatesFrom({PoolType::Vector}, [&](TypeValue tval) {
          return mlir::dyn_cast<VectorType>(tval.type).getRank() > 1;
        });
    if (srcCandidates.empty()) {
      int rank = max(UR(rank_ub) + 1, 2);
      SmallVector<int64_t> shape;
      for (int i = 0; i < rank; ++i) {
        shape.push_back(dimPool[UR(dimPool.size())]);
      }
      auto vTy =
          VectorType::get(shape, randomIntOrFloatType(builder.getContext()));
      srcCandidates.push_back(pool.generateVector(builder, loc, vTy));
    }
    auto src = sampleTypedValueFrom(srcCandidates, "vector.scan");
    auto srcVTy = mlir::dyn_cast<VectorType>(src.type);

    std::vector<vector::CombiningKind> kinds =
        mlir::dyn_cast<IntegerType>(srcVTy.getElementType()) ? intKinds
                                                             : floatKinds;
    auto kind = kinds[UR(kinds.size())];

    SmallVector<int64_t> accShape;
    int64_t reduction_dim = UR(srcVTy.getRank());
    for (int i = 0; i < srcVTy.getRank(); ++i) {
      if (i == reduction_dim) {
        continue;
      }
      accShape.push_back(srcVTy.getDimSize(i));
    }
    auto accTy = VectorType::get(accShape, srcVTy.getElementType());
    auto accCandidates = pool.searchCandidatesFrom(
        {PoolType::Vector}, [&](TypeValue tVal) { return tVal.type == accTy; });
    if (accCandidates.empty()) {
      accCandidates.push_back(pool.generateTypedValue(builder, loc, accTy));
    }
    auto init = sampleTypedValueFrom(accCandidates, "vector.scan");

    bool inclusive = UR(2);
    auto results = vector::ScanOp::create(builder, loc, kind, src.val, init.val,
                                          reduction_dim, inclusive);
    pool.addVector(TypeValue(srcVTy, results.getResult(0)), "vector.scan");
    pool.addTypeValue(TypeValue(accTy, results.getResult(1)), "vector.scan");

    return results.getOperation();
  };
}

OpGenerator vectorScatterGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto pool = region.pool;
    auto baseCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedMemref, PoolType::DynamicShapedMemref});
    if (baseCandidates.empty()) {
      baseCandidates.push_back(region.pool.generateRankedMemref(
          builder, loc, randomRankedMemrefType(builder.getContext())));
    }
    auto base = sampleTypedValueFrom(baseCandidates, "vector.scatter");
    auto indices = region.pool.randomIndicesForShapedType(
        mlir::dyn_cast<ShapedType>(base.type), builder, loc);

    auto indexVTy = random1DVectorType(builder.getIndexType());
    auto indexVecCandidates =
        pool.searchCandidatesFrom({PoolType::Vector}, [&](TypeValue tval) {
          return tval.type == indexVTy;
        });
    if (indexVecCandidates.empty()) {
      indexVecCandidates.push_back(pool.generateVector(builder, loc, indexVTy));
    }
    auto indexVec = sampleTypedValueFrom(indexVecCandidates, "vector.scatter");

    auto maskVTy = VectorType::get(indexVTy.getShape(), builder.getI1Type());
    auto maskCandidates =
        pool.searchCandidatesFrom({PoolType::Vector}, [&](TypeValue tval) {
          return tval.type == maskVTy;
        });
    if (maskCandidates.empty()) {
      maskCandidates.push_back(pool.generateVector(builder, loc, maskVTy));
    }
    auto mask = sampleTypedValueFrom(maskCandidates, "vector.scatter");

    auto vecToStoreTy =
        VectorType::get(indexVTy.getShape(),
                        mlir::dyn_cast<MemRefType>(base.type).getElementType());
    auto vecToStoreCandidates =
        pool.searchCandidatesFrom({PoolType::Vector}, [&](TypeValue tval) {
          return tval.type == vecToStoreTy;
        });
    if (vecToStoreCandidates.empty()) {
      vecToStoreCandidates.push_back(
          pool.generateVector(builder, loc, vecToStoreTy));
    }
    auto vecToStore =
        sampleTypedValueFrom(vecToStoreCandidates, "vector.scatter");

    return vector::ScatterOp::create(builder, loc, /*resultType=*/nullptr,
                                     base.val, ValueRange(indices),
                                     indexVec.val, mask.val, vecToStore.val)
        .getOperation();
  };
}

bool hasSameTrailingShape(VectorType v1, VectorType v2) {
  if (v1.getRank() != v2.getRank()) {
    return false;
  }
  if (v1.getElementType() != v2.getElementType()) {
    return false;
  }
  for (int i = 1; i < v1.getRank(); ++i) {
    if (v1.getDimSize(i) != v2.getDimSize(i)) {
      return false;
    }
  }
  return true;
}

OpGenerator vectorShuffleGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto pool = region.pool;
    auto operand0Candidates = pool.getCandidatesFromVector(
        builder, loc, randomVectorType(builder.getContext()));
    auto operand0 = sampleTypedValueFrom(operand0Candidates, "vector.shuffle");
    auto vTy0 = mlir::dyn_cast<VectorType>(operand0.type);

    auto operand1Candidates =
        pool.searchCandidatesFrom({PoolType::Vector}, [&](TypeValue tVal) {
          return hasSameTrailingShape(mlir::dyn_cast<VectorType>(tVal.type),
                                      vTy0);
        });
    if (operand1Candidates.empty()) {
      operand1Candidates.push_back(pool.generateVector(builder, loc, vTy0));
    }
    auto operand1 = sampleTypedValueFrom(operand1Candidates, "vector.shuffle");
    auto vTy1 = mlir::dyn_cast<VectorType>(operand1.type);

    SmallVector<int64_t> mask;
    if (vTy0.getRank() == 0) {
      mask.push_back(0);
      mask.push_back(1);
    } else {
      for (int i = 0; i < vTy0.getDimSize(0) + vTy1.getDimSize(0); ++i) {
        if (UR(2)) {
          mask.push_back(i);
        }
      }
      // vector types must have positive constant sizes but got 0
      if (mask.empty()) {
        mask.push_back(0);
      }
    }
    SmallVector<int64_t> resShape;
    resShape.push_back(mask.size());
    for (int i = 1; i < vTy1.getRank(); ++i) {
      resShape.push_back(vTy1.getDimSize(i));
    }
    auto resTy = VectorType::get(resShape, vTy1.getElementType());

    auto res = vector::ShuffleOp::create(builder, loc, operand0.val,
                                         operand1.val, mask);
    pool.addVector(TypeValue(resTy, res), "vector.shuffle");
    return res.getOperation();
  };
}

OpGenerator vectorSplatGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto pool = region.pool;
    auto inputCandidates = region.pool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::Index},
        [&](TypeValue) { return true; });
    if (inputCandidates.empty()) {
      inputCandidates.push_back(pool.generateTypedValue(
          builder, loc, randomElementaryOrIndexType(builder.getContext())));
    }
    auto input = sampleTypedValueFrom(inputCandidates, "vector.splat");
    auto vTy = randomVectorType(input.type);

    // ea291d0e8c93d47d7953eff5ca1048891a5fcc55: vector::SplatOp is replaced by
    // vector::BroadcastOp.
    auto res = vector::BroadcastOp::create(builder, loc, vTy, input.val);
    pool.addVector(TypeValue(vTy, res), "vector.splat");
    return res.getOperation();
  };
}

OpGenerator vectorTransposeGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto pool = region.pool;
    auto vecCandidates = region.pool.getCandidatesFromVector(
        builder, loc, randomVectorType(builder.getContext()));
    auto vec = sampleTypedValueFrom(vecCandidates, "vector.transpose");
    auto vTy = mlir::dyn_cast<VectorType>(vec.type);

    SmallVector<int64_t> transp;
    for (int i = 0; i < vTy.getRank(); ++i) {
      transp.insert(transp.begin() + UR(transp.size() + 1), i);
    }
    SmallVector<int64_t> transposedShape;
    for (int i = 0; i < vTy.getRank(); ++i) {
      transposedShape.push_back(vTy.getDimSize(transp[i]));
    }

    return vector::TransposeOp::create(builder, loc, vec.val, transp)
        .getOperation();
  };
}

OpGenerator vectorTransferReadGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto srcCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::StaticShapedMemref,
         PoolType::DynamicShapedMemref, PoolType::DynamicShapedTensor},
        [&](TypeValue tval) {
          auto shapedTy = mlir::dyn_cast<ShapedType>(tval.type);
          return shapedTy.getRank() > 0 &&
                 shapedTy.getElementType().isIntOrFloat();
        });
    if (srcCandidates.empty()) {
      srcCandidates.push_back(region.pool.generateStaticShapedMemref(
          builder, loc, randomStaticShapedMemrefType(builder.getContext())));
    }
    auto src = sampleTypedValueFrom(srcCandidates, "vector.transfer_read");

    auto srcRank = mlir::dyn_cast<ShapedType>(src.type).getRank();
    auto srcElemTy = mlir::dyn_cast<ShapedType>(src.type).getElementType();
    auto indexCandidates = region.pool.getCandidatesFromIndex(builder, loc);
    SmallVector<Value> indices;
    for (int i = 0; i < srcRank; ++i) {
      indices.push_back(
          sampleTypedValueFrom(indexCandidates, "vector.transfer_read").val);
    }
    assert(srcRank > 0);
    auto vectorRank = UR(srcRank);
    SmallVector<int64_t> vectorShape;
    for (int i = 0; i < vectorRank; ++i) {
      vectorShape.push_back(dimPool[UR(dimPool.size())]);
    }
    auto vecTy = VectorType::get(
        vectorShape, mlir::dyn_cast<ShapedType>(src.type).getElementType());

    SmallVector<AffineExpr> results;
    for (int i = 0; i < vectorRank; ++i) {
      if (UR(2)) {
        results.push_back(builder.getAffineDimExpr(i));
      } else {
        results.push_back(builder.getAffineConstantExpr(0));
      }
    }
    auto map = AffineMap::get(srcRank, 0, results, builder.getContext());
    auto permutationMapAttr = AffineMapAttr::get(map);

    auto paddingCandidates = region.pool.searchCandidatesFrom(
        {PoolType::IntOrFloat}, typeEquivalentFilter(srcElemTy));

    if (paddingCandidates.empty()) {
      if (mlir::isa<IntegerType>(srcElemTy)) {
        paddingCandidates.push_back(region.pool.generateInteger(
            builder, loc, mlir::dyn_cast<IntegerType>(srcElemTy)));
      } else {
        paddingCandidates.push_back(region.pool.generateFloat(
            builder, loc, mlir::dyn_cast<FloatType>(srcElemTy)));
      }
    }
    auto padding =
        sampleTypedValueFrom(paddingCandidates, "vector.transfer_read");

    return vector::TransferReadOp::create(builder, loc, vecTy, src.val, indices,
                                          padding.val,
                                          std::optional<ArrayRef<bool>>())
        .getOperation();

    // if (UR(2)) {
    //   return builder
    //       .create<vector::TransferReadOp>(loc, vecTy, src.val, indices,
    //                                       permutationMapAttr, nullptr)
    //       .getOperation();
    // } else {
    //   return builder
    //       .create<vector::TransferReadOp>(loc, vecTy, src.val, indices,
    //                                       padding.val,
    //                                       std::optional<ArrayRef<bool>>())
    //       .getOperation();
    // }
  };
}

OpGenerator vectorTransferWriteGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto destCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::StaticShapedMemref,
         PoolType::DynamicShapedMemref, PoolType::DynamicShapedTensor},
        [&](TypeValue tval) {
          auto shapedTy = mlir::dyn_cast<ShapedType>(tval.type);
          return shapedTy.getRank() > 0 &&
                 shapedTy.getElementType().isIntOrFloat();
        });
    if (destCandidates.empty()) {
      // TODO- new value type rank should be greater than 0
      destCandidates.push_back(region.pool.generateStaticShapedMemref(
          builder, loc, randomStaticShapedMemrefType(builder.getContext())));
    }
    auto dest = sampleTypedValueFrom(destCandidates, "vector.transfer_write");

    auto destRank = mlir::dyn_cast<ShapedType>(dest.type).getRank();
    auto destElemTy = mlir::dyn_cast<ShapedType>(dest.type).getElementType();
    auto indexCandidates = region.pool.getCandidatesFromIndex(builder, loc);
    SmallVector<Value> indices;
    for (int i = 0; i < destRank; ++i) {
      indices.push_back(
          sampleTypedValueFrom(indexCandidates, "vector.transfer_write").val);
    }
    assert(destRank > 0);
    auto vectorRank = UR(destRank);
    SmallVector<int64_t> vectorShape;
    for (int i = 0; i < vectorRank; ++i) {
      vectorShape.push_back(dimPool[UR(dimPool.size())]);
    }
    auto vecTy = VectorType::get(
        vectorShape, mlir::dyn_cast<ShapedType>(dest.type).getElementType());
    auto vectorCandidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector}, typeEquivalentFilter(vecTy));
    if (vectorCandidates.empty()) {
      vectorCandidates.push_back(
          region.pool.generateVector(builder, loc, vecTy));
    }
    auto vector =
        sampleTypedValueFrom(vectorCandidates, "vector.transfer_write");

    SmallVector<AffineExpr> results;
    for (int i = 0; i < vectorRank; ++i) {
      results.push_back(builder.getAffineDimExpr(i));
    }
    auto map = AffineMap::get(destRank, 0, results, builder.getContext());
    auto permutationMapAttr = AffineMapAttr::get(map);

    auto paddingCandidates = region.pool.searchCandidatesFrom(
        {PoolType::IntOrFloat}, typeEquivalentFilter(destElemTy));

    // give all default value false
    llvm::SmallVector<mlir::Attribute> attrs =
        llvm::SmallVector<mlir::Attribute>();
    for (int i = 0; i < vectorShape.size(); i++)
      attrs.push_back(mlir::BoolAttr::get(builder.getContext(), false));

    return vector::TransferWriteOp::create(
               builder, loc, vector.val, dest.val, indices, permutationMapAttr,
               nullptr, mlir::ArrayAttr::get(builder.getContext(), attrs))
        .getOperation();
  };
}

// OpGenerator vectorWarpExecuteOnLane0Op() {
//   return [](OpBuilder &builder, Location loc, OpRegion &parent) {
//     auto laneId = parent.pool.getConstantIndex(builder, loc, 0);
//     int warpSize = 32;

//     auto op = vector::WarpExecuteOnLane0Op::create(builder, loc,
//     TypeRange({}),
//                                                            laneId, warpSize);

//     auto point = builder.saveInsertionPoint();
//     builder.setInsertionPointToEnd(&op.getRegion().front());

//     OpRegion region("vector.warp_execute_on_lane0", parent.depth + 1,
//     parent.cur_child); region.pool.merge(parent.pool); RegionGen gen(&region,
//     {OpNameFilter(opsForVectorWarpExecuteOnLane0)}); gen.apply(builder, loc,
//     8); vector::YieldOp::create(builder, loc);

//     builder.restoreInsertionPoint(point);
//     return op.getOperation();
//   };
// }