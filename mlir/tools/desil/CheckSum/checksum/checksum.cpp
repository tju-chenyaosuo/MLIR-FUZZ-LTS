#include "checksum/checksum.h"

mlir::Operation *CheckSum::idx0op;
mlir::Operation *CheckSum::idx1op;

inline std::string CheckSum::getValueTypeStr(mlir::Value v) {
  mlir::Type vTy = v.getType();

  llvm::SmallVector<char, 128> buffer;
  llvm::raw_svector_ostream stream(buffer);
  vTy.print(stream);

  std::string ty = stream.str().str();
  return ty;
}

inline std::string CheckSum::getTypeStr(mlir::Type type) {

  llvm::SmallVector<char, 128> buffer;
  llvm::raw_svector_ostream stream(buffer);
  type.print(stream);

  std::string ty = stream.str().str();
  return ty;
}

void CheckSum::addValue(mlir::Value val) {
  std::string optype = getValueTypeStr(val);

  // mlir::IntegerType type_16 = mlir::IntegerType::get(val.getContext(), 16);
  // mlir::Type aa = type_16;
  // mlir::IntegerType bb = aa.dyn_cast<mlir::IntegerType>();

  std::string numtype;

  if (optype.find("index") < optype.length()) {
    numtype = "index";
  } else if (optype.find("i1>") < optype.length() || optype == "i1") {
    numtype = "i1";
  } else if (optype.find("i16") < optype.length()) {
    numtype = "i16";
  } else if (optype.find("i32") < optype.length()) {
    numtype = "i32";
  } else if (optype.find("i64") < optype.length()) {
    numtype = "i64";
  }

  value_type valtype;
  // after affine, vector and memref are included, the push back line can be
  // moved out of the if field
  if (optype.length() < 6) {
    valtype = SCALAR;
    checksumValuePool[numtype][valtype].push_back(val); // to delete
  } else if (optype.find("tensor") < optype.length()) {
    valtype = TENSOR;
    checksumValuePool[numtype][valtype].push_back(val); // to delete
    // llvm::outs()<< val.getType()<< "\n";
  } else if (optype.find("vector") < optype.length()) {
    valtype = VECTOR;
    checksumValuePool[numtype][valtype].push_back(val); // to delete
    // llvm::outs()<< val.getType()<< "\n";
  } else if (optype.find("memref") < optype.length()) {
    valtype = MEMREF;
    // llvm::outs()<<"checksum for memref add pool\n";
    checksumValuePool[numtype][valtype].push_back(val); // to delete
    // llvm::outs()<< val.getType()<< "\n";
  }

  // checksumValuePool[numtype][valtype].push_back(val); // to use
  return;
}

mlir::Value CheckSum::calCheckSumAll(mlir::Location loc,
                                     mlir::OpBuilder &builder) {

  mlir::IntegerType type_i64 = mlir::IntegerType::get(builder.getContext(), 64);
  auto zeroconstantop = mlir::arith::ConstantOp::create(
      builder, loc, type_i64, mlir::IntegerAttr::get(type_i64, 0));

  CheckSum::idx0op = mlir::index::ConstantOp::create(
      builder, loc, builder.getIndexType(),
      mlir::IntegerAttr::get(builder.getIndexType(), 0));

  CheckSum::idx1op = mlir::index::ConstantOp::create(
      builder, loc, builder.getIndexType(),
      mlir::IntegerAttr::get(builder.getIndexType(), 1));

  builder.setInsertionPointAfter(idx1op);
  mlir::Value lstval = zeroconstantop.getResult();
  for (auto typemap : checksumValuePool) {
    std::string stringtype = typemap.first;
    mlir::Type type;
    if (stringtype == "")
      continue;
    if (stringtype == "index")
      type = mlir::IndexType::get(builder.getContext());
    else if (stringtype == "i1")
      type = mlir::IntegerType::get(builder.getContext(), 1);
    else if (stringtype == "i16")
      type = mlir::IntegerType::get(builder.getContext(), 16);
    else if (stringtype == "i32")
      type = mlir::IntegerType::get(builder.getContext(), 32);
    else if (stringtype == "i64")
      type = mlir::IntegerType::get(builder.getContext(), 64);
    mlir::Value nowval = calCheckSumByType(loc, builder, type);
    lstval = mlir::arith::AddIOp::create(builder, loc, lstval, nowval);
  }

  mlir::vector::PrintOp::create(builder, loc, lstval);
  auto tmpval = mlir::arith::ConstantOp::create(
      builder, loc, builder.getIntegerType(64), builder.getI64IntegerAttr(1));
  mlir::vector::PrintOp::create(builder, loc, tmpval);
  lstval = mlir::arith::AddIOp::create(builder, loc, lstval, tmpval);
  mlir::vector::PrintOp::create(builder, loc, lstval);
  return lstval;
}

mlir::Value CheckSum::calCheckSumByType(mlir::Location loc,
                                        mlir::OpBuilder &builder,
                                        mlir::Type type) {

  mlir::Value zeroval = builder
                            .create<mlir::arith::ConstantOp>(
                                loc, type, mlir::IntegerAttr::get(type, 0))
                            .getResult();

  std::string typestr = getTypeStr(type);

  mlir::Value lstval = zeroval;

  // to prevent the crash in FuncBufferizableOpInterfaceImpl.cpp:0:0, I removed this line.
  // lstval = calTensorCheckSumByType(loc, builder, type, typestr, zeroval);

  lstval = calVectorCheckSumByType(loc, builder, type, typestr, lstval);

  lstval = calMemrefCheckSumByType(loc, builder, type, typestr, lstval);

  // calScalarCheckSumByType
  builder.setInsertionPointAfter(lstval.getDefiningOp());
  lstval = zeroval;
  for (auto val : checksumValuePool[typestr][SCALAR]) {
    lstval = mlir::arith::AddIOp::create(builder, loc, lstval, val);
  }

  // type cast, and return
  mlir::Value retval;
  mlir::IntegerType type_i64 = mlir::IntegerType::get(lstval.getContext(), 64);
  if (mlir::isa<mlir::IndexType>(type)) {
    retval = mlir::arith::IndexCastOp::create(builder, loc, type_i64, lstval)
                 .getResult();
  } else if (mlir::isa<mlir::IntegerType>(type)) {
    if (mlir::dyn_cast<mlir::IntegerType>(type) != type_i64)
      retval = mlir::arith::ExtUIOp::create(builder, loc, type_i64, lstval)
                   .getResult();
    else
      retval = lstval;
  }
  return retval;
}

mlir::Value CheckSum::calTensorCheckSumByType(mlir::Location loc,
                                              mlir::OpBuilder &builder,
                                              mlir::Type type,
                                              std::string typestr,
                                              mlir::Value lstval) {

  if (checksumValuePool[typestr][TENSOR].size() > 0) {
    mlir::func::FuncOp tensorFuncOp = this->calCheckSumTensor(type);
    funcOpPool[TENSOR][typestr] = tensorFuncOp;
    builder.setInsertionPointAfter(lstval.getDefiningOp());
    for (auto val : checksumValuePool[typestr][TENSOR]) {

      mlir::Value tocasstval;
      if (mlir::dyn_cast<mlir::RankedTensorType>(val.getType()).getRank() > 1) {
        llvm::SmallVector<mlir::ReassociationIndices> reindices_vector;
        mlir::ReassociationIndices reindices;
        // llvm::outs()<<val.getType()<<"\n";
        for (int i = 0;
             i <
             mlir::dyn_cast<mlir::RankedTensorType>(val.getType()).getRank();
             i++) {
          reindices.push_back(i);
        }
        reindices_vector.push_back(reindices);
        auto collapseshapeop = mlir::tensor::CollapseShapeOp::create(
            builder, loc, val, reindices_vector);
        tocasstval = collapseshapeop.getResult();
      } else
        tocasstval = val;
      mlir::RankedTensorType tensorType = mlir::RankedTensorType::get(
          /*shape=*/{mlir::ShapedType::kDynamic}, type);

      mlir::Value tochecksumval;
      // llvm::outs() <<
      // tocasstval.getType().dyn_cast<mlir::MemRefType>().getDimSize(0) <<"\n";
      if (mlir::dyn_cast<mlir::RankedTensorType>(tocasstval.getType())
              .getDimSize(0) > 0)
        tochecksumval =
            mlir::tensor::CastOp::create(builder, loc, tensorType, tocasstval)
                ->getResult(0);
      else
        tochecksumval = tocasstval;
      // auto castOp =
      //     mlir::tensor::CastOp::create(builder, loc, tensorType, tocasstval);
      llvm::SmallVector<mlir::Value> arguments;
      arguments.push_back(tochecksumval);
      auto funcCallOp = mlir::func::CallOp::create(builder, loc, tensorFuncOp,
                                                   mlir::ValueRange(arguments));

      lstval = funcCallOp.getResult(0);
      addValue(funcCallOp.getResult(0));
    }
  }
  return lstval;
}

mlir::Value CheckSum::calMemrefCheckSumByType(mlir::Location loc,
                                              mlir::OpBuilder &builder,
                                              mlir::Type type,
                                              std::string typestr,
                                              mlir::Value lstval) {

  if (checksumValuePool[typestr][MEMREF].size() > 0) {
    // llvm::outs()<<typestr <<" "<<"memref value exists\n";
    mlir::func::FuncOp memrefFuncOp = this->calCheckSumMemref(type);
    funcOpPool[MEMREF][typestr] = memrefFuncOp;
    builder.setInsertionPointAfter(lstval.getDefiningOp());
    for (auto val : checksumValuePool[typestr][MEMREF]) {
      // auto val = lstval;
      mlir::Value tocasstval;
      if (mlir::dyn_cast<mlir::MemRefType>(val.getType()).getRank() > 1) {
        llvm::SmallVector<mlir::ReassociationIndices> reindices_vector;
        mlir::ReassociationIndices reindices;
        // llvm::outs()<<val.getType()<<"\n";
        for (int i = 0;
             i < mlir::dyn_cast<mlir::MemRefType>(val.getType()).getRank();
             i++) {
          reindices.push_back(i);
        }
        reindices_vector.push_back(reindices);
        auto collapseshapeop = mlir::memref::CollapseShapeOp::create(
            builder, loc, val, reindices_vector);
        tocasstval = collapseshapeop.getResult();
      } else
        tocasstval = val;
      mlir::MemRefType tensorType = mlir::MemRefType::get(
          /*shape=*/{mlir::ShapedType::kDynamic}, type);

      mlir::Value tochecksumval;
      // llvm::outs() <<
      // tocasstval.getType().dyn_cast<mlir::MemRefType>().getDimSize(0) <<"\n";
      if (mlir::dyn_cast<mlir::MemRefType>(tocasstval.getType()).getDimSize(0) >
          0)
        tochecksumval =
            mlir::memref::CastOp::create(builder, loc, tensorType, tocasstval)
                ->getResult(0);
      else
        tochecksumval = tocasstval;
      llvm::SmallVector<mlir::Value> arguments;
      arguments.push_back(tochecksumval);
      auto funcCallOp = mlir::func::CallOp::create(builder, loc, memrefFuncOp,
                                                   mlir::ValueRange(arguments));

      lstval = funcCallOp.getResult(0);
      addValue(funcCallOp.getResult(0));
    }
  }
  return lstval;
}

mlir::Value CheckSum::calVectorCheckSumByType(mlir::Location loc,
                                              mlir::OpBuilder &builder,
                                              mlir::Type type,
                                              std::string typestr,
                                              mlir::Value lstval) {

  builder.setInsertionPointAfter(lstval.getDefiningOp());
  mlir::Value originLstval = lstval;
  for (auto val : checksumValuePool[typestr][VECTOR]) {

    auto vectorTy = mlir::dyn_cast<mlir::VectorType>(val.getType());
    // auto size = vectorTy.getDimSize(0);
    // llvm::outs()<<vectorTy.getRank()<<"\n";

    int newDim = 1;
    mlir::Value toSumVectorValue;

    if (vectorTy.getRank() == 0)
      return lstval;
    else if (vectorTy.getRank() == 1) {
      newDim = vectorTy.getDimSize(0);
      toSumVectorValue = val;
    } else {
      for (int i = 0; i < vectorTy.getRank(); i++) {
        newDim *= vectorTy.getDimSize(i);
      }
      mlir::VectorType vectorType = mlir::VectorType::get(
          /*shape=*/{newDim}, type);
      auto castOp =
          mlir::vector::ShapeCastOp::create(builder, loc, vectorType, val);
      toSumVectorValue = castOp.getResult();
    }
    auto idxdimop = mlir::index::ConstantOp::create(
        builder, loc, builder.getIndexType(),
        mlir::IntegerAttr::get(builder.getIndexType(), newDim));

    auto blockBuilder = [&](mlir::OpBuilder &b, mlir::Location loc,
                            mlir::Value iv /*loop iterator*/,
                            mlir::ValueRange args) {
      auto extractop =
          mlir::vector::ExtractOp::create(b, loc, toSumVectorValue, iv);
      auto addop =
          mlir::arith::AddIOp::create(b, loc, args[0], extractop.getResult());

      mlir::scf::YieldOp::create(b, loc, addop.getResult());
    };
    auto forop = mlir::scf::ForOp::create(
        builder, loc, CheckSum::idx0op->getResult(0), idxdimop.getResult(),
        CheckSum::idx1op->getResult(0), llvm::ArrayRef(lstval), blockBuilder);
    // auto addop =
    //         mlir::arith::AddIOp::create(builder, loc, lstval,
    //         forop.getResult(0));
    lstval = forop.getResult(0);
    // addValue(forop.getResult(0));
  }
  if (lstval != originLstval)
    addValue(lstval);

  return lstval;
}

mlir::func::FuncOp CheckSum::calCheckSumTensor(mlir::Type type) {

  mlir::Operation *op = CheckSum::mainfunc;

  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);
  mlir::RankedTensorType tensorType = mlir::RankedTensorType::get(
      /*shape=*/{mlir::ShapedType::kDynamic}, type);
  llvm::SmallVector<mlir::Type> argTypes;
  argTypes.push_back(tensorType);
  llvm::SmallVector<mlir::Type> retTypes;
  retTypes.push_back(type);
  auto funcType = mlir::FunctionType::get(
      op->getContext(), mlir::TypeRange(argTypes), mlir::TypeRange(retTypes));

  std::string funcname = "tensor_" + CheckSum::getTypeStr(type);
  auto newFuncOp =
      mlir::func::FuncOp::create(builder, op->getLoc(), funcname, funcType);
  newFuncOp.addEntryBlock();
  builder.setInsertionPointToEnd(&newFuncOp.getBody().front());

  auto idx0op = mlir::index::ConstantOp::create(
      builder, op->getLoc(), builder.getIndexType(),
      mlir::IntegerAttr::get(builder.getIndexType(), 0));

  auto idx1op = mlir::index::ConstantOp::create(
      builder, op->getLoc(), builder.getIndexType(),
      mlir::IntegerAttr::get(builder.getIndexType(), 1));

  auto zeroconstantop = mlir::arith::ConstantOp::create(
      builder, op->getLoc(), type, mlir::IntegerAttr::get(type, 0));

  auto dimop = mlir::tensor::DimOp::create(
      builder, op->getLoc(), newFuncOp.getBody().getArguments()[0],
      idx0op.getResult());

  auto blockBuilder = [&](mlir::OpBuilder &b, mlir::Location loc,
                          mlir::Value iv /*loop iterator*/,
                          mlir::ValueRange args) {
    auto extractop = mlir::tensor::ExtractOp::create(
        b, loc, newFuncOp.getBody().getArguments()[0], iv);
    auto addop =
        mlir::arith::AddIOp::create(b, loc, args[0], extractop.getResult());

    mlir::scf::YieldOp::create(b, op->getLoc(), addop.getResult());
  };
  auto forop = mlir::scf::ForOp::create(
      builder, op->getLoc(), idx0op.getResult(), dimop.getResult(),
      idx1op.getResult(), llvm::ArrayRef(zeroconstantop.getResult()),
      blockBuilder);

  mlir::func::ReturnOp::create(builder, op->getLoc(), forop.getResult(0));
  return newFuncOp;
}

mlir::func::FuncOp CheckSum::calCheckSumMemref(mlir::Type type) {

  mlir::Operation *op = CheckSum::mainfunc;

  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);
  mlir::MemRefType memrefType = mlir::MemRefType::get(
      /*shape=*/{mlir::ShapedType::kDynamic}, type);
  llvm::SmallVector<mlir::Type> argTypes;
  argTypes.push_back(memrefType);
  llvm::SmallVector<mlir::Type> retTypes;
  retTypes.push_back(type);
  auto funcType = mlir::FunctionType::get(
      op->getContext(), mlir::TypeRange(argTypes), mlir::TypeRange(retTypes));

  std::string funcname = "memref_" + CheckSum::getTypeStr(type);
  auto newFuncOp =
      mlir::func::FuncOp::create(builder, op->getLoc(), funcname, funcType);
  newFuncOp.addEntryBlock();
  builder.setInsertionPointToEnd(&newFuncOp.getBody().front());

  auto idx0op = mlir::index::ConstantOp::create(
      builder, op->getLoc(), builder.getIndexType(),
      mlir::IntegerAttr::get(builder.getIndexType(), 0));

  auto idx1op = mlir::index::ConstantOp::create(
      builder, op->getLoc(), builder.getIndexType(),
      mlir::IntegerAttr::get(builder.getIndexType(), 1));

  auto zeroconstantop = mlir::arith::ConstantOp::create(
      builder, op->getLoc(), type, mlir::IntegerAttr::get(type, 0));

  auto dimop = mlir::memref::DimOp::create(
      builder, op->getLoc(), newFuncOp.getBody().getArguments()[0],
      idx0op.getResult());

  auto blockBuilder = [&](mlir::OpBuilder &b, mlir::Location loc,
                          mlir::Value iv /*loop iterator*/,
                          mlir::ValueRange args) {
    auto loadop = mlir::memref::LoadOp::create(
        b, loc, newFuncOp.getBody().getArguments()[0], iv);
    auto addop =
        mlir::arith::AddIOp::create(b, loc, args[0], loadop.getResult());

    mlir::scf::YieldOp::create(b, op->getLoc(), addop.getResult());
  };
  auto forop = mlir::scf::ForOp::create(
      builder, op->getLoc(), idx0op.getResult(), dimop.getResult(),
      idx1op.getResult(), llvm::ArrayRef(zeroconstantop.getResult()),
      blockBuilder);

  mlir::func::ReturnOp::create(builder, op->getLoc(), forop.getResult(0));
  return newFuncOp;
}