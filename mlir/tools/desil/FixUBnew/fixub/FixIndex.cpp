// #include <cmath>
#include <random>

#include "mlir/Dialect/Affine/Passes.h"
// #include "mlir/Dialect/Index/IR/IndexEnums.h.inc"
#include "mlir/InitAllDialects.h"

#include "fixub/common.h"
#include "fixub/ConstantPool.h"
#include "fixub/FixIndex.h"
#include "fixub/FixUBUtils.h"

void FixIndex::fixindex(mlir::Operation *op) {
  std::string opname = op->getName().getStringRef().str();
  if (opname == "index.shl" || opname == "index.shrs" ||
      opname == "index.shru") {
    // If the RHS operand is equal to or greater than the index bitwidth, the
    // result is a poison value.
    FixIndex::fixshlr(op);
  }
  if (opname == "index.ceildivs" || opname == "index.divs" ||
      opname == "index.floordivs") {
    // division by zero and signed division overflow are undefined behaviour.
    FixIndex::fixdivsi(op);
  }
  if (opname == "index.ceildivu" || opname == "index.divu" ||
      opname == "index.remu") {
    // Note: division by zero is undefined behaviour.
    FixIndex::fixdivui(op);
  }
  if (opname == "index.rems") {
    // ??? Maybe same as division by zero
    FixIndex::fixremsi(op);
  }
  return;
}

void FixIndex::fixshlr(mlir::Operation *op) { // viewed
  mlir::Value val1 = op->getOperand(0), val2 = op->getOperand(1);
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);
  mlir::Location loc = op->getLoc();

  auto bitwidthresult = mlir::index::SizeOfOp::create(builder, loc);

  auto comparisonop = mlir::index::CmpOp::create(builder, 
      loc, mlir::index::IndexCmpPredicate::UGE, val2, bitwidthresult);
  auto GreaterBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
    std::random_device rd;
    std::mt19937 gen(rd());
    // this "63" is for 64-bit computation, should be changed to "31" for 32-bit
    // computation
    std::uniform_int_distribution<unsigned> dis(0,
                                                FixIndex::INDEX_BITWIDTH - 1);
    unsigned randomValue = dis(gen);
    auto rdmconstantop =
        ConstantPool::getValue(randomValue, builder.getIndexType());
    mlir::scf::YieldOp::create(b, loc, rdmconstantop);
  };
  auto NotGreaterBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
    mlir::scf::YieldOp::create(b, loc, op->getOperand(1));
  };
  auto ifOp = mlir::scf::IfOp::create(builder, 
      loc, comparisonop->getResult(0), GreaterBuilder, NotGreaterBuilder);
  op->setOperand(1, ifOp.getResult(0));
}

void FixIndex::fixdivsi(mlir::Operation *op) { // viewed
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);
  mlir::Location loc = op->getLoc();

  mlir::Value value = op->getOperand(1);
  // this should be change to 32 in 32-bit system
  auto minConstant = mlir::index::ConstantOp::create(builder, 
      loc, value.getType(),
      mlir::IntegerAttr::get(value.getType(),
                             FixUBUtils::bitsmin(FixIndex::INDEX_BITWIDTH)));
  auto comparison1op = mlir::index::CmpOp::create(builder, 
      loc, mlir::index::IndexCmpPredicate::EQ, op->getOperand(0),
      minConstant.getResult());
  int64_t bitw1 =
      FixIndex::INDEX_BITWIDTH; // this should be change to 32 in 32-bit system

  auto minus1op = mlir::index::ConstantOp::create(builder, 
      loc, value.getType(), mlir::IntegerAttr::get(value.getType(), -1));
  auto comparison2op = mlir::index::CmpOp::create(builder, 
      loc, mlir::index::IndexCmpPredicate::EQ, op->getOperand(1),
      minus1op.getResult());
  auto andop = mlir::arith::AndIOp::create(builder, 
      loc, comparison1op.getResult(), comparison2op.getResult());

  auto EqminBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
    int64_t bitwidth = FixIndex::INDEX_BITWIDTH; // this should be change to 32
                                                 // in 32-bit system
    int64_t rdmnum = 0;
    while (rdmnum == 0 || rdmnum == -1)
      rdmnum = FixUBUtils::randsval(bitwidth);
    auto rdmval = mlir::index::ConstantOp::create(b, 
                       loc, value.getType(),
                       mlir::IntegerAttr::get(value.getType(), rdmnum))
                      .getResult();
    mlir::scf::YieldOp::create(b, loc, rdmval);
  };
  auto NotEqminBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
    auto zeroconstantop = mlir::index::ConstantOp::create(builder, 
        loc, value.getType(), mlir::IntegerAttr::get(value.getType(), 0));
    auto comparison3op = mlir::index::CmpOp::create(builder, 
        loc, mlir::index::IndexCmpPredicate::EQ, op->getOperand(1),
        zeroconstantop.getResult());

    auto Eq0Builder = [&](mlir::OpBuilder &insideb, mlir::Location insideloc) {
      int64_t bitwidth = FixIndex::INDEX_BITWIDTH;
      int64_t rdmnum = 0;
      while (rdmnum == 0) {
        rdmnum = FixUBUtils::randsval(bitwidth);
      }
      auto rdmval = insideb
                        .create<mlir::index::ConstantOp>(
                            insideloc, value.getType(),
                            mlir::IntegerAttr::get(value.getType(), rdmnum))
                        .getResult();
      mlir::scf::YieldOp::create(b, loc, rdmval);
    };
    auto NotEq0Builder = [&](mlir::OpBuilder &insideb,
                             mlir::Location insideloc) {
      mlir::scf::YieldOp::create(b, loc, op->getOperand(1));
    };
    auto insideifOp = mlir::scf::IfOp::create(builder, 
        loc, comparison3op->getResult(0), Eq0Builder, NotEq0Builder);
    mlir::scf::YieldOp::create(b, loc, insideifOp.getResult(0));
  };
  auto ifOp = mlir::scf::IfOp::create(builder, loc, andop->getResult(0),
                                              EqminBuilder, NotEqminBuilder);
  op->setOperand(1, ifOp.getResult(0));
}

void FixIndex::fixdivui(mlir::Operation *op) {
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);
  mlir::Location loc = op->getLoc();

  mlir::Value value = op->getOperand(1);

  auto zeroConstant = mlir::index::ConstantOp::create(builder, 
      loc, value.getType(), mlir::IntegerAttr::get(value.getType(), 0));
  auto comparisonop = mlir::index::CmpOp::create(builder, 
      loc, mlir::index::IndexCmpPredicate::EQ, op->getOperand(1),
      zeroConstant.getResult());

  auto Eq0Builder = [&](mlir::OpBuilder &b, mlir::Location loc) {
    uint64_t bitwidth = FixIndex::INDEX_BITWIDTH;
    uint64_t rdmnum = 0;
    while (rdmnum == 0) {
      rdmnum = FixUBUtils::randuval(bitwidth);
    }
    auto rdmval = mlir::index::ConstantOp::create(b, 
                       loc, value.getType(),
                       mlir::IntegerAttr::get(value.getType(), rdmnum))
                      .getResult();
    mlir::scf::YieldOp::create(b, loc, rdmval);
  };
  auto NotEq0Builder = [&](mlir::OpBuilder &b, mlir::Location loc) {
    mlir::scf::YieldOp::create(b, loc, op->getOperand(1));
  };
  auto ifOp = mlir::scf::IfOp::create(builder, loc, comparisonop->getResult(0),
                                              Eq0Builder, NotEq0Builder);
  op->setOperand(1, ifOp.getResult(0));
}

void FixIndex::fixremsi(mlir::Operation *op) {
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);
  mlir::Location loc = op->getLoc();

  mlir::Value value = op->getOperand(1);
  auto zeroConstant = mlir::index::ConstantOp::create(builder, 
      loc, value.getType(), mlir::IntegerAttr::get(value.getType(), 0));
  auto comparisonop = mlir::index::CmpOp::create(builder, 
      loc, mlir::index::IndexCmpPredicate::EQ, op->getOperand(1),
      zeroConstant.getResult());
  auto Eq0Builder = [&](mlir::OpBuilder &b, mlir::Location loc) {
    int64_t bitwidth = FixIndex::INDEX_BITWIDTH;
    int64_t rdmnum = 0;
    while (rdmnum == 0) {
      rdmnum = FixUBUtils::randsval(bitwidth);
    }
    auto rdmval = mlir::index::ConstantOp::create(b, 
                       loc, value.getType(),
                       mlir::IntegerAttr::get(value.getType(), rdmnum))
                      .getResult();
    mlir::scf::YieldOp::create(b, loc, rdmval);
  };
  auto NotEq0Builder = [&](mlir::OpBuilder &b, mlir::Location loc) {
    mlir::scf::YieldOp::create(b, loc, op->getOperand(1));
  };
  auto ifOp = mlir::scf::IfOp::create(builder, loc, comparisonop->getResult(0),
                                              Eq0Builder, NotEq0Builder);
  op->setOperand(1, ifOp.getResult(0));
}
