#ifndef TRAVERSE_h
#define TRAVERSE_h

#include<fstream>
#include <algorithm>

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/Value.h"

#include "MutationUtil.h"
#include "Debug.h"

// -------------------------------------------------------------------------
// dependencies
struct Dependency {
  mlir::Operation* op;
  std::string type;

  Dependency(mlir::Operation* op, std::string type) :
             op(op),
             type(type) { }

  inline std::string print();
};

inline std::string Dependency::print() {
  // return type + " " + getOperationName(op);
  return type + " " + getOperationNameParamType(op);
}

std::string printLayer(std::vector<Dependency> dependency) {
  std::string res = "";
  unsigned idx = 0;
  while (idx < dependency.size()) {
    res = res + trim(dependency[idx].print()) + " ";
    ++idx;
  }
  return trim(res);
}

inline bool compareDependency(Dependency a, Dependency b) {
  return comparePosition(a.op, b.op) > 0;
}

mlir::Operation *getDifiningOpIncludeArgs(mlir::Value val) {
  /**
   * This function is used to find the defining operation of a value.
   * Including the situation that this value is the block argument.
   *
   * Params:
   * mlir::OpBuilder builder, to support MLIR C++ API
   * mlir::Location loc, to support MLIR C++ API
   * mlir::Value val, the value which we need to find the defining op.
   *
   * Return:
   * The pointer of the defining operation of val.
   */
  if (val.getDefiningOp()) {
    return val.getDefiningOp();
  } else {
    mlir::BlockArgument arg = mlir::dyn_cast<mlir::BlockArgument>(val);
    assert(arg);
    mlir::Block *block = arg.getOwner();
    mlir::Operation *parentOp = block->getParentOp();
    if (block->getNumArguments() != parentOp->getNumOperands())
      return NULL;
    unsigned idx = 0;
    if (parentOp->getNumOperands() == 0) {
      return NULL;
    }
    while (idx < block->getNumArguments()) {
      if (block->getArgument(idx) == val)
        break;
      idx++;
    }
    return getDifiningOpIncludeArgs(parentOp->getOperand(idx));
  }
}

void getDependency(Operation* o, std::vector<Dependency>* dependencies) {
  mlir::Operation* parent = o->getParentOp();
  if (parent) {
    dependencies->push_back(Dependency(parent, "control"));
  }
  for (int oIdx = 0; oIdx < o->getNumOperands(); oIdx++) {
    mlir::Value operand = o->getOperand(oIdx);
    // mlir::Operation* operandOperation = operand.getDefiningOp();
    mlir::Operation* operandOperation = getDifiningOpIncludeArgs(operand);
    if (operandOperation) {
      dependencies->push_back(Dependency(operandOperation, "data"));
    }
  }
  sort(dependencies->begin(), dependencies->end(), compareDependency);
}

inline std::string getCovId(mlir::Operation* o) {
  std::string name = getOperationName(o);
  std::string loc = getLocationStr(o);
  return name + "-" + loc;
}
// -------------------------------------------------------------------------
// parser/mutator
struct MutationParser {
  unsigned depth_limit;
  mlir::Operation* inter;
  std::ofstream covFile;
  // std::map<operation_cov_id, std::vector<std::vector<Dependency>>>
  std::map<std::string, std::vector<std::vector<Dependency>>> covTable;

  MutationParser(unsigned depth_limit, std::string covFilePath, mlir::Operation* inter) : 
                 depth_limit(depth_limit),
                 inter(inter) {

    covFile.open(covFilePath, std::ios::app);
    covTable = std::map<std::string, std::vector<std::vector<Dependency>>>();
    initCov(inter);
    getCov();
  }

  void initCov(Operation* op);
  void getCov();
  void writeCoverage();
};

void MutationParser::initCov(Operation* op) {
  // Traverse Logic
  for (Region &region : op->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (Operation &op : block.getOperations()) {
        initCov(&op);
      }
    }
  }
  
  // get coverage.
  std::vector<Dependency> dependency = std::vector<Dependency>();
  getDependency(op, &dependency);
  covTable[getCovId(op)] = std::vector<std::vector<Dependency>>();
  covTable[getCovId(op)].push_back({Dependency(op, "")});
  covTable[getCovId(op)].push_back(dependency);
}

void MutationParser::getCov() {
  unsigned depth = 2;
  while (depth <= depth_limit) {
    for (auto it = covTable.begin(); it != covTable.end(); ++it) {
      auto key = it->first;
      std::vector<Dependency> dependencies = covTable[key][depth-1];
      std::vector<Dependency> nextLayer = std::vector<Dependency>();
      for (Dependency d : dependencies) {
        nextLayer.insert(nextLayer.end(), covTable[getCovId(d.op)][1].begin(), covTable[getCovId(d.op)][1].end());
      }
      covTable[key].push_back(nextLayer);
    }
    ++depth;
  }
}

// void MutationParser::writeCoverage_old() {
//   unsigned depth = 0;
//   // For each depth
//   while (depth <= depth_limit) {
//     // For each operation
//     for (auto it = covTable.begin(); it != covTable.end(); ++it) {
//       auto key = it->first;
//       std::string curCov = "";
//       // For current operation, we should append all of the table content together.
//       unsigned curDepth = 0;
//       while (curDepth <= depth && curDepth < covTable[key].size()) {
//         curCov = trim(curCov + printLayer(covTable[key][curDepth])) + " ";
//         ++curDepth;
//       }
//       curCov = trim(curCov);
//       covFile << std::to_string(depth) << ": " << curCov << std::endl;
//     }
//     ++depth;
//   }
//   covFile.flush();
//   covFile.close();
// }

void MutationParser::writeCoverage() {
  for (auto it = covTable.begin(); it != covTable.end(); ++it) {
    auto key = it->first;
    std::string curCov = "";
    // For current operation, we should append all of the table content together.
    unsigned curDepth = 0;
    while (curDepth <= depth_limit && curDepth < covTable[key].size()) {
      curCov = trim(curCov + printLayer(covTable[key][curDepth])) + " ";
      ++curDepth;
    }
    curCov = trim(curCov);
    covFile << curCov << std::endl;
  }
  covFile.flush();
  covFile.close();
}


#endif // TRAVERSE_h