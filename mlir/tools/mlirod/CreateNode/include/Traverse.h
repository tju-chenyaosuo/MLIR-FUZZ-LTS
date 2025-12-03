#ifndef TRAVERSE_h
#define TRAVERSE_h

#include "NodeGenerator.h"

// -------------------------------------------------------------------------
// parser/mutator
struct MutationParser {
  mlir::Operation *inter;
  unsigned mutationCnt;
  unsigned mutPos;
  unsigned curPos;

  MutationParser(mlir::Operation *inter, unsigned mutPos)
      : inter(inter), mutPos(mutPos) {
    mutationCnt = 0;
    curPos = 0;
  }

  void printOperation(Operation *op, MutationBlock *mb);
  // void getAllDataDependency(MutationBlock* op, std::vector<mlir::Operation*>*
  // dataDependencies);
};

void MutationParser::printOperation(Operation *op, MutationBlock *mb) {
  // // Mutator
  // // 1. randomly determine to mutates
  unsigned oldMutation = mutationCnt;
  if (curPos >= mutPos) {
    // 2. randomly choose a mutation

#ifdef DEBUG
    llvm::outs() << "[MutationParser::printOperation] Mutation start!\n";
#endif

    mlir::OpBuilder builder(op);
    mlir::Location loc = op->getLoc();
    unsigned idx = rollIdx(generators.size());

    // std::string target_mutator = requiredOp;
    // idx = 0;
    // while (idx < generatorsStr.size()) {
    //   if (generatorsStr[idx] == target_mutator) { break; }
    //   ++idx;
    // }

    bool validMutation = true;
    validMutation =
        validMutation && ("vectorMaskGenerator" == generatorsStr[idx] &&
                          getOperationName(op) != "builtin.module" &&
                          getOperationName(op) != "vector.mask" &&
                          getOperationName(op) != "vector.multi_reduction" &&
                          getOperationName(op) != "vector.yield");
#ifdef DEBUG
    llvm::outs() << "Current operation: " << getOperationName(op) << "\n";
#endif

    if (validMutation) {
#ifdef DEBUG
      llvm::outs() << "[MutationParser::printOperation] Selected mutator: "
                   << generatorsStr[idx] << "\n";
#endif
      OpGenerator mutator = generators[idx];
      if (mutator(builder, loc, mb)) {
        ++mutationCnt;
#ifdef DEBUG
        llvm::outs() << "[MutationParser::printOperation] Mutation success!\n";
#endif
      }
    }
  }

  for (Region &region : op->getRegions()) {
    if (oldMutation != mutationCnt) {
      break;
    }
    for (Block &block : region.getBlocks()) {
      if (oldMutation != mutationCnt) {
        break;
      }
      MutationBlock *childMb = new MutationBlock(*mb);
      for (Operation &op : block.getOperations()) {
        if (oldMutation != mutationCnt) {
          break;
        }
        printOperation(&op, childMb);
      }
      delete childMb;
    }
  }
  // Then add the return value of current operation into variable pool
  // Use this strategy to deal with some opreation that create the region.
  for (int rIdx = 0; rIdx < op->getNumResults(); rIdx++) {
    Value res = op->getOpResult(rIdx);
    mb->add2Pool(res);
  }
  ++curPos;
}

#endif // TRAVERSE_h