#ifndef TRAVERSE_h
#define TRAVERSE_h

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/Value.h"

#include "Debug.h"
#include "MutationUtil.h"

// -------------------------------------------------------------------------
// To record the variables in the region.
struct MutationValuePool {
  std::map<std::string, std::vector<mlir::Value>> pool;

  MutationValuePool(std::map<std::string, std::vector<mlir::Value>> pool =
                        std::map<std::string, std::vector<mlir::Value>>())
      : pool(pool) {}

  void addValue(mlir::Value &v);
  void merge(MutationValuePool mvp);
  void clear();
#ifdef DEBUG
  void print();
#endif
  mlir::Value search(std::string ty, std::string operationName);
};

// Add a value to MutationValuePool
void MutationValuePool::addValue(mlir::Value &v) {
  std::string vTy = getValueTypeStr(v);
  std::vector<mlir::Value> typedValuePool;
  if (!pool.count(vTy))
    pool[vTy] = std::vector<mlir::Value>();
  pool[vTy].push_back(v);
  return;
}

// Merge the existing MutationValuePool into this pool.
void MutationValuePool::merge(MutationValuePool mvp) {
#ifdef DEBUG
  llvm::outs() << "[MutationValuePool::merge] mvp info:";
  mvp.print();
#endif

  for (auto it = mvp.pool.begin(); it != mvp.pool.end(); ++it) {
    std::string key = it->first;
    std::vector<mlir::Value> value = it->second;

    if (!pool.count(key)) {
      pool.insert(std::make_pair(key, std::vector<mlir::Value>()));
    }
    pool[key].insert(pool[key].end(), mvp.pool[key].begin(),
                     mvp.pool[key].end());
  }
  return;
}

mlir::Value MutationValuePool::search(std::string ty,
                                      std::string operationName) {
  std::vector<int> candidate = std::vector<int>();
  for (int idx = 0; idx < pool[ty].size(); ++idx) {
    mlir::Value var = pool[ty][idx];

    if (getOperationName(var.getDefiningOp()) != operationName)
      candidate.push_back(idx);
  }
  int idx = 0;
  if (candidate.size()) {
    idx = candidate[rollIdx(candidate.size())];
  } else {
    idx = rollIdx(pool[ty].size());
  }
  return pool[ty][idx];
}

// -------------------------------------------------------------------------
// A warpper of region.
struct MutationBlock {
  MutationValuePool valuePool;

  MutationBlock() { valuePool = MutationValuePool(); }

  MutationBlock(const MutationBlock &block) {
    valuePool = MutationValuePool();
    valuePool.merge(block.valuePool);
  }

  void add2Pool(mlir::Value &v);
  mlir::Value search(std::string ty, std::string operationName);
};

void MutationBlock::add2Pool(mlir::Value &v) { valuePool.addValue(v); }

mlir::Value MutationBlock::search(std::string ty, std::string operationName) {
  return valuePool.search(ty, operationName);
}

// -------------------------------------------------------------------------
// parser/mutator
struct MutationParser {
  mlir::Operation *inter;
  unsigned mutationCnt;
  unsigned mutPos;
  unsigned curPos;

  MutationParser(mlir::Operation *inter, unsigned mutPos)
      : inter(inter), mutPos(mutPos) {
#ifdef DEBUG
    llvm::outs()
        << "[MutationParser] initializer: the mutation probability is: \n"
        << "Operation mutation probability is " << std::to_string(operationProb)
        << "\n";
#endif
    mutationCnt = 0;
    curPos = 0;
  }

  void printOperation(Operation *op, MutationBlock *mb);
  void getAllDataDependency(Operation *op,
                            std::vector<mlir::Operation *> *dataDependencies);
  void findRange(Operation *cur, Operation *start, Operation *end,
                 unsigned depth, std::vector<mlir::Operation *> *range);
  mlir::Operation *getDifiningOpIncludeArgs(mlir::Value val);
  void checkAllDataDependency(Operation *op, bool *dependentOnBlockArgs);
  bool dependentBlockArgs(mlir::Value val);
};

void MutationParser::findRange(Operation *cur, Operation *start, Operation *end,
                               unsigned depth,
                               std::vector<mlir::Operation *> *range) {
  // TODO:
  if (comparePosition(cur, start) > 0 && comparePosition(cur, end) <= 0 &&
      getControlDepth(cur) < depth)
    range->push_back(cur);

  // Traverse logic
  for (Region &region : cur->getRegions()) {
#ifdef DEBUG
    llvm::outs() << "[MutationParser::findRange] deal with region:\n";
    llvm::outs() << "Region with " << region.getBlocks().size() << " blocks:\n";
#endif
    for (Block &block : region.getBlocks()) {
#ifdef DEBUG
      llvm::outs() << "[MutationParser::findRange] deal with block:\n";
#endif
      for (Operation &op : block.getOperations()) {
        findRange(&op, start, end, depth, range);
      }
    }
  }
}

void MutationParser::printOperation(Operation *op, MutationBlock *mb) {
  mlir::Operation *parent = op->getParentOp();
  std::string parentName = getOperationName(parent);

  // Mutator
  std::string opName = getOperationName(op);
  bool operationDice = curPos >= mutPos;
  bool validParent = parent && parentName != "func.func" &&
                     parentName != "builtin.module" &&
                     parent->getBlockOperands().size() == 0;

  // getNumArguments ()
  bool mutate = validParent && operationDice && op->getNumRegions() == 0 &&
                opName.find("yield") == std::string::npos &&
                opName.find("return") == std::string::npos &&
                opName.find("scf.condition") == std::string::npos;

  std::vector<std::string> invalidOpts = {
      // "arith.addi", "arith.addf", "arith.muli", "arith.mulf",
      // "arith.divf", "arith.divi", "arith.shli", "arith.shlf"
  };
  for (auto s : invalidOpts) {
    if (s == opName) {
      mutate = false;
      break;
    }
  }

  bool dependentOnBlockArgs = false;
  checkAllDataDependency(op, &dependentOnBlockArgs);
  mutate = mutate && !dependentOnBlockArgs;

  if (mutate) {
    std::string grandParentName = getOperationName(parent->getParentOp());

    std::vector<mlir::Operation *> dataDependencies =
        std::vector<mlir::Operation *>();
    getAllDataDependency(op, &dataDependencies);

    // calculate move operations
    std::vector<mlir::Operation *> ops2move = std::vector<mlir::Operation *>();
    for (int ddIdx = 0; ddIdx < dataDependencies.size(); ++ddIdx) {
      std::string tmpParentName =
          getOperationName(dataDependencies[ddIdx]->getParentOp());
      if (tmpParentName == parentName)
        ops2move.push_back(dataDependencies[ddIdx]);
    }

    // calculate insertion range
    std::vector<mlir::Operation *> startCandidates =
        std::vector<mlir::Operation *>();
    for (int ddIdx = 0; ddIdx < dataDependencies.size(); ++ddIdx) {
      std::string tmpParentName =
          getOperationName(dataDependencies[ddIdx]->getParentOp());
      if (tmpParentName == grandParentName)
        startCandidates.push_back(dataDependencies[ddIdx]);
    }

    // TODO: if there is no start candidates.
    if (!startCandidates.size()) {
      mlir::Block *block = parent->getBlock();
      mlir::Operation *firstOp = &block->front();
      startCandidates.push_back(firstOp);
    }

    Operation *start = startCandidates[0];
    for (int ddIdx = 0; ddIdx < startCandidates.size(); ++ddIdx) {
      if (comparePosition(start, startCandidates[ddIdx]) < 0)
        start = startCandidates[ddIdx];
    }
    // To get the insertion range and sample the insertion point.
    unsigned oriDepth = getControlDepth(op);
    std::vector<mlir::Operation *> range = std::vector<mlir::Operation *>();
    findRange(inter, start, parent, oriDepth, &range);

#ifdef DEBUG
    llvm::outs() << "Size of range is: " << std::to_string(range.size())
                 << "\n";
#endif

    if (range.size()) {
      auto selectedPos = rollIdx(range.size());

#ifdef DEBUG
      llvm::outs() << "Selected size is: " << selectedPos << "\n";
#endif

      Operation *insertionPoint = range[selectedPos];
      // insert all of the data dependency within same parents(of target
      // operation) into the selected insertion point.
      for (int ddIdx = 0; ddIdx < ops2move.size(); ++ddIdx) {
        ops2move[ddIdx]->moveBefore(insertionPoint);
      }
      ++mutationCnt;
      return;
    }
  }

  // Traverse logic
  unsigned oldMutation = mutationCnt;

#ifdef DEBUG
  llvm::outs() << " " << op->getNumRegions() << " nested regions:\n";
#endif
  for (Region &region : op->getRegions()) {
#ifdef DEBUG
    llvm::outs() << "[MutationParser::printOperation] deal with region:\n";
    llvm::outs() << "Region with " << region.getBlocks().size() << " blocks:\n";
#endif
    if (oldMutation != mutationCnt) {
      break;
    }
    for (Block &block : region.getBlocks()) {
#ifdef DEBUG
      llvm::outs() << "[MutationParser::printOperation] deal with block:\n";
#endif
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
#ifdef DEBUG
    llvm::outs() << "[MutationParser::printOperation] deal with operation:\n";
#endif
    Value res = op->getOpResult(rIdx);
    mb->add2Pool(res);
#ifdef DEBUG
    mb->print();
#endif
  }
  ++curPos;
}

mlir::Operation *MutationParser::getDifiningOpIncludeArgs(mlir::Value val) {
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
    return parentOp;
    // unsigned idx = 0;
    //   while (idx < parentOp->getNumOperands()) {
    //   if (block->getArgument(idx) == val)
    //   break;
    //   idx++;
    // }
    // return
    // MutationParser::getDifiningOpIncludeArgs(parentOp->getOperand(idx));
  }
}

// Note that: in the dataDepdencies vector, all operands must insert before the
// user opration. There is no need to change the order(to keep the data
// dependency).
void MutationParser::checkAllDataDependency(Operation *op,
                                            bool *dependentOnBlockArgs) {
  if (*dependentOnBlockArgs)
    return;
  if (!op)
    return;
  for (int oIdx = 0; oIdx < op->getNumOperands(); oIdx++) {
    mlir::Value operand = op->getOperand(oIdx);
    *dependentOnBlockArgs =
        (operand.getDefiningOp() == NULL) || *dependentOnBlockArgs;
    mlir::Operation *operandOperation = operand.getDefiningOp();
    // mlir::Operation* operandOperation = getDifiningOpIncludeArgs(operand);
    checkAllDataDependency(operandOperation, dependentOnBlockArgs);
  }
}

// Note that: in the dataDepdencies vector, all operands must insert before the
// user opration. There is no need to change the order(to keep the data
// dependency).
void MutationParser::getAllDataDependency(
    Operation *op, std::vector<mlir::Operation *> *dataDependencies) {
  if (!op)
    return;
  mlir::Operation *opParent = op->getParentOp();
  std::string opParentName = getOperationName(opParent);
  // if (op->getNumOperands() == 0)
  //     if (std::find(dataDependencies->begin(), dataDependencies->end(), op)
  //     == dataDependencies->end())
  //       dataDependencies->push_back(op);
  for (int oIdx = 0; oIdx < op->getNumOperands(); oIdx++) {
    mlir::Value operand = op->getOperand(oIdx);
    mlir::Operation *operandOperation = operand.getDefiningOp();
    // mlir::Operation* operandOperation = getDifiningOpIncludeArgs(operand);
    getAllDataDependency(operandOperation, dataDependencies);
  }
  if (std::find(dataDependencies->begin(), dataDependencies->end(), op) ==
      dataDependencies->end())
    dataDependencies->push_back(op);
}

#endif // TRAVERSE_h