#include "NodeGenerator.h"
#include "MutationUtil.h"
#include "TypeBase.h"

OpGenerator indexAddGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(10)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
		mlir::Value res = index::AddOp::create(builder, loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexAndGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(10)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
    mlir::Value res = index::AndOp::create(builder, loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexBoolConstantGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    mlir::Value res = index::BoolConstantOp::create(builder, loc, rollIdx(2));
		mb->add2Pool(res);
    return true;
  };
}

OpGenerator indexCastSGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    if (rollIdx(2)) {
			std::vector<mlir::Value> intCandidates = std::vector<mlir::Value>();
			mb->search(int_filter, &intCandidates);
			if (intCandidates.empty()) {
				intCandidates.push_back(generateInteger(builder, loc, builder.getI32Type()));
			}
			auto operand = intCandidates[rollIdx(intCandidates.size())];
			mlir::Value res = index::CastSOp::create(builder, loc, builder.getIndexType(), operand);
			mb->add2Pool(res);
    } else {
			std::vector<mlir::Value> idxCandidates = std::vector<mlir::Value>();
			mb->search(index_filter, &idxCandidates);
			if (idxCandidates.empty()) {
				idxCandidates.push_back(generateIndex(builder, loc, rollIdx(10)));
			}
			auto operand = idxCandidates[rollIdx(idxCandidates.size())];
      mlir::Value res = index::CastSOp::create(builder, loc, builder.getI32Type(), operand);
			mb->add2Pool(res);
    }
		return true;
  };
}

OpGenerator indexCastUGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    if (rollIdx(2)) {
			std::vector<mlir::Value> intCandidates = std::vector<mlir::Value>();
			mb->search(int_filter, &intCandidates);
			if (intCandidates.empty()) {
				intCandidates.push_back(generateInteger(builder, loc, builder.getI32Type()));
			}
			auto operand = intCandidates[rollIdx(intCandidates.size())];
			mlir::Value res = index::CastUOp::create(builder, loc, builder.getIndexType(), operand);
			mb->add2Pool(res);
    } else {
			std::vector<mlir::Value> idxCandidates = std::vector<mlir::Value>();
			mb->search(index_filter, &idxCandidates);
			if (idxCandidates.empty()) {
				idxCandidates.push_back(generateIndex(builder, loc, rollIdx(10)));
			}
			auto operand = idxCandidates[rollIdx(idxCandidates.size())];
			mlir::Value res = index::CastUOp::create(builder, loc, builder.getI32Type(), operand);
			mb->add2Pool(res);
    }
		return true;
  };
}

OpGenerator indexCeilDivSGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
		mlir::Value res = index::CeilDivSOp::create(builder, loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexCeilDivUGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
    mlir::Value res = index::CeilDivUOp::create(builder, loc, operand0, operand1);
    mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexConstantGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    mlir::Value res = index::ConstantOp::create(builder, loc, rollIdx(8));
    mb->add2Pool(res);
		return true;		
  };
}

OpGenerator indexDivSGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
    mlir::Value res = index::DivSOp::create(builder, loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexDivUGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
    mlir::Value res = index::DivUOp::create(builder, loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexFloorDivSGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
    mlir::Value res = index::FloorDivSOp::create(builder, loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexMaxSGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
    mlir::Value res = index::MaxSOp::create(builder, loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexMaxUGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
		mlir::Value res = index::MaxUOp::create(builder, loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexMulGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
		mlir::Value res = index::MulOp::create(builder, loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexOrGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
		mlir::Value res = index::OrOp::create(builder, loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexRemSGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
		mlir::Value res = index::RemSOp::create(builder, loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexRemUGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
		mlir::Value res = index::RemUOp::create(builder, loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexShLGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
		mlir::Value res = index::ShlOp::create(builder, loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexShrSGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
		mlir::Value res = index::ShrSOp::create(builder, loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexShrUGenerator(){
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
		mlir::Value res = index::ShrUOp::create(builder, loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexSizeOfGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		mlir::Value res =  index::SizeOfOp::create(builder, loc);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexSubGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
		mlir::Value res = index::SubOp::create(builder, loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexXorGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
		mlir::Value res = index::XOrOp::create(builder, loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}