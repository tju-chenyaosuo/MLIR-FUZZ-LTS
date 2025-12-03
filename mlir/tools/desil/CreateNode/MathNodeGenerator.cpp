#include "NodeGenerator.h"
#include "MutationUtil.h"
#include "TypeBase.h"

OpGenerator absFGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_float_filter(s) || ranked_tensor_filter(s, v);
				if (!cond1) { return false; }
				return v.getType().isa<FloatType>() || 
					(v.getType().isa<ShapedType>() && 
					v.getType().dyn_cast<ShapedType>().getElementType().isa<FloatType>());
			},
			&operandCandidates
		);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
		auto operand = operandCandidates[rollIdx(operandCandidates.size())];
		mlir::Value value = math::AbsFOp::create(builder, loc, operand);
		mb->add2Pool(value);
		return true;
  };
}

OpGenerator absIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_float_filter(s) || ranked_tensor_filter(s, v);
				if (!cond1) { return false; }
				return v.getType().isa<IntegerType>() || 
					(v.getType().isa<ShapedType>() && 
					v.getType().dyn_cast<ShapedType>().getElementType().isa<IntegerType>());
			},
			&operandCandidates
		);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(generateInteger(builder, loc, builder.getI32Type()));
    }    
    auto operand = operandCandidates[rollIdx(operandCandidates.size())];
    mlir::Value value = math::AbsIOp::create(builder, loc, operand);
		mb->add2Pool(value);
    return true;
  };
}

OpGenerator atanGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_float_filter(s) || ranked_tensor_filter(s, v);
				if (!cond1) { return false; }
				return v.getType().isa<FloatType>() || 
					(v.getType().isa<ShapedType>() && 
					v.getType().dyn_cast<ShapedType>().getElementType().isa<FloatType>());
			},
			&operandCandidates
		);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = operandCandidates[rollIdx(operandCandidates.size())];
    mlir::Value value = math::AtanOp::create(builder, loc, operand);
		mb->add2Pool(value);
		return true;
  };
}

OpGenerator atan2Generator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		auto elemTy = builder.getF32Type();
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(float_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateElement(builder, loc, elemTy));
		}

		std::vector<mlir::Value> tensorCandidates = std::vector<mlir::Value>();
		mb->search(static_tensor_filter, &tensorCandidates);
		if (tensorCandidates.empty()) {
			tensorCandidates.push_back(
				generateStaticShapedTensor(builder, loc, randomStaticShapedTensorType(elemTy)));
		}

		candidates.insert(candidates.end(), tensorCandidates.begin(), tensorCandidates.end());
		mlir::Value op1 = candidates[rollIdx(candidates.size())];
		std::string op1Str = getValueTypeStr(op1);
		mlir::Value op2 = op1;
		std::vector<mlir::Value> realCandidates = std::vector<mlir::Value>();
		for (mlir::Value op : candidates) {
			std::string opStr = getValueTypeStr(op);
			if (op1Str == opStr) { realCandidates.push_back(op); }
		}
		if (realCandidates.empty()) {
			return false;
		}
		op2 = realCandidates[rollIdx(realCandidates.size())];
		mlir::Value value = math::Atan2Op::create(builder, loc, op1, op2);
		mb->add2Pool(value);
		return true;
  };
}

OpGenerator ceilGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_float_filter(s) || ranked_tensor_filter(s, v);
				if (!cond1) { return false; }
				return v.getType().isa<FloatType>() || 
					(v.getType().isa<ShapedType>() && 
					v.getType().dyn_cast<ShapedType>().getElementType().isa<FloatType>());
			},
			&operandCandidates
		);
    if (operandCandidates.empty()) {
      operandCandidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = operandCandidates[rollIdx(operandCandidates.size())];
    mlir::Value value = math::CeilOp::create(builder, loc, operand);
		mb->add2Pool(value);
    return true;
  };
}

OpGenerator copySignGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		auto elemTy = builder.getF32Type();
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(float_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateElement(builder, loc, elemTy));
		}

		// std::vector<mlir::Value> tensorCandidates = std::vector<mlir::Value>();
		// mb->search(static_tensor_filter, &tensorCandidates);
		// if (tensorCandidates.empty()) {
		// 	tensorCandidates.push_back(
		// 		generateStaticShapedTensor(builder, loc, randomStaticShapedTensorType(elemTy)));
		// }
		// candidates.insert(candidates.end(), tensorCandidates.begin(), tensorCandidates.end());
		
		mlir::Value op1 = candidates[rollIdx(candidates.size())];
		std::string op1Str = getValueTypeStr(op1);
		mlir::Value op2 = op1;
		std::vector<mlir::Value> realCandidates = std::vector<mlir::Value>();
		for (mlir::Value op : candidates) {
			std::string opStr = getValueTypeStr(op);
			if (op1Str == opStr) { realCandidates.push_back(op); }
		}
		if (realCandidates.empty()) {
			return false;
		}
		op2 = realCandidates[rollIdx(realCandidates.size())];
		mlir::Value value = math::CopySignOp::create(builder, loc, op1, op2);
		mb->add2Pool(value);
		return true;
  };
} // f b

OpGenerator cosGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_float_filter(s) || ranked_tensor_filter(s, v);
				if (!cond1) { return false; }
				return v.getType().isa<FloatType>() || 
					(v.getType().isa<ShapedType>() && 
					v.getType().dyn_cast<ShapedType>().getElementType().isa<FloatType>());
			},
			&operandCandidates
		);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
		auto operand = operandCandidates[rollIdx(operandCandidates.size())];
		mlir::Value value = math::CosOp::create(builder, loc, operand);
		mb->add2Pool(value);
		return true;
  };
} // f u

OpGenerator sinGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_float_filter(s) || ranked_tensor_filter(s, v);
				if (!cond1) { return false; }
				return v.getType().isa<FloatType>() || 
					(v.getType().isa<ShapedType>() && 
					v.getType().dyn_cast<ShapedType>().getElementType().isa<FloatType>());
			},
			&operandCandidates
		);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }

    auto operand = operandCandidates[rollIdx(operandCandidates.size())];
    mlir::Value value = math::SinOp::create(builder, loc, operand);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator ctlzGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_filter(s) || ranked_tensor_filter(s, v);
				if (!cond1) { return false; }
				return v.getType().isa<IntegerType>() || 
					(v.getType().isa<ShapedType>() && 
					v.getType().dyn_cast<ShapedType>().getElementType().isa<IntegerType>());
			},
			&operandCandidates
		);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(generateInteger(builder, loc, builder.getI32Type()));
    }

    auto operand = operandCandidates[rollIdx(operandCandidates.size())];
    mlir::Value value = math::CountLeadingZerosOp::create(builder, loc, operand);
		mb->add2Pool(value);
		return true;
  };
}

// TODO
OpGenerator cttzGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_filter(s) || ranked_tensor_filter(s, v);
				if (!cond1) { return false; }
				return v.getType().isa<IntegerType>() || 
					(v.getType().isa<ShapedType>() && 
					v.getType().dyn_cast<ShapedType>().getElementType().isa<IntegerType>());
			},
			&operandCandidates
		);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(generateInteger(builder, loc, builder.getI32Type()));
    }
    auto operand = operandCandidates[rollIdx(operandCandidates.size())];
    mlir::Value value = math::CountTrailingZerosOp::create(builder, loc, operand);
		mb->add2Pool(value);
    return true;
  };
}

OpGenerator ctpopGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_filter(s) || ranked_tensor_filter(s, v);
				if (!cond1) { return false; }
				return v.getType().isa<IntegerType>() || 
					(v.getType().isa<ShapedType>() && 
					v.getType().dyn_cast<ShapedType>().getElementType().isa<IntegerType>());
			},
			&operandCandidates
		);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(generateInteger(builder, loc, builder.getI32Type()));
    }
		auto operand = operandCandidates[rollIdx(operandCandidates.size())];
    mlir::Value value = math::CtPopOp::create(builder, loc, operand);
		mb->add2Pool(value);
    return true;
  };
}

OpGenerator expGenerator() {
	return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_float_filter(s) || ranked_tensor_filter(s, v);
				if (!cond1) { return false; }
				return v.getType().isa<FloatType>() || 
					(v.getType().isa<ShapedType>() && 
					v.getType().dyn_cast<ShapedType>().getElementType().isa<FloatType>());
			},
			&operandCandidates
		);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
		auto operand = operandCandidates[rollIdx(operandCandidates.size())];
    mlir::Value value = math::ExpOp::create(builder, loc, operand);
		mb->add2Pool(value);
    return true;
  };
}

OpGenerator exp2Generator() {
	return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_float_filter(s) || ranked_tensor_filter(s, v);
				if (!cond1) { return false; }
				return v.getType().isa<FloatType>() || 
					(v.getType().isa<ShapedType>() && 
					v.getType().dyn_cast<ShapedType>().getElementType().isa<FloatType>());
			},
			&operandCandidates
		);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
		auto operand = operandCandidates[rollIdx(operandCandidates.size())];
    mlir::Value value = math::Exp2Op::create(builder, loc, operand);
		mb->add2Pool(value);
    return true;
  };
}

OpGenerator expm1Generator() {
	return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_float_filter(s) || ranked_tensor_filter(s, v);
				if (!cond1) { return false; }
				return v.getType().isa<FloatType>() || 
					(v.getType().isa<ShapedType>() && 
					v.getType().dyn_cast<ShapedType>().getElementType().isa<FloatType>());
			},
			&operandCandidates
		);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
		auto operand = operandCandidates[rollIdx(operandCandidates.size())];
    mlir::Value value = math::ExpM1Op::create(builder, loc, operand);
		mb->add2Pool(value);
    return true;
  };
}

OpGenerator floorGenerator() {

	return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_float_filter(s) || ranked_tensor_filter(s, v);
				if (!cond1) { return false; }
				return v.getType().isa<FloatType>() || 
					(v.getType().isa<ShapedType>() && 
					v.getType().dyn_cast<ShapedType>().getElementType().isa<FloatType>());
			},
			&operandCandidates
		);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
		auto operand = operandCandidates[rollIdx(operandCandidates.size())];
    mlir::Value value = math::FloorOp::create(builder, loc, operand);
		mb->add2Pool(value);
    return true;
  };
}

// OpGenerator fmaGenerator() {
// 	return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
// 		auto elemTy = builder.getF32Type();
// 		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
// 		mb->search(int_float_filter, &candidates);
// 		if (candidates.empty()) {
// 			candidates.push_back(generateElement(builder, loc, elemTy));
// 		}

// 		std::vector<mlir::Value> tensorCandidates = std::vector<mlir::Value>();
// 		mb->search(static_tensor_filter, &tensorCandidates);
// 		if (tensorCandidates.empty()) {
// 			tensorCandidates.push_back(
// 				generateStaticShapedTensor(builder, loc, randomStaticShapedTensorType(elemTy)));
// 		}
// 		candidates.insert(candidates.end(), tensorCandidates.begin(), tensorCandidates.end());
		
// 		mlir::Value op1 = candidates[rollIdx(candidates.size())];
// 		std::string op1Str = getValueTypeStr(op1);
// 		mlir::Value op2 = op1;
// 		std::vector<mlir::Value> realCandidates = std::vector<mlir::Value>();
// 		for (mlir::Value op : candidates) {
// 			std::string opStr = getValueTypeStr(op);
// 			if (op1Str == opStr) { realCandidates.push_back(op); }
// 		}
// 		if (realCandidates.empty()) {
// 			return false;
// 		}
// 		op2 = realCandidates[rollIdx(realCandidates.size())];
// 		mlir::Value value = math::FmaOp::create(builder, loc, op1, op2);
// 		mb->add2Pool(value);
// 		return true;
//   };
// } // f t

OpGenerator ipowiGenerator() {
	return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		auto elemTy = builder.getI32Type();
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(int_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateElement(builder, loc, elemTy));
		}
		
		mlir::Value op1 = candidates[rollIdx(candidates.size())];
		std::string op1Str = getValueTypeStr(op1);
		mlir::Value op2 = op1;
		std::vector<mlir::Value> realCandidates = std::vector<mlir::Value>();
		for (mlir::Value op : candidates) {
			std::string opStr = getValueTypeStr(op);
			if (op1Str == opStr) { realCandidates.push_back(op); }
		}
		if (realCandidates.empty()) {
			return false;
		}
		op2 = realCandidates[rollIdx(realCandidates.size())];
		mlir::Value value = math::IPowIOp::create(builder, loc, op1, op2);
		mb->add2Pool(value);
		return true;
  };
} // i b

OpGenerator logGenerator() {
	return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_float_filter(s) || ranked_tensor_filter(s, v);
				if (!cond1) { return false; }
				return v.getType().isa<FloatType>() || 
					(v.getType().isa<ShapedType>() && 
					v.getType().dyn_cast<ShapedType>().getElementType().isa<FloatType>());
			},
			&operandCandidates
		);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
		auto operand = operandCandidates[rollIdx(operandCandidates.size())];
    mlir::Value value = math::LogOp::create(builder, loc, operand);
		mb->add2Pool(value);
    return true;
  };
} // f u

OpGenerator log10Generator() {
	return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_float_filter(s) || ranked_tensor_filter(s, v);
				if (!cond1) { return false; }
				return v.getType().isa<FloatType>() || 
					(v.getType().isa<ShapedType>() && 
					v.getType().dyn_cast<ShapedType>().getElementType().isa<FloatType>());
			},
			&operandCandidates
		);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
		auto operand = operandCandidates[rollIdx(operandCandidates.size())];
    mlir::Value value = math::Log10Op::create(builder, loc, operand);
		mb->add2Pool(value);
    return true;
  };
}

OpGenerator log1pGenerator() {
	return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_float_filter(s) || ranked_tensor_filter(s, v);
				if (!cond1) { return false; }
				return v.getType().isa<FloatType>() || 
					(v.getType().isa<ShapedType>() && 
					v.getType().dyn_cast<ShapedType>().getElementType().isa<FloatType>());
			},
			&operandCandidates
		);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
		auto operand = operandCandidates[rollIdx(operandCandidates.size())];
    mlir::Value value = math::Log1pOp::create(builder, loc, operand);
		mb->add2Pool(value);
    return true;
  };
}

OpGenerator log2Generator() {
	return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_float_filter(s) || ranked_tensor_filter(s, v);
				if (!cond1) { return false; }
				return v.getType().isa<FloatType>() || 
					(v.getType().isa<ShapedType>() && 
					v.getType().dyn_cast<ShapedType>().getElementType().isa<FloatType>());
			},
			&operandCandidates
		);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
		auto operand = operandCandidates[rollIdx(operandCandidates.size())];
    mlir::Value value = math::Log2Op::create(builder, loc, operand);
		mb->add2Pool(value);
    return true;
  };
}

OpGenerator powfGenerator() {
	return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		auto elemTy = builder.getF32Type();
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(float_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateElement(builder, loc, elemTy));
		}

		std::vector<mlir::Value> tensorCandidates = std::vector<mlir::Value>();
		mb->search(static_tensor_filter, &tensorCandidates);
		if (tensorCandidates.empty()) {
			tensorCandidates.push_back(
				generateStaticShapedTensor(builder, loc, randomStaticShapedTensorType(elemTy)));
		}
		candidates.insert(candidates.end(), tensorCandidates.begin(), tensorCandidates.end());
		
		mlir::Value op1 = candidates[rollIdx(candidates.size())];
		std::string op1Str = getValueTypeStr(op1);
		mlir::Value op2 = op1;
		std::vector<mlir::Value> realCandidates = std::vector<mlir::Value>();
		for (mlir::Value op : candidates) {
			std::string opStr = getValueTypeStr(op);
			if (op1Str == opStr) { realCandidates.push_back(op); }
		}
		if (realCandidates.empty()) {
			return false;
		}
		op2 = realCandidates[rollIdx(realCandidates.size())];
		mlir::Value value = math::PowFOp::create(builder, loc, op1, op2);
		mb->add2Pool(value);
		return true;
  };
} // f b

OpGenerator rsqrtGenerator() {
	return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_float_filter(s) || ranked_tensor_filter(s, v);
				if (!cond1) { return false; }
				return v.getType().isa<FloatType>() || 
					(v.getType().isa<ShapedType>() && 
					v.getType().dyn_cast<ShapedType>().getElementType().isa<FloatType>());
			},
			&operandCandidates
		);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
		auto operand = operandCandidates[rollIdx(operandCandidates.size())];
    mlir::Value value = math::RsqrtOp::create(builder, loc, operand);
		mb->add2Pool(value);
    return true;
  };
} // f u

OpGenerator sqrtGenerator() {
	return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_float_filter(s) || ranked_tensor_filter(s, v);
				if (!cond1) { return false; }
				return v.getType().isa<FloatType>() || 
					(v.getType().isa<ShapedType>() && 
					v.getType().dyn_cast<ShapedType>().getElementType().isa<FloatType>());
			},
			&operandCandidates
		);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
		auto operand = operandCandidates[rollIdx(operandCandidates.size())];
    mlir::Value value = math::SqrtOp::create(builder, loc, operand);
		mb->add2Pool(value);
    return true;
  };
}

OpGenerator tanGenerator() {
	return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_float_filter(s) || ranked_tensor_filter(s, v);
				if (!cond1) { return false; }
				return v.getType().isa<FloatType>() || 
					(v.getType().isa<ShapedType>() && 
					v.getType().dyn_cast<ShapedType>().getElementType().isa<FloatType>());
			},
			&operandCandidates
		);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
		auto operand = operandCandidates[rollIdx(operandCandidates.size())];
    mlir::Value value = math::TanOp::create(builder, loc, operand);
		mb->add2Pool(value);
    return true;
  };
}

OpGenerator tanhGenerator() {
	return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_float_filter(s) || ranked_tensor_filter(s, v);
				if (!cond1) { return false; }
				return v.getType().isa<FloatType>() || 
					(v.getType().isa<ShapedType>() && 
					v.getType().dyn_cast<ShapedType>().getElementType().isa<FloatType>());
			},
			&operandCandidates
		);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
		auto operand = operandCandidates[rollIdx(operandCandidates.size())];
    mlir::Value value = math::TanhOp::create(builder, loc, operand);
		mb->add2Pool(value);
    return true;
  };
}

OpGenerator roundEvenGenerator() {
	return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_float_filter(s) || ranked_tensor_filter(s, v);
				if (!cond1) { return false; }
				return v.getType().isa<FloatType>() || 
					(v.getType().isa<ShapedType>() && 
					v.getType().dyn_cast<ShapedType>().getElementType().isa<FloatType>());
			},
			&operandCandidates
		);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
		auto operand = operandCandidates[rollIdx(operandCandidates.size())];
    mlir::Value value = math::RoundEvenOp::create(builder, loc, operand);
		mb->add2Pool(value);
    return true;
  };
}

OpGenerator roundGenerator() {
	return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_float_filter(s) || ranked_tensor_filter(s, v);
				if (!cond1) { return false; }
				return v.getType().isa<FloatType>() || 
					(v.getType().isa<ShapedType>() && 
					v.getType().dyn_cast<ShapedType>().getElementType().isa<FloatType>());
			},
			&operandCandidates
		);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
		auto operand = operandCandidates[rollIdx(operandCandidates.size())];
    mlir::Value value = math::RoundOp::create(builder, loc, operand);
		mb->add2Pool(value);
    return true;
  };
}

OpGenerator truncGenerator() {
	return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_float_filter(s) || ranked_tensor_filter(s, v);
				if (!cond1) { return false; }
				return v.getType().isa<FloatType>() || 
					(v.getType().isa<ShapedType>() && 
					v.getType().dyn_cast<ShapedType>().getElementType().isa<FloatType>());
			},
			&operandCandidates
		);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
		auto operand = operandCandidates[rollIdx(operandCandidates.size())];
    mlir::Value value = math::TruncOp::create(builder, loc, operand);
		mb->add2Pool(value);
    return true;
  };
}

OpGenerator fpowiGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		auto elemTy = builder.getF32Type();
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(float_filter, &candidates);

		std::vector<mlir::Value> tensorCandidates = std::vector<mlir::Value>();
		auto tensorTy = randomStaticShapedTensorType(elemTy);
		std::string tensorTyStr = getValueTypeStr(tensorTy);
		if (mb->valuePool.pool.count(tensorTyStr)) {
			tensorCandidates.insert(
				tensorCandidates.end(), 
				mb->valuePool.pool[tensorTyStr].begin(), 
				mb->valuePool.pool[tensorTyStr].end());
		}
		if (tensorCandidates.empty()) {
			tensorCandidates.push_back(generateStaticShapedTensor(builder, loc, tensorTy));
		}
		candidates.insert(candidates.end(), tensorCandidates.begin(), tensorCandidates.end());

		auto operand0 = candidates[rollIdx(candidates.size())];

		auto ctx = builder.getContext();
    auto i32 = IntegerType::get(ctx, 32);
    std::vector<mlir::Value> candidates2;
		if (operand0.getType().dyn_cast<FloatType>()) {
			std::vector<mlir::Value> eleCandidates = std::vector<mlir::Value>();
			mb->search(int_float_filter, &eleCandidates);
			if (eleCandidates.empty()) { 
				eleCandidates.push_back(generateInteger(builder, loc, builder.getI32Type())); 
			}
      for (auto ele : eleCandidates) {
        if (ele.getType() == i32) {
          candidates2.push_back(ele);
        }
      }
    } else { // otherwise it should be Tensor Type
      auto t = RankedTensorType::get(operand0.getType().dyn_cast<ShapedType>().getShape(), i32);
      std::string tstr = getValueTypeStr(t);
			std::vector<mlir::Value> candidates2 = std::vector<mlir::Value>();
			candidates2.push_back(generateStaticShapedTensor(builder, loc, t));
    }
		if (candidates2.empty()) {
      Type ty;
      if (operand0.getType().isa<FloatType>()) {
        ty = builder.getI32Type();
      } else {
        ty = RankedTensorType::get(operand0.getType().dyn_cast<ShapedType>().getShape(), i32);
      }
      candidates2.push_back(generateTypedValue(builder, loc, ty, mb));
    }
		auto operand1 = candidates2[rollIdx(candidates2.size())];

    mlir::Value value = math::FPowIOp::create(builder, loc, operand0, operand1);
		mb->add2Pool(value);
		return true;
  };
} // f,i
