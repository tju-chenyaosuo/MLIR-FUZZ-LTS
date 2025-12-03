{
	cd mlirsmith/
	time timeout 600 python3 mlirsmith.py \
	--mlirsmith /MLIR-FUZZ-LTS/build/bin/mlirsmith \
	--testcase_base testcase \
	--mliropt /MLIR-FUZZ-LTS/build/bin/mlir-opt \
	--optfile opt.txt \
	--collector /MLIR-FUZZ-LTS/build/bin/CollectCoverage \
	--out_base report > log
} &

{
	cd mlirod/

	python3 generate-seed.py \
	--mlirsmith /MLIR-FUZZ-LTS/build/bin/mlirsmith \
	--opt /MLIR-FUZZ-LTS/build/bin/mlir-opt

	cp -r seeds existing_seeds

	time timeout 600 python3 mlirfuzz.py \
	--seed existing \
	--existing_seed_dir existing_seeds/ \
	--replace_data_edge /MLIR-FUZZ-LTS/build/bin/ReplaceDataEdge \
	--replace_control_edge /MLIR-FUZZ-LTS/build/bin/ReplaceControlEdge \
	--delete_node /MLIR-FUZZ-LTS/build/bin/DeleteNode \
	--create_node /MLIR-FUZZ-LTS/build/bin/CreateNode \
	--mlir_opt /MLIR-FUZZ-LTS/build/bin/mlir-opt \
	--optfile opt.txt \
	--collector /MLIR-FUZZ-LTS/build/bin/CollectCoverage \
	--clean True > log.txt
} &

{
	cd MLIRod_FIXUB/

	python3 generate-seed.py \
	--mlirsmith /MLIR-FUZZ-LTS/build/bin/desilmlirsmith \
	--opt /MLIR-FUZZ-LTS/build/bin/mlir-opt

	cp -r seeds existing_seeds

	time timeout 600 python3 MLIRod_FIXUB.py \
	--seed existing \
	--diff_num 1 \
	--opt_num 1 \
	--init_size 3000 \
	--lowering_iter_limit 50 \
	--existing_seed_dir existing_seeds \
	--mlir_opt /MLIR-FUZZ-LTS/build/bin/mlir-opt \
	--mlirsmith /MLIR-FUZZ-LTS/build/bin/desilmlirsmith \
	--fixub /MLIR-FUZZ-LTS/build/bin/FixUBnew \
	--checksum /MLIR-FUZZ-LTS/build/bin/CheckSum \
	--replace_data_edge /MLIR-FUZZ-LTS/build/bin/ReplaceDataEdge \
	--replace_control_edge /MLIR-FUZZ-LTS/build/bin/ReplaceControlEdge \
	--delete_node /MLIR-FUZZ-LTS/build/bin/DeleteNode \
	--create_node /MLIR-FUZZ-LTS/build/bin/CreateNode \
	--collector /MLIR-FUZZ-LTS/build/bin/CollectCoverage \
	--clean True >log.log 2>error.log
} &

{
	cd MLIRSmith_FIXUB/

	time timeout 600 python3 MLIRSmith_FIXUB.py \
	--diff_num 1 \
	--opt_num 1 \
	--init_size 3000 \
	--lowering_iter_limit 50 \
	--mlir_opt /MLIR-FUZZ-LTS/build/bin/mlir-opt \
	--mlirsmith /MLIR-FUZZ-LTS/build/bin/desilmlirsmith \
	--fixub /MLIR-FUZZ-LTS/build/bin/FixUBnew \
	--checksum /MLIR-FUZZ-LTS/build/bin/CheckSum \
	--clean True > log.log
} &

wait
echo "done"