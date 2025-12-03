"""
This file is used to test the functionality of mlirsmith, mlirod, and DESIL.
"""
import os
import secrets
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread
from enum import Enum
from tqdm import tqdm

MLIRSMITH = '/MLIR-FUZZ-LTS/build/bin/mlirsmith'
DESIL_MLIRSMITH = '/MLIR-FUZZ-LTS/build/bin/desilmlirsmith'
MLIR_OPT = '/MLIR-FUZZ-LTS/build/bin/mlir-opt'

CHECKSUM = '/MLIR-FUZZ-LTS/build/bin/CheckSum'
FIXUB = '/MLIR-FUZZ-LTS/build/bin/FixUBnew'

REPLACE_DATA_EDGE = '/MLIR-FUZZ-LTS/build/bin/ReplaceDataEdge'
OPERAND_PROB = 0.1
REPLACE_CONTROL_EDGE = '/MLIR-FUZZ-LTS/build/bin/ReplaceControlEdge'
DELETE_NODE = '/MLIR-FUZZ-LTS/build/bin/DeleteNode'
CREATE_NODE = '/MLIR-FUZZ-LTS/build/bin/CreateNode'
COLLECTOR = '/MLIR-FUZZ-LTS/build/bin/CollectCoverage'
DEPTH = 2

class DaemonThreadPoolExecutor(ThreadPoolExecutor):
    def _thread_factory(self, *args, **kwargs):
        t = Thread(*args, **kwargs)
        t.daemon = True
        return t

    def submit(self, fn, *args, **kwargs):
        # Use default submit
        return super().submit(fn, *args, **kwargs)

class MLIRSmithTestResult(Enum):
	SUCCESS = 1
	GENERATION_FAILED = 2
	PARSE_FAILED = 3

class MLIRodTestResult(Enum):
	GENERATION_FAILED = 'GENERATION_FAILED'

	REPLACE_DATA_EDGE_FAILED = 'REPLACE_DATA_EDGE_FAILED'
	REPLACE_DATA_EDGE_PARSE_FAILED = 'REPLACE_DATA_EDGE_PARSE_FAILED'
	REPALCE_DATA_EDGE_COLLECT_FAILED = 'REPALCE_DATA_EDGE_COLLECT_FAILED'

	REPALCE_CONTROL_EDGE_FAILED = 'REPALCE_CONTROL_EDGE_FAILED'
	REPLACE_CONTROL_EDGE_PARSE_FAILED = 'REPLACE_CONTROL_EDGE_PARSE_FAILED'
	REPLACE_CONTROL_EDGE_COLLECT_FAILED = 'REPLACE_CONTROL_EDGE_COLLECT_FAILED'

	DELETE_NODE_FAILED = 'DELETE_NODE_FAILED'
	DELETE_NODE_PARSE_FAILED = 'DELETE_NODE_PARSE_FAILED'
	DELETE_NODE_COLLECT_FAILED = 'DELETE_NODE_COLLECT_FAILED'

	CREATE_NODE_FAILED = 'CREATE_NODE_FAILED'
	CREATE_NODE_PARSE_FAILED = 'CREATE_NODE_PARSE_FAILED'
	CREATE_NODE_COLLECT_FAILED = 'CREATE_NODE_COLLECT_FAILED'

	COLLECT_FAILED = 'COLLECT_FAILED'

	SUCCESS = 'SUCCESS'

class DESILTestResult(Enum):
	GENERATION_FAILED = 'GENERATION_FAILED'
	GENERATION_PARSE_FAILED = 'GENERATION_PARSE_FAILED'
	FIXUB_FAILED = 'FIXUB_FAILED'
	FIXUB_PARSE_FAIELD = 'FIXUB_PARSE_FAIELD'
	CHECKSUM_FAILED = 'CHECKSUM_FAILED'
	CHECKSUM_PARSE_FAILED = 'CHECKSUM_PARSE_FAILED'

	REPLACE_CONTROL_FAILED = 'REPLACE_CONTROL_FAILED'
	REPLACE_CONTROL_PARSE_FAILED = 'REPLACE_CONTROL_PARSE_FAILED'
	REPLACE_CONTROL_FIXUB_FAILED = 'REPLACE_CONTROL_FIXUB_FAILED'
	REPLACE_CONTROL_FIXUB_PARSE_FAILED = 'REPLACE_CONTROL_FIXUB_PARSE_FAILED'
	REPLACE_CONTROL_CHECKSUM_FAILED = 'REPLACE_CONTROL_CHECKSUM_FAILED'
	REPALCE_CONTROL_CHECKSUM_PARSE_FAILED = 'REPALCE_CONTROL_CHECKSUM_PARSE_FAILED'

	REPLACE_DATA_FAILED = 'REPLACE_DATA_FAILED'
	REPLACE_DATA_PARSE_FAILED = 'REPLACE_DATA_PARSE_FAILED'
	REPLACE_DATA_FIXUB_FAILED = 'REPLACE_DATA_FIXUB_FAILED'
	REPLACE_DATA_FIXUB_PARSE_FAILED = 'REPLACE_DATA_FIXUB_PARSE_FAILED'
	REPLACE_DATA_CHECKSUM_FAILED = 'REPLACE_DATA_CHECKSUM_FAILED'
	REPALCE_DATA_CHECKSUM_PARSE_FAILED = 'REPALCE_DATA_CHECKSUM_PARSE_FAILED'

	DELETE_NODE_FAILED = 'DELETE_NODE_FAILED'
	DELETE_NODE_PARSE_FAILED = 'DELETE_NODE_PARSE_FAILED'
	DELETE_NODE_FIXUB_FAILED = 'DELETE_NODE_FIXUB_FAILED'
	DELETE_NODE_FIXUB_PARSE_FAILED = 'DELETE_NODE_FIXUB_PARSE_FAILED'
	DELETE_NODE_CHECKSUM_FAILED = 'DELETE_NODE_CHECKSUM_FAILED'
	DELETE_NODE_CHECKSUM_PARSE_FAILED = 'DELETE_NODE_CHECKSUM_PARSE_FAILED'

	CREATE_NODE_FAILED = 'CREATE_NODE_FAILED'
	CREATE_NODE_PARSE_FAILED = 'CREATE_NODE_PARSE_FAILED'
	CREATE_NODE_FIXUB_FAILED = 'CREATE_NODE_FIXUB_FAILED'
	CREATE_NODE_FIXUB_PARSE_FAILED = 'CREATE_NODE_FIXUB_PARSE_FAILED'
	CREATE_NODE_CHECKSUM_FAILED = 'CREATE_NODE_CHECKSUM_FAILED'
	CREATE_NODE_CHECKSUM_PARSE_FAILED = 'CREATE_NODE_CHECKSUM_PARSE_FAILED'

	SUCCESS = 'SUCCESS'
		
def test_desil_single() -> DESILTestResult:
	mlir_file: str = f"{secrets.token_hex(8)}.mlir"
	fixed_mlir_file: str = f"{secrets.token_hex(8)}.mlir"
	checksumed_mlir_file: str = f"{secrets.token_hex(8)}.mlir"
	replace_control_mlir_file: str = f"{secrets.token_hex(8)}.mlir"
	repalce_data_mlir_file: str = f"{secrets.token_hex(8)}.mlir"
	delete_node_mlir_file: str = f"{secrets.token_hex(8)}.mlir"
	create_node_mlir_file: str = f"{secrets.token_hex(8)}.mlir"
	replace_control_fixub_mlir_file: str = f"{secrets.token_hex(8)}.mlir"
	replace_control_fixub_checksum_mlir_file: str = f"{secrets.token_hex(8)}.mlir"
	replace_data_fixub_mlir_file: str = f"{secrets.token_hex(8)}.mlir"
	replace_data_fixub_checksum_mlir_file: str = f"{secrets.token_hex(8)}.mlir"
	delete_node_fixub_mlir_file: str = f"{secrets.token_hex(8)}.mlir"
	delete_node_fixub_checksum_mlir_file: str = f"{secrets.token_hex(8)}.mlir"
	create_node_fixub_mlir_file: str = f"{secrets.token_hex(8)}.mlir"
	create_node_fixub_checksum_mlir_file: str = f"{secrets.token_hex(8)}.mlir"

	def run_helper(cmds: List[str], errors: List[DESILTestResult]):
		for cmd, err in zip(cmds, errors):
			if os.system(cmd):
				return err
		return DESILTestResult.SUCCESS

	def remove(file):
		if os.path.exists(file):
			os.remove(file)

	program_generate_cmd: str = f"{DESIL_MLIRSMITH} 2>{mlir_file}"
	program_generate_opt_cmd: str = f"{MLIR_OPT} {mlir_file} > /dev/null"
	fixub_cmd: str = f'{FIXUB} {mlir_file} 2>{fixed_mlir_file}'
	fixub_opt_cmd: str = f"{MLIR_OPT} {fixed_mlir_file} > /dev/null"
	checksum_cmd: str = f"{CHECKSUM} {fixed_mlir_file} 2>{checksumed_mlir_file}"
	checksum_opt_cmd: str = f"{MLIR_OPT} {checksumed_mlir_file} > /dev/null"
	ori_cmds = [program_generate_cmd, program_generate_opt_cmd, fixub_cmd, fixub_opt_cmd, checksum_cmd, checksum_opt_cmd]
	ori_errs = [DESILTestResult.GENERATION_FAILED, DESILTestResult.GENERATION_PARSE_FAILED, DESILTestResult.FIXUB_FAILED, DESILTestResult.FIXUB_PARSE_FAIELD, DESILTestResult.CHECKSUM_FAILED, DESILTestResult.CHECKSUM_PARSE_FAILED]
	res = run_helper(ori_cmds, ori_errs)
	if res != DESILTestResult.SUCCESS:
		return res

	replace_control_edge_cmd: str = f"{REPLACE_CONTROL_EDGE} {mlir_file} {replace_control_mlir_file} 1>/dev/null 2>/dev/null"
	replace_control_edge_opt_cmd: str = f"{MLIR_OPT} {replace_control_mlir_file} > /dev/null"
	replace_control_edge_fixub_cmd: str = f"{FIXUB} {replace_control_mlir_file} 2>{replace_control_fixub_mlir_file}"
	replace_control_edge_fixub_opt_cmd: str = f"{MLIR_OPT} {replace_control_fixub_mlir_file} > /dev/null"
	replace_control_edge_checksum_cmd: str = f"{CHECKSUM} {replace_control_fixub_mlir_file} 2>{replace_control_fixub_checksum_mlir_file}"
	repalce_control_edge_checksum_opt_cmd: str = f"{MLIR_OPT} {replace_control_fixub_checksum_mlir_file} > /dev/null"
	repalce_control_edge_cmds: List[str] = [
		replace_control_edge_cmd, replace_control_edge_opt_cmd, replace_control_edge_fixub_cmd, replace_control_edge_fixub_opt_cmd, replace_control_edge_checksum_cmd, repalce_control_edge_checksum_opt_cmd
	]
	repalce_control_edge_errs: List[DESILTestResult] = [
		DESILTestResult.REPLACE_CONTROL_FAILED, DESILTestResult.REPLACE_CONTROL_PARSE_FAILED, DESILTestResult.REPLACE_CONTROL_FIXUB_FAILED, DESILTestResult.REPLACE_CONTROL_FIXUB_PARSE_FAILED, DESILTestResult.REPLACE_CONTROL_CHECKSUM_FAILED, DESILTestResult.REPALCE_CONTROL_CHECKSUM_PARSE_FAILED
	]
	remove(fixed_mlir_file)
	remove(checksumed_mlir_file)
	res = run_helper(repalce_control_edge_cmds, repalce_control_edge_errs)
	if res != DESILTestResult.SUCCESS:
		return res

	replace_data_edge_cmd: str = f"{REPLACE_DATA_EDGE} --operand-prob {OPERAND_PROB} {mlir_file} {repalce_data_mlir_file} 1>/dev/null 2>/dev/null"
	replace_data_edge_opt_cmd: str = f"{MLIR_OPT} {repalce_data_mlir_file} > /dev/null"
	replace_data_edge_fixub_cmd: str = f"{FIXUB} {repalce_data_mlir_file} 2>{replace_data_fixub_mlir_file}"
	replace_data_edge_fixub_opt_cmd: str = f"{MLIR_OPT} {replace_data_fixub_mlir_file} > /dev/null"
	replace_data_edge_checksum_cmd: str = f"{CHECKSUM} {replace_data_fixub_mlir_file} 2>{replace_data_fixub_checksum_mlir_file}"
	repalce_data_edge_checksum_opt_cmd: str = f"{MLIR_OPT} {replace_data_fixub_checksum_mlir_file} > /dev/null"
	repalce_data_edge_cmds: List[str] = [
		replace_data_edge_cmd, replace_data_edge_opt_cmd, replace_data_edge_fixub_cmd, replace_data_edge_fixub_opt_cmd, replace_data_edge_checksum_cmd, repalce_data_edge_checksum_opt_cmd
	]
	repalce_data_edge_errs: List[DESILTestResult] = [
		DESILTestResult.REPLACE_DATA_FAILED, DESILTestResult.REPLACE_DATA_PARSE_FAILED, DESILTestResult.REPLACE_DATA_FIXUB_FAILED, DESILTestResult.REPLACE_DATA_FIXUB_PARSE_FAILED, DESILTestResult.REPLACE_DATA_CHECKSUM_FAILED, DESILTestResult.REPALCE_DATA_CHECKSUM_PARSE_FAILED
	]
	remove(replace_control_mlir_file)
	remove(replace_control_fixub_mlir_file)
	remove(replace_control_fixub_checksum_mlir_file)
	res = run_helper(repalce_data_edge_cmds, repalce_data_edge_errs)
	if res != DESILTestResult.SUCCESS:
		return res

	delete_node_cmd: str = f"{DELETE_NODE} {mlir_file} {delete_node_mlir_file} 1>/dev/null 2>/dev/null"
	delete_node_opt_cmd: str = f"{MLIR_OPT} {delete_node_mlir_file} > /dev/null"
	delete_node_fixub_cmd: str = f"{FIXUB} {delete_node_mlir_file} 2>{delete_node_fixub_mlir_file}"
	delete_node_fixub_opt_cmd: str = f"{MLIR_OPT} {delete_node_fixub_mlir_file} > /dev/null"
	delete_node_checksum_cmd: str = f"{CHECKSUM} {delete_node_fixub_mlir_file} 2>{delete_node_fixub_checksum_mlir_file}"
	delete_node_checksum_opt_cmd: str = f"{MLIR_OPT} {delete_node_fixub_checksum_mlir_file} > /dev/null"
	delete_node_cmds: List[str] = [
		delete_node_cmd, delete_node_opt_cmd, delete_node_fixub_cmd, delete_node_fixub_opt_cmd, delete_node_checksum_cmd, delete_node_checksum_opt_cmd
	]
	delete_node_errs: List[DESILTestResult] = [
		DESILTestResult.DELETE_NODE_FAILED, DESILTestResult.DELETE_NODE_PARSE_FAILED, DESILTestResult.DELETE_NODE_FIXUB_FAILED, DESILTestResult.DELETE_NODE_FIXUB_PARSE_FAILED, DESILTestResult.DELETE_NODE_CHECKSUM_FAILED, DESILTestResult.DELETE_NODE_CHECKSUM_PARSE_FAILED
	]
	remove(repalce_data_mlir_file)
	remove(replace_data_fixub_mlir_file)
	remove(replace_data_fixub_checksum_mlir_file)
	res = run_helper(delete_node_cmds, delete_node_errs)
	if res != DESILTestResult.SUCCESS:
		return res

	create_node_cmd: str = f"{CREATE_NODE} {mlir_file} {create_node_mlir_file} 1>/dev/null 2>/dev/null"
	create_node_opt_cmd: str = f"{MLIR_OPT} {create_node_mlir_file} > /dev/null"
	create_node_fixub_cmd: str = f"{FIXUB} {create_node_mlir_file} 2>{create_node_fixub_mlir_file}"
	create_node_fixub_opt_cmd: str = f"{MLIR_OPT} {create_node_fixub_mlir_file} > /dev/null"
	create_node_checksum_cmd: str = f"{CHECKSUM} {create_node_fixub_mlir_file} 2>{create_node_fixub_checksum_mlir_file}"
	create_node_checksum_opt_cmd: str = f"{MLIR_OPT} {create_node_fixub_checksum_mlir_file} > /dev/null"
	create_node_cmds: List[str] = [
		create_node_cmd, create_node_opt_cmd, create_node_fixub_cmd, create_node_fixub_opt_cmd, create_node_checksum_cmd, create_node_checksum_opt_cmd
	]
	create_node_errs: List[DESILTestResult] = [
		DESILTestResult.CREATE_NODE_FAILED, DESILTestResult.CREATE_NODE_PARSE_FAILED, DESILTestResult.CREATE_NODE_FIXUB_FAILED, DESILTestResult.CREATE_NODE_FIXUB_PARSE_FAILED, DESILTestResult.CREATE_NODE_CHECKSUM_FAILED, DESILTestResult.CREATE_NODE_CHECKSUM_PARSE_FAILED
	]
	remove(delete_node_mlir_file)
	remove(delete_node_fixub_mlir_file)
	remove(delete_node_fixub_checksum_mlir_file)
	res = run_helper(create_node_cmds, create_node_errs)
	if res != DESILTestResult.SUCCESS:
		return res
	
	remove(create_node_mlir_file)
	remove(create_node_fixub_mlir_file)
	remove(create_node_fixub_checksum_mlir_file)
	return DESILTestResult.SUCCESS


def test_mlirod_single() -> MLIRodTestResult:
	mlir_file: str = f"{secrets.token_hex(8)}.mlir"
	repalce_data_mlir_file: str = f"{secrets.token_hex(8)}.mlir"
	replace_control_mlir_file: str = f"{secrets.token_hex(8)}.mlir"
	delete_node_mlir_file: str = f"{secrets.token_hex(8)}.mlir"
	create_node_mlir_file: str = f"{secrets.token_hex(8)}.mlir"
	collect_file: str = f"{secrets.token_hex(8)}.txt"
	
	program_generate_cmd: str = f"{MLIRSMITH} 2>{mlir_file}"
	
	replace_data_edge_cmd: str = f"{REPLACE_DATA_EDGE} --operand-prob {OPERAND_PROB} {mlir_file} {repalce_data_mlir_file} 1>/dev/null 2>/dev/null"
	replace_data_edge_opt_cmd: str = f"{MLIR_OPT} {repalce_data_mlir_file} 1>/dev/null 2>/dev/null"
	repalce_data_edge_collect_cmd: str = f"{COLLECTOR} {repalce_data_mlir_file} --depth {DEPTH} {collect_file} 1>/dev/null 2>/dev/null"

	replace_control_edge_cmd: str = f"{REPLACE_CONTROL_EDGE} {mlir_file} {replace_control_mlir_file} 1>/dev/null 2>/dev/null"
	replace_control_edge_opt_cmd: str = f"{MLIR_OPT} {replace_control_mlir_file} 1>/dev/null 2>/dev/null"
	repalce_control_edge_collect_cmd: str = f"{COLLECTOR} {replace_control_mlir_file} --depth {DEPTH} {collect_file} 1>/dev/null 2>/dev/null"

	delete_node_cmd: str = f"{DELETE_NODE} {mlir_file} {delete_node_mlir_file} 1>/dev/null 2>/dev/null"
	delete_node_opt_cmd: str = f"{MLIR_OPT} {delete_node_mlir_file} 1>/dev/null 2>/dev/null"
	delete_node_collect_cmd: str = f"{COLLECTOR} {delete_node_mlir_file} --depth {DEPTH} {collect_file} 1>/dev/null 2>/dev/null"

	create_node_cmd: str = f"{CREATE_NODE} {mlir_file} {create_node_mlir_file} 1>/dev/null 2>/dev/null"
	create_node_opt_cmd: str = f"{MLIR_OPT} {create_node_mlir_file} 1>/dev/null 2>/dev/null"
	create_node_collect_cmd: str = f"{COLLECTOR} {create_node_mlir_file} --depth {DEPTH} {collect_file} 1>/dev/null 2>/dev/null"

	collect_cmd: str = f"{COLLECTOR} {mlir_file} --depth {DEPTH} {collect_file} 1>/dev/null 2>/dev/null"

	if os.system(program_generate_cmd):
		return MLIRodTestResult.GENERATION_FAILED

	def run_helper(cmd_list: List[str], failed_enums: List[MLIRodTestResult]) -> MLIRodTestResult:
		if os.system(cmd_list[0]):
			return failed_enums[0]
		else:
			if os.system(cmd_list[1]):
				return failed_enums[1]
			if os.system(cmd_list[2]):
				return failed_enums[2]
		return MLIRodTestResult.SUCCESS

	def remove(file):
		if os.path.exists(file):
			os.remove(file)

	cmd_list: List[str] = [replace_data_edge_cmd, replace_data_edge_opt_cmd, repalce_data_edge_collect_cmd]
	res_list: List[MLIRodTestResult] = [MLIRodTestResult.REPLACE_DATA_EDGE_FAILED, MLIRodTestResult.REPLACE_DATA_EDGE_PARSE_FAILED, MLIRodTestResult.REPALCE_DATA_EDGE_COLLECT_FAILED]
	res: MLIRodTestResult = run_helper(cmd_list, res_list)
	if res != MLIRodTestResult.SUCCESS:
		return res

	remove(collect_file)
	remove(repalce_data_mlir_file)
	cmd_list: List[str] = [replace_control_edge_cmd, replace_control_edge_opt_cmd, repalce_control_edge_collect_cmd]
	res_list: List[MLIRodTestResult] = [MLIRodTestResult.REPALCE_CONTROL_EDGE_FAILED, MLIRodTestResult.REPLACE_CONTROL_EDGE_PARSE_FAILED, MLIRodTestResult.REPLACE_CONTROL_EDGE_COLLECT_FAILED]
	res: MLIRodTestResult = run_helper(cmd_list, res_list)
	if res != MLIRodTestResult.SUCCESS:
		return res
	
	remove(collect_file)
	remove(replace_control_mlir_file)
	cmd_list: List[str] = [delete_node_cmd, delete_node_opt_cmd, delete_node_collect_cmd]
	res_list: List[MLIRodTestResult] = [MLIRodTestResult.DELETE_NODE_FAILED, MLIRodTestResult.DELETE_NODE_PARSE_FAILED, MLIRodTestResult.DELETE_NODE_COLLECT_FAILED]
	res: MLIRodTestResult = run_helper(cmd_list, res_list)
	if res != MLIRodTestResult.SUCCESS:
		return res
	
	remove(collect_file)
	remove(delete_node_mlir_file)
	cmd_list: List[str] = [create_node_cmd, create_node_opt_cmd, create_node_collect_cmd]
	res_list: List[MLIRodTestResult] = [MLIRodTestResult.CREATE_NODE_FAILED, MLIRodTestResult.CREATE_NODE_PARSE_FAILED, MLIRodTestResult.CREATE_NODE_COLLECT_FAILED]
	res: MLIRodTestResult = run_helper(cmd_list, res_list)
	if res != MLIRodTestResult.SUCCESS:
		return res

	remove(collect_file)
	remove(create_node_mlir_file)
	if os.system(collect_cmd):
		return MLIRodTestResult.COLLECT_FAILED
	
	remove(mlir_file)
	remove(repalce_data_mlir_file)
	remove(replace_control_mlir_file)
	remove(delete_node_mlir_file)
	remove(create_node_mlir_file)
	remove(collect_file)
	return MLIRodTestResult.SUCCESS
	
def test_mlirsmith_single() -> MLIRSmithTestResult:
	mlir_file: str = f"{secrets.token_hex(8)}.mlir"
	program_generate_cmd: str = f"{MLIRSMITH} 2>{mlir_file}"
	mlir_opt_cmd: str = f"{MLIR_OPT} {mlir_file} > /dev/null"
	
	if os.system(program_generate_cmd):
		return MLIRSmithTestResult.GENERATION_FAILED
	if os.system(mlir_opt_cmd):
		return MLIRSmithTestResult.PARSE_FAILED
	os.remove(mlir_file)
	return MLIRSmithTestResult.SUCCESS

def test_mlirsmith_main():
	failed_generate_time = 0
	failed_parse_time = 0
	total_test_time = 10

	cpu_num: Optional[int] = os.cpu_count()
	if not cpu_num:
		cpu_num = 4

	with DaemonThreadPoolExecutor(max_workers=cpu_num,) as executor, tqdm(total=total_test_time) as pbar:
		futures = [
			executor.submit(test_mlirsmith_single, ) for t in range(total_test_time)
		]

		for future in as_completed(futures):
			res: MLIRSmithTestResult = future.result()
			if res == MLIRSmithTestResult.GENERATION_FAILED:
				failed_generate_time += 1
			elif res == MLIRSmithTestResult.PARSE_FAILED:
				failed_parse_time += 1
			pbar.update(1)
	
	print(f'{"".ljust(50, "*")} test MLIRSmith {"".ljust(50, "*")}')
	print(f"Test over!")
	print(f"Generation success/failed: {total_test_time}/{failed_generate_time}")
	print(f"Parse      success/failed: {total_test_time-failed_generate_time}/{failed_parse_time}")

def test_mlirod_main():
	total_test_time = 10

	cpu_num: Optional[int] = os.cpu_count()
	if not cpu_num:
		cpu_num = 4

	failed_cnt: dict[MLIRodTestResult, int] = {er: 0 for er in list(MLIRodTestResult)}

	with DaemonThreadPoolExecutor(max_workers=int(cpu_num/2),) as executor, tqdm(total=total_test_time) as pbar:
		futures = [
			executor.submit(test_mlirod_single, ) for t in range(total_test_time)
		]

		for future in as_completed(futures):
			res: MLIRodTestResult = future.result()
			failed_cnt[res] = failed_cnt.setdefault(res, 0) + 1
			pbar.update(1)
	
	print(f'{"".ljust(50, "*")} test MLIRod {"".ljust(50, "*")}')
	print(f"Test over!")
	for res in failed_cnt:
		print(f"{res.value.lower().replace('_failed', '').ljust(50)} {'failed/total' if res != MLIRodTestResult.SUCCESS else 'success/total'}: {failed_cnt[res]}/{total_test_time}")

def test_desil_main():
	total_test_time = 10

	cpu_num: Optional[int] = os.cpu_count()
	if not cpu_num:
		cpu_num = 4

	failed_cnt: dict[DESILTestResult, int] = {er: 0 for er in list(DESILTestResult)}

	with DaemonThreadPoolExecutor(max_workers=int(cpu_num/2),) as executor, tqdm(total=total_test_time) as pbar:
		futures = [
			executor.submit(test_desil_single, ) for t in range(total_test_time)
		]

		for future in as_completed(futures):
			res: DESILTestResult = future.result()
			failed_cnt[res] = failed_cnt.setdefault(res, 0) + 1
			pbar.update(1)
	
	print(f'{"".ljust(50, "*")} test DESIL {"".ljust(50, "*")}')
	print(f"Test over!")
	for res in failed_cnt:
		print(f"{res.value.lower().replace('_failed', '').ljust(50)} {'failed/total' if res != DESILTestResult.SUCCESS else 'success/total'}: {failed_cnt[res]}/{total_test_time}")
	pass

if __name__ == "__main__":
	test_mlirsmith_main()
	test_mlirod_main()
	test_desil_main()
	# os.system("/data/llvm-project/build/bin/CreateNode 7468cdad70c16d81.mlir a.mlir 1>/dev/null 2>/dev/null")
