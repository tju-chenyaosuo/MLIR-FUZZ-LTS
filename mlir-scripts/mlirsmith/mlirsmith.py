import os
import re
import random
import argparse
import time
from enum import Enum


DIALECT_COV = set()
OPERATION_COV = set()


def execmd(cmd):
    import os
    print('[execmd] ' + cmd)
    pipe = os.popen(cmd)
    reval = pipe.read()
    pipe.close()
    return reval


def random_file_prefix():
    # pattern: "/tmp/tmp.WeWmILaF2A"
    cmd = "mktemp -p " + args.testcase_base
    random_name = execmd(cmd)
    random_name = random_name[:-1] if random_name.endswith('\n') else random_name
    res_name = random_name.split('/')[-1]
    cmd = "rm -f " + random_name
    os.system(cmd)
    return res_name


def execmd_limit_time(cmd, time_limit):
    import time
    start = time.time()
    execmd("timeout " + str(time_limit) + " " + cmd)
    end = time.time()
    return (end - start) >= time_limit


def get_file_content(file_path):
    f = open(file_path, 'r')
    content = f.read()
    f.close()
    return content


def get_file_lines(path):
    c = get_file_content(path)
    if c == '':
        return ''
    if c[-1] == '\n':
        return c[:-1].split('\n')
    else:
        return c.split('\n')


def put_file_content(path, content):
    f = open(path, 'a+')
    f.write(content)
    f.flush()
    f.close()


class Coverage:
    COVERAGE = set()

    def __init__(self, mlir_path):
        self.mlir_path = mlir_path
        pass

    def collect_cov(self, depth):
        cmd = ' '.join([args.collector, self.mlir_path, "--depth", str(depth), args.cov_file])
        print(f'[Coverage.collect_cov] {cmd}')
        os.system(cmd)
        cov = set()
        if os.path.exists(args.cov_file):
            cov = set(get_file_lines(args.cov_file))
        os.system(f"rm -f {args.cov_file}")
        return cov

    def collect_and_compare(self):
        lines_before = len(Coverage.COVERAGE)
        cov = self.collect_cov(args.cov_depth)
        Coverage.COVERAGE |= cov
        lines_after = len(Coverage.COVERAGE)
        assert lines_after >= lines_before
        update = lines_before < lines_after
        print("[Coverage.collect_and_compare] coverage length=" + str(len(Coverage.COVERAGE)))
        return update


class TestState(Enum):
    # config_timeout = 0
    # config_crash = 1
    # config_success = 2
    config_success = 0
    config_failed = 1
    gen_success = 3
    gen_timeout = 4
    gen_crash = 5
    test_success = 6
    test_timeout = 7
    test_crash = 8


def gen_one_mlir(mlir_file):
    gen_cmd = f"{args.mlirsmith} 2>{mlir_file}"
    timeout = execmd_limit_time(gen_cmd, args.timeout)
    if timeout:
        return TestState.gen_timeout
    f = open(mlir_file)
    f_lines = f.readlines()
    f.close()
    if len(f_lines) < 100:
        return TestState.gen_crash
    return TestState.gen_success


def test_each_mlir(mlir_file, opt, out_mlir, report_dir):
    crash_log = report_dir + os.sep + os.path.basename(out_mlir) + '.crash.txt'
    cmd = ' '.join([args.mliropt, opt, mlir_file, '1>/dev/null', '2>' + crash_log])
    timeout = execmd_limit_time(cmd, 10)
    if timeout:
        timeout_log = report_dir + os.sep + os.path.basename(out_mlir) + '.timeout.txt'
        put_file_content(timeout_log, cmd)
        os.system('rm -f ' + crash_log)
        return TestState.test_timeout
    log_content = get_file_content(crash_log)
    crash = len(log_content) > 10
    if crash:
        put_file_content(crash_log, '\n' + cmd + '\n')
        return TestState.test_crash
    else:
        os.system('rm -f ' + crash_log)
    return TestState.test_success


def test_each_option(mlir_file, options):
    if args.optnum < 100:
        return options
    else:
        # create temp directory
        temp_dir = execmd('mktemp -d')
        temp_dir = temp_dir[:-1] if temp_dir.endswith('\n') else temp_dir

        # for each option, we do test
        valid_options = []
        for idx in range(len(options)):
            out_mlir = temp_dir + os.sep + str(idx) + '.mlir'
            report_dir = temp_dir + os.sep + 'report'
            os.system('mkdir -p ' + report_dir)
            test_state = test_each_mlir(mlir_file, options[idx], out_mlir, report_dir)
            if test_state == TestState.test_success:
                valid_options.append(options[idx])

        os.system('rm -rf ' + temp_dir)
        return valid_options


def get_opts():
    assert os.path.exists(args.optfile)
    f = open(args.optfile)
    f_lines = f.readlines()
    f.close()
    options = [_[:-1] if _.endswith('\n') else _ for _ in f_lines]
    return options


def get_optimization_sequences(options):
    # option_number = random.randint(5, 10)
    # return [' '.join([options[random.randint(0, len(options) - 1)]
    #                   for _ in range(option_number)])
    #         for __ in range(args.optnum)]
    return options


def main():
    global OPERATION_COV, DIALECT_COV

    cur_hour_num = 0
    init_start_time = time.time()

    options = get_opts()

    init_end_time = time.time()
    test_time = init_end_time - init_start_time

    while test_time <= args.fuzz_time:
        start_time = time.time()

        mf = args.testcase_base + os.sep + random_file_prefix() + '.mlir'
        res = gen_one_mlir(mf)
        while res != TestState.gen_success:
            res = gen_one_mlir(mf)
        # valid_options = test_each_option(mf, options)
        # valid_options = options
        # if len(valid_options) == 0:
        #     end_time = time.time()
        #     test_time += end_time - start_time
        #     continue
        # optimization_sequences = get_optimization_sequences(valid_options)
        # print('[test_mlir] valid option length: ' + str(len(valid_options)))
        # print('[test_mlir] valid option ' + ', '.join(valid_options))

        for idx in range(len(options)):
            out_dir = args.out_base + os.sep + os.path.basename(mf) + '.output'
            report_dir = out_dir + os.sep + 'report'
            os.system('mkdir -p ' + out_dir)
            os.system('mkdir -p ' + report_dir)
            out_mlir = out_dir + os.sep + str(idx) + '.mlir'
            state = test_each_mlir(mf, options[idx], out_mlir, report_dir)
            # if state != TestState.test_success:
            os.system('rm -f ' + out_mlir)

        end_time = time.time()
        test_time += end_time - start_time

        if args.collect_dialect_coverage:
            c = Coverage(mf)
            operations = list(c.collect_cov(0))
            operations = {_.split('(')[0] for _ in operations}
            dialects = {_.split('.')[0] for _ in operations}
            OPERATION_COV = OPERATION_COV | operations
            DIALECT_COV = DIALECT_COV | dialects

        if cur_hour_num != int(test_time/args.collect_interval):
            cur_time = time.time()
            put_file_content(args.time_stamp_file, str(cur_time) + '\n')
            if args.collect_dialect_coverage:
                cur_hour_num = int(test_time/args.collect_interval)
                d_cov_file = f'{args.dialect_coverage_dir}/{cur_hour_num}.txt'
                o_cov_file = f'{args.operation_coverage_dir}/{cur_hour_num}.txt'
                put_file_content(d_cov_file, '\n'.join(sorted(list(DIALECT_COV))))
                put_file_content(o_cov_file, '\n'.join(sorted(list(OPERATION_COV))))
            if args.collect_code_coverage:
                cur_hour_num = int(test_time / args.collect_interval)
                ori_dir = os.getcwd()
                gcov_dir = args.gcov_dir + os.sep + str(cur_hour_num) + os.sep
                os.system(f'mkdir -p {gcov_dir}')
                os.system(f'cp {args.gcov_script} {gcov_dir}')
                os.chdir(gcov_dir)
                os.system(f'python3 {args.gcov_script} --project {args.gcov_project}')
                os.chdir(ori_dir)

    cur_time = time.time()
    put_file_content(args.time_stamp_file, str(cur_time) + '\n')
    if args.collect_dialect_coverage:
        d_cov_file = f'{args.dialect_coverage_dir}/lasttime.txt'
        o_cov_file = f'{args.operation_coverage_dir}/lasttime.txt'
        put_file_content(d_cov_file, '\n'.join(sorted(list(DIALECT_COV))))
        put_file_content(o_cov_file, '\n'.join(sorted(list(OPERATION_COV))))
    if args.collect_code_coverage:
        ori_dir = os.getcwd()
        gcov_dir = args.gcov_dir + os.sep + 'lasttime' + os.sep
        os.system(f'mkdir -p {gcov_dir}')
        os.system(f'cp {args.gcov_script} {gcov_dir}')
        os.chdir(gcov_dir)
        os.system(f'python3 {args.gcov_script} --project {args.gcov_project}')
        os.chdir(ori_dir)


"""
Usage:
For Test:
python mlirsmith.py \
--mlirsmith /data3/2025.04.07.liangjingjing/bin/llvm-13c6abfac84fca4bc55c0721d1853ce86a385678/build/bin/mlirsmith \
--testcase_base testcase \
--mliropt /data3/2025.04.07.liangjingjing/bin/llvm-13c6abfac84fca4bc55c0721d1853ce86a385678/build/bin/mlir-opt \
--optfile opt.txt \
--collect_interval 200 \
--fuzz_time 600 \
--collector /data3/2025.04.07.liangjingjing/bin/llvm-13c6abfac84fca4bc55c0721d1853ce86a385678/build/bin/CollectCoverage \
--out_base report > log

Run experiment (do not collect dialect/operation coverage):
python mlirsmith.py \
--mlirsmith /data3/2025.04.07.liangjingjing/bin/llvm-13c6abfac84fca4bc55c0721d1853ce86a385678/build/bin/mlirsmith \
--testcase_base testcase \
--mliropt /data3/2025.04.07.liangjingjing/bin/llvm-13c6abfac84fca4bc55c0721d1853ce86a385678/build/bin/mlir-opt \
--optfile opt.txt \
--collect_dialect_coverage False \
--collector /data3/2025.04.07.liangjingjing/bin/llvm-13c6abfac84fca4bc55c0721d1853ce86a385678/build/bin/CollectCoverage \
--out_base report > log

Run experiment (Collect dialect/operation coverage):
python mlirsmith.py \
--mlirsmith /data3/2025.04.07.liangjingjing/bin/llvm-13c6abfac84fca4bc55c0721d1853ce86a385678/build/bin/mlirsmith \
--testcase_base testcase \
--mliropt /data3/2025.04.07.liangjingjing/bin/llvm-13c6abfac84fca4bc55c0721d1853ce86a385678/build/bin/mlir-opt \
--optfile opt.txt \
--collector /data3/2025.04.07.liangjingjing/bin/llvm-13c6abfac84fca4bc55c0721d1853ce86a385678/build/bin/CollectCoverage \
--out_base report > log

Run experiment (collect dialect/operation coverage as well as code coverage):
python mlirsmith.py \
--mlirsmith /data3/2025.04.07.liangjingjing/bin/llvm-13c6abfac84fca4bc55c0721d1853ce86a385678/build/bin/mlirsmith \
--testcase_base testcase \
--mliropt /data3/2025.04.07.liangjingjing/bin/llvm-13c6abfac84fca4bc55c0721d1853ce86a385678/build/bin/mlir-opt \
--optfile opt.txt \
--collector /data3/2025.04.07.liangjingjing/bin/llvm-13c6abfac84fca4bc55c0721d1853ce86a385678/build/bin/CollectCoverage \
--collect_code_coverage True \
--gcov_project /data3/2025.04.07.liangjingjing/bin/llvm-13c6abfac84fca4bc55c0721d1853ce86a385678 \
--out_base report > log
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=bool, help="Clean", default=False)
    parser.add_argument("--fuzz_time", type=int, help="Fuzz time", default=24 * 60 * 60)
    parser.add_argument("--mlirsmith", type=str, help="path of mlirsmith.")
    parser.add_argument("--testcase_base", type=str, help="path of generated testcase directory.")
    parser.add_argument("--timeout", type=int, default=10, help="path of generated testcase directory.")
    parser.add_argument("--mliropt", type=str, help="path of mlir-opt.")
    parser.add_argument("--optfile", type=str, help="path of option file.")
    parser.add_argument("--optnum", type=int, help="number of options.")
    parser.add_argument("--out_base", type=str, help="path of output.")
    # arguments in dilaect/operation coverage
    parser.add_argument("--collect_dialect_coverage", type=bool,
                        help="Whether collect dialect and operation coverage per-hour.", default=True)
    parser.add_argument("--dialect_coverage_dir", type=str, help="Dialect coverage directory.", default="dialect_cov")
    parser.add_argument("--operation_coverage_dir", type=str, help="Operation coverage directory.",
                        default="operation_cov")
    parser.add_argument("--collect_interval", type=int,
                        help="The time interval for dialet/operation coverage collection.", default=60 * 60)
    # arguments in hour timestamp recorder.
    parser.add_argument("--time_stamp_file", type=str,
                        help="Files that record time stamp for each ``collect_interval''", default="time.interval.txt")
    parser.add_argument("--collector", type=str, help="Coverage collector", default="")
    parser.add_argument("--cov_file", type=str, help="Coverage file.", default="coverage.txt")
    parser.add_argument("--cov_depth", type=int, help="Coverage depth", default=2)
    # arguments in code coverage collection
    parser.add_argument("--collect_code_coverage", type=bool, help="Whether collect code coverage per-hour.", default=False)
    parser.add_argument("--gcov_dir", type=str, help="folder to collect all of the gcov.", default="collectcov/gcovs")
    parser.add_argument("--gcov_script", type=str, help="coverage collect script", default="collectcov.py")
    parser.add_argument("--gcov_project", type=str, help="project to be collected", default="")
    args = parser.parse_args()

    if args.clean:
        os.system(f'rm -rf {args.testcase_base} {args.out_dir} {args.dialect_coverage_dir}')
        os.system(f"rm -rf {args.time_stamp_file} {args.operation_coverage_dir} {args.gcov_dir}")

    os.system(f"mkdir -p {args.testcase_base}")
    os.system(f"mkdir -p {args.out_base}")
    os.system(f"mkdir -p {args.dialect_coverage_dir}")
    os.system(f"mkdir -p {args.operation_coverage_dir}")
    main()
