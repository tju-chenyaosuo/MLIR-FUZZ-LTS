import re
import os
import argparse
import random
import time


def get_file_content(file_path: str):
    f = open(file_path, 'r')
    content = f.read()
    f.close()
    return content


def get_file_lines(file_path):
    f = open(file_path)
    lines = f.readlines()
    return lines


def put_file_content(path, content):
    f = open(path, 'a+')
    f.write(content)
    f.flush()
    f.close()


def diff(file1, file2):
    return execmd(f"diff {file1} {file2}")


def file_cmp(path1, path2):
    content1 = get_file_content(path1)
    content2 = get_file_content(path2)
    tranc1 = content1[0: min(len(content1), 10000)]
    tranc2 = content2[0: min(len(content2), 10000)]
    if tranc1 == tranc2:
        return True
    return False


def execmd(cmd):
    import os
    print('[execmd] ' + cmd)
    try:
        pipe = os.popen(cmd)
        reval = pipe.read()
        pipe.close()
        return reval
    except BlockingIOError:
        print("[execmd] trigger BlockingIOError")
        return "None"


def execmd_limit_time(cmd, time_limit):
    import time
    start = time.time()
    execmd("timeout " + str(time_limit) + " " + cmd)
    end = time.time()
    return (end - start) >= time_limit


def not_mlir(file):
    cmd = f"file -r {file}"
    ret = execmd(cmd)
    print(ret)
    return ": ASCII text" not in ret


def is_mlir(file):
    file_content = get_file_content(file)
    return file_content.startswith("module") or file_content.startswith("#map =") or file_content.startswith("#set =")


def get_line_number(file):
    return len(get_file_lines(file))


def random_file_prefix():
    # pattern: "/tmp/tmp.WeWmILaF2A"
    cmd = "mktemp -p " + args.seed_dir
    random_name = execmd(cmd)
    random_name = random_name[:-1] if random_name.endswith('\n') else random_name
    res_name = random_name.split('/')[-1]
    cmd = "rm -f " + random_name
    os.system(cmd)
    return res_name


def get_optimization(file: str):
    """
    Read the optimization file, and return the dict: { dialect_name: dialect_opts }

    Parameters:
        file: str, optpasses.txt

    Returns:
        dict, { dialect_name: passes }
    """
    dialect_opts = {}
    file = get_file_content(file).split("---------------------------------")
    for item in file:
        name = item.split(":")[0].strip()
        options = item.split(":")[1].strip()
        opts = []
        for opt in options.split("\n"):
            opts.append(opt)
        dialect_opts[name] = opts
    return dialect_opts


def recommend_opts(dialects: dict):
    """
    Recommend optimizations for given dialect collection.

    Parameters:
        dialects: dict, { dialect_name: appear_time }, in which appear_time is to expand this method.

    Return:
        return a random pass sequence with 0-10 length matches the given dialects.
    """
    possible_opts = []
    for dialect in dialects:
        possible_opts.extend(DIALECT_OPTS.setdefault(dialect, []))
    possible_opts.extend(DIALECT_OPTS.setdefault('general', []))
    print(f"[recommend_opts] dialects={dialects}")
    print(f"[recommend_opts] possible_opts={possible_opts}")
    if len(possible_opts) == 0:
        print(f"[recommend_opts] warning: no possible optimization!")
    ret = [possible_opts[random.randint(0, len(possible_opts)-1)] for _ in range(args.opt_num)]
    return ret


def has_crash_message(path: str):
    """
    Check the crash message in the given file.

    Parameters:
        path: str, the path of given file.

    Return:
        True if any crash message in the given file.
    """
    txt = get_file_content(path)
    return "PLEASE submit" in txt


class Seed:
    def __init__(self, path):
        self.path = path
        self.cnt = 0
        self.tested = False
        self.deprecated = False

    def can_discard(self):
        return self.cnt >= args.max_selected or self.deprecated

    def not_mlir(self):
        return not_mlir(self.path)

    def __str__(self):
        return f'seed: [path={self.path}, mutation_cnt={self.cnt}, is_tested={self.tested}]'


class MLIRSmithSeedFactory:
    def __init__(self):
        self.conf_success = "conf_success"
        self.conf_timeout = "conf_timeout"
        self.conf_crash = "conf_crash"
        self.mlir_success = "mlir_success"
        self.mlir_timeout = "mlir_timeout"
        self.mlir_crash = "mlir_crash"

    def gen_conf(self):
        """
        Generate MLIRSmith configuration

        Return:
            state
            { self.conf_timeout, self.conf_crash, self.conf_success }
        """
        # generate new configuration
        cmd = f"{args.mlirsmith} -emit=config >{args.conf}"
        timeout = execmd_limit_time(cmd, args.timeout)
        if timeout:
            return self.conf_timeout
        conf_lines_len = get_line_number(args.conf)
        if conf_lines_len < 100:
            return self.conf_crash
        return self.conf_success

    def gen_mlir(self, mlir_path: str):
        """
        Generate mlir program

        Parameters:
            mlir_path: str, path of mlir program to be generated

        Return:
            state
            { self.mlir_timeout, self.mlir_crash, self.mlir_success }
        """
        cmd = f"{args.mlirsmith} -emit=mlir-affine -dfile={args.conf} {args.mlir_template} 2>{mlir_path}"
        timeout = execmd_limit_time(cmd, args.timeout)
        if timeout:
            return self.mlir_timeout
        mlir_file_len = get_line_number(mlir_path)
        if mlir_file_len < 100:
            return self.mlir_crash
        return self.mlir_success

    def gen_seed(self):
        """
        Generate a seed

        Return:
            MLIR program seed
        """
        # generate a new conf
        res = self.gen_conf()
        while res != self.conf_success:
            res = self.gen_conf()

        # gen random mlir file
        mlir_name = random_file_prefix() + '.mlir'
        mlir_path = args.seed_dir + os.sep + mlir_name

        # gen mlir file
        res = self.gen_mlir(mlir_path)
        while res != self.mlir_success:
            res = self.gen_mlir(mlir_path)
        s = Seed(mlir_path)
        return s


class ExistingSeedFactory:
    def __init__(self):
        self.existing_mlirs = os.listdir(args.existing_seed_dir)
        self.existing_mlirs = [args.existing_seed_dir + os.sep + _ for _ in self.existing_mlirs]
        self.unselected_idx = [_ for _ in range(len(self.existing_mlirs))]

    def gen_seed(self):
        """
        Generate a seed

        Return:
            MLIR program seed
        """
        tmp = random.randint(0, len(self.unselected_idx) - 1)
        idx = self.unselected_idx[tmp]
        del self.unselected_idx[tmp]

        rname = random_file_prefix() + '.mlir'
        seed_path = args.seed_dir + os.sep + rname
        cmd = f'cp {self.existing_mlirs[idx]} {seed_path}'
        os.system(cmd)
        s = Seed(seed_path)

        return s


class SeedPool:
    def __init__(self):
        self.pool = []

    def init(self):
        print(f'[SeedPool.init] the pool initializer is: {args.seed}')
        factory = None
        if args.seed == "mlirsmith":
            factory = MLIRSmithSeedFactory()
        if args.seed == "existing":
            factory = ExistingSeedFactory()

        assert factory is not None
        while len(self.pool) < args.init_size:
            self.add(factory.gen_seed())
        print('[SeedPool.init] Seed pool initialization is over!')
        print('[SeedPool.init] The seed information is:')
        for s in self.pool:
            print(s)

    def select(self):
        idx = random.randint(0, len(self.pool) - 1)
        selected = self.pool[idx]
        while selected.not_mlir():
            del self.pool[idx]
            idx = random.randint(0, len(self.pool) - 1)
            selected = self.pool[-1]
        del self.pool[idx]
        return selected

    def add(self, s: Seed):
        if not s.not_mlir():
            print('[SeedPool.add] new seed: ' + str(s))
            self.pool.append(s)

    def empty(self):
        return len(self.pool) == 0

    def clean(self):
        idx = 0
        while idx < len(self.pool):
            if self.pool[idx].can_discard():
                del self.pool[idx]
            else:
                idx += 1

    def __str__(self):
        res = f"pool size: {len(self.pool)}\n"
        for seed in self.pool:
            res += f'{seed}\n'
        return res


def collect_dialects(mlir_path: str):
    """
    Collect the dialects in mlir_path.

    Parameters:
        mlir_path: str, the path of given mlir test program.

    Return:
        dict() = { dialect_name: cnt }
    """
    dialects = {}
    file_lines = get_file_lines(mlir_path)
    for line in file_lines:
        pattern = re.compile(r'[a-zA-Z][a-zA-Z0-9]+\.[a-zA-Z0-9]+')
        m = pattern.findall(line)
        if m is None or len(m) == 0:
            continue
        m[0] = m[0].split(".")[0]
        dialects[m[0]] = dialects.setdefault(m[0], 0) + 1
    return dialects


class DynamicPassSelector:
    # the data of the lowering sequence of an operation eg: {"arith.addi" : ["--convert-arith-to-llvm"]}
    lower_map = {}

    """
    the data of the passes that have prerequisite passes
    eg: 
    {"--one-shot-bufferize" : 
    ["--one-shot-bufferize="dialect-filter=tensor,bufferization"", 
    "--func-bufferize", 
    "--one-shot-bufferize="dialect-filter=linalg""]}
    for "--reconcile-unrealized-casts", is {"--reconcile-unrealized-casts" : ["all"]},
    and will be special judged in the prerequisite considering algorithm
    """
    prerequisite_passes = {}

    """
    the data of the regular expressions for different operations, 
    including same operation that requires different lowering path eg : {"scf.while .+ tensor" : "scf.while2"}.
    These regular expressions are used to recognize specific operations with specific type operands.
    For exmaple, ''scf.while .+ tensor'' means that scf.while operation with tensor operands.
    """
    regexp = {}

    # special judges for continuous passes
    @staticmethod
    def special_judge(_pass: str):
        #if _pass == "--one-shot-bufferize=\"dialect-filter=linalg\"":
        #    return " --convert-linalg-to-loops"
        #if _pass == "--expand-strided-metadata":
        #    return " --func-bufferize"
        #if _pass == "--one-shot-bufferize=\"dialect-filter=arith\"":
        #    return " --convert-arith-to-llvm"
        if _pass == "--convert-vector-to-scf":
            return " --convert-vector-to-llvm"
        return "not special case"

    @staticmethod
    def init_dics(lowermap_file, prerequisite_passes_file, regexp_file):
        """
        TODO: ASK author write this

        :param lowermap_file: lowermap.txt
        :param prerequisite_passes_file: prerequisitepasses.txt
        :param regexp_file: regexp.txt
        :return: None
        """

        two_file = [lowermap_file, prerequisite_passes_file, regexp_file]
        for file in two_file:
            lines = get_file_lines(file)
            for line in lines:
                key = line.split(":")[0].strip()
                values = []
                if file == regexp_file:
                    values = line.split(":")[1].strip().split(" ")
                else:
                    values = line.split(":")[1].strip().split("--")
                    values = ["--" + value.strip() for value in values if value != "" ]
                    values = ["all"] if values[0] == "--all" else values
                passlist = []
                passes = []

                to_continue = 0
                for value in values:
                    if to_continue:
                        to_continue = 0
                        continue

                    # special judges for continuous passes
                    judgeres = DynamicPassSelector.special_judge(value)  # why not directly use the replace method.
                    if judgeres != "not special case":
                        passes.append(value + judgeres)
                        to_continue = 1
                        continue

                    passes.append(value)
                if file == lowermap_file:
                    DynamicPassSelector.lower_map[key] = passes
                elif file == regexp_file:
                    DynamicPassSelector.regexp[key] = passes
                else:
                    if key in DynamicPassSelector.prerequisite_passes:
                        for _pass in passes:
                            DynamicPassSelector.prerequisite_passes[key].append(_pass)
                    else:
                        DynamicPassSelector.prerequisite_passes[key] = passes
        return

    def __init__(self, mlir, last_failed_passes):
        self.mlir_path = mlir.mlir_path      # the path of mlir program
        self.opt_recorder = mlir.opt_recorder  # the path of optimization record of this mlir program
        self.possible_passes = {}          # ???
        self.operations = {}               # the operations in MLIR file.
        self.last_failed_passes = last_failed_passes  # last failed lowering passes.
        self.lowering_finished = "lowering_finished"
        self.selector_error = "selector_error"
        return

    def get_operation(self):
        file_lines = get_file_lines(self.mlir_path)
        for line in file_lines:
            for exp in DynamicPassSelector.regexp:
                pattern = re.compile(rf'{exp}')
                m = pattern.findall(line)
                if m is None or len(m) == 0:
                    continue
                if DynamicPassSelector.regexp[exp][0] == "all":
                    m = pattern.findall(line)
                else:
                    m = DynamicPassSelector.regexp[exp]

                m[0] = m[0].strip()
                self.operations[m[0]] = self.operations.setdefault(m[0], 1)

    def collect_possible_passes(self):
        """
        Collect all passes
        :return:
        """
        for operation in self.operations:
            if operation not in DynamicPassSelector.lower_map:
                continue
            passes = DynamicPassSelector.lower_map[operation]
            first = 1

            for _pass in passes:
                if first == 1:
                    first = 0
                    self.possible_passes[_pass] = self.possible_passes.setdefault(_pass, 0)
                    continue
                else:
                    self.possible_passes[_pass] = self.possible_passes.setdefault(_pass, 0)
                    self.possible_passes[_pass] = self.possible_passes[_pass] + 1

    def add_prerequisite(self):
        for _pass, prerequisites in self.prerequisite_passes.items():
            if _pass not in self.possible_passes:
                continue
            if _pass == "--reconcile-unrealized-casts":  # why?
                self.possible_passes[_pass] = self.possible_passes[_pass] + len(self.possible_passes) - 1
                continue

            for prerequisite in prerequisites:
                if prerequisite not in self.possible_passes:
                    continue
                self.possible_passes[_pass] = self.possible_passes[_pass] + 1

        for failed_pass in self.last_failed_passes:
            if failed_pass not in self.possible_passes:
                continue
            self.possible_passes[failed_pass] = self.possible_passes[failed_pass] + 1

    def get_lowering_pass(self):
        """
        :return:
        """
        passes = []
        for _pass, in_num in self.possible_passes.items():
            if in_num == 0:
                passes.append(_pass)

        if len(self.possible_passes) == 0:
            return self.lowering_finished
        if len(passes) == 0:
            return self.selector_error
        return random.sample(passes, 1)[0]

    def main(self):
        self.get_operation()
        print(f"[DynamicPassSelector.main] {self.operations}")
        self.collect_possible_passes()
        print(f"[DynamicPassSelector.main] {self.possible_passes}")
        self.add_prerequisite()
        print(f"[DynamicPassSelector.main] {self.possible_passes}")
        return self.get_lowering_pass()


class ExecuteResult:
    def __init__(self, path, result):
        self.path = path
        self.result = result


class MLIRLoweringCase:
    """
    Represents the lowering case of the original mlir program.
    """
    def __init__(self, test_dir: str, case_id: int, case_name: str):
        self.case_id = case_id
        self.mlir_name = f"{case_id}.{case_name}"
        self.mlir_path = f"{test_dir}/mlir/{self.mlir_name}.mlir"
        self.opt_recorder = f"{test_dir}/opt/{case_id}.{case_name}.txt"


class TestBatch:
    def __init__(self, mlir_path: str):
        """
        Initialize a set batch.

        Parameters:
            mlir_path: str, which is the path of MLIR test program from seed pool.
        """

        self.checksum_mlir = None
        self.fix_ub_mlir = None

        self.case_name = mlir_path.split(".")[-2]
        self.test_dir = f"{args.test_dir}/{self.case_name}"
        self.mlir_dir = f"{self.test_dir}/mlir"
        self.opt_dir = f"{self.test_dir}/opt"
        self.crash_log_dir = f"{self.test_dir}/crashlog"
        self.executable_dir = f"{self.test_dir}/out"
        self.tmp_dir = f"{self.test_dir}/tmp"
        self.result_dir = f"{self.test_dir}/result"

        # define some status
        self.fixub_timeout = "fixub_timeout"
        self.fixub_crash = "fixub_crash"
        self.fixub_failed = "fixub_failed"
        self.fixub_success = "fixub_success"
        self.checksum_timeout = "checksum_timeout"
        self.checksum_crash = "checksum_crash"
        self.checksum_failed = "checksum_failed"
        self.checksum_success = "checksum_success"
        self.compile_timeout = "compile_timeout"
        self.compile_crash = "compile_crash"
        self.compile_failed = "compile_failed"
        self.compile_success = "compile_success"
        self.lowering_timeout = "lowering_timeout"
        self.lowering_crash = "lowering_crash"
        self.lowering_selector_error = "lowering_selector_error"
        self.lowering_failed = "lowering_failed"
        self.lowering_success = "lowering_success"
        self.translate_timeout = "translate_timeout"
        self.translate_crash = "translate_crash"
        self.translate_success = "translate_success"
        self.sed_timeout = "sed_timeout"
        self.sed_crash = "sed_crash"
        self.sed_success = "sed_success"
        self.clang_timeout = "clang_timeout"
        self.clang_crash = "clang_crash"
        self.clang_success = "clang_success"
        self.execute_timeout = "execute_timeout"
        self.execute_crash = "execute_crash"
        self.execute_success = "execute_success"
        self.success = "success"
        self.wrongcode = "wrongcode"
        self.re_lowering_success = "re_lowering_success"

        # define some exception directories
        self.report_dir = f"{self.test_dir}/report"

        # define the global directories
        self.global_fixub_crash_dir = f"{args.report_dir}/fixub_crash"
        self.global_fixub_timeout_dir = f"{args.report_dir}/fixub_timeout"
        self.global_fixub_failed_dir = f"{args.report_dir}/fixub_failed"
        self.global_checksum_crash_dir = f"{args.report_dir}/checksum_crash"
        self.global_checksum_timeout_dir = f"{args.report_dir}/checksum_timeout"
        self.global_checksum_failed_dir = f"{args.report_dir}/checksum_failed"
        self.global_compile_crash_dir = f"{args.report_dir}/compile_crash"
        self.global_compile_timeout_dir = f"{args.report_dir}/compile_timeout"
        self.global_lowering_selector_error_dir = f"{args.report_dir}/lowering_selector_error"
        self.global_lowering_failed_dir = f"{args.report_dir}/lowering_failed"
        self.global_translate_timeout_dir = f"{args.report_dir}/translate_timeout"
        self.global_translate_crash_dir = f"{args.report_dir}/translate_crash"
        self.global_sed_timeout_dir = f"{args.report_dir}/sed_timeout"
        self.global_sed_crash_dir = f"{args.report_dir}/sed_crash"
        self.global_clang_timeout_dir = f"{args.report_dir}/clang_timeout"
        self.global_clang_crash_dir = f"{args.report_dir}/clang_crash"
        self.global_execute_timeout_dir = f"{args.report_dir}/execute_timeout"
        self.global_execute_crash_dir = f"{args.report_dir}/execute_crash"
        self.global_wrongcode_dir = f"{args.report_dir}/wrongcode"

        self.global_exception_flags = {
            self.fixub_crash: self.global_fixub_crash_dir,
            self.fixub_timeout: self.global_fixub_timeout_dir,
            self.fixub_failed: self.global_fixub_failed_dir,
            self.checksum_crash: self.global_checksum_crash_dir,
            self.checksum_timeout: self.global_checksum_timeout_dir,
            self.checksum_failed: self.global_checksum_failed_dir,
            self.compile_crash: self.global_compile_crash_dir,
            self.compile_timeout: self.global_compile_timeout_dir,
            self.lowering_selector_error: self.global_lowering_selector_error_dir,
            self.lowering_failed: self.global_lowering_failed_dir,
            self.translate_timeout: self.global_translate_timeout_dir,
            self.translate_crash: self.global_translate_crash_dir,
            self.sed_crash: self.global_sed_crash_dir,
            self.sed_timeout: self.global_sed_timeout_dir,
            self.clang_crash: self.global_clang_crash_dir,
            self.clang_timeout: self.global_clang_timeout_dir,
            self.execute_timeout: self.global_execute_timeout_dir,
            self.execute_crash: self.global_execute_crash_dir,
            self.wrongcode: self.global_wrongcode_dir
        }

        os.system(f"mkdir -p {self.test_dir}")
        os.system(f"mkdir -p {self.mlir_dir}")
        os.system(f"mkdir -p {self.opt_dir}")
        os.system(f"mkdir -p {self.crash_log_dir}")
        os.system(f"mkdir -p {self.executable_dir}")
        os.system(f"mkdir -p {self.tmp_dir}")
        os.system(f"mkdir -p {self.result_dir}")

        os.system(f"mkdir -p {self.report_dir}")

        if not os.path.exists(self.global_fixub_crash_dir):
            os.system(f"mkdir -p {self.global_fixub_crash_dir}")
            os.system(f"mkdir -p {self.global_fixub_timeout_dir}")
            os.system(f"mkdir -p {self.global_fixub_failed_dir}")
            os.system(f"mkdir -p {self.global_checksum_crash_dir}")
            os.system(f"mkdir -p {self.global_checksum_timeout_dir}")
            os.system(f"mkdir -p {self.global_checksum_failed_dir}")
            os.system(f"mkdir -p {self.global_compile_crash_dir}")
            os.system(f"mkdir -p {self.global_compile_timeout_dir}")
            os.system(f"mkdir -p {self.global_lowering_selector_error_dir}")
            os.system(f"mkdir -p {self.global_lowering_failed_dir}")
            os.system(f"mkdir -p {self.global_translate_timeout_dir}")
            os.system(f"mkdir -p {self.global_translate_crash_dir}")
            os.system(f"mkdir -p {self.global_sed_timeout_dir}")
            os.system(f"mkdir -p {self.global_sed_crash_dir}")
            os.system(f"mkdir -p {self.global_clang_timeout_dir}")
            os.system(f"mkdir -p {self.global_clang_crash_dir}")
            os.system(f"mkdir -p {self.global_execute_timeout_dir}")
            os.system(f"mkdir -p {self.global_execute_crash_dir}")

        # this two file can be useless
        os.system(f"cp {args.opt_file} {args.test_dir}/opt.txt")
        os.system(f"cp {args.opt_passes_file} {args.test_dir}/optpasses.txt")

        self.failed_log = f"{self.test_dir}/failed_log.txt"
        self.mlir = f"{self.test_dir}/{self.case_name}.mlir"
        os.system(f"cp {mlir_path} {self.mlir}")

        # for each differential case, we hold the last file.
        self.lowering_cases = []
        self.lowering_finished_flags = []
        self.last_lowering_failed_pass = []

    def handle_exception(self, flag: str):
        """
        Handle exception cases

        Parameters:
            flag: str, exception type
        """
        assert flag in self.global_exception_flags, flag
        os.system(f"cp -r {self.test_dir} {self.global_exception_flags[flag]}")

    def fix_ub(self, in_mlir: str, out_mlir: str):
        """
        Conduct fix ub tool on the given mlir test program.
        Generate the ub-free mlir program.
        Records the crash and timeout behaviors during this process.

        Parameters:
            in_mlir: str, path of given mlir test program
            out_mlir: str, path of given mlir test program.

        Return:
            {self.fixub_success, self.fixub_crash, self.fixub_timeout}
        """
        cmd = f"{args.fixub} {in_mlir} 2>{out_mlir}"
        timeout = execmd_limit_time(cmd, args.timeout)
        if timeout:
            timeout_report = f"{self.report_dir}/{self.case_name}.fixub.timeout.txt"
            os.system(f"cp {out_mlir} {timeout_report}")
            return self.fixub_timeout
        if has_crash_message(out_mlir):
            crash_report = f"{self.report_dir}/{self.case_name}.fixub.crash.txt"
            os.system(f"cp {out_mlir} {crash_report}")
            return self.fixub_crash
        if not is_mlir(out_mlir):
            failed_report = f"{self.report_dir}/{self.case_name}.fixub.failed.txt"
            os.system(f"cp {out_mlir} {failed_report}")
            return self.fixub_failed
        return self.fixub_success

    def checksum(self, in_mlir: str, out_mlir: str):
        """
        Generate checksum for given mlir test program.
        Records all timeout and crashes during this process.

        Parameters:
            in_mlir: str, which is the given mlir test program.
            out_mlir: str, output mlir program that has checksum.

        Return:
            Predefined status strings.
            self.checksum_timeout, self.checksum_crash, self.checksum_success
        """
        cmd = f"{args.checksum} {in_mlir} 2>{out_mlir}"
        timeout = execmd_limit_time(cmd, args.timeout)
        if timeout:
            timeout_report = f"{self.report_dir}/{self.case_name}.checksum.timeout.txt"
            os.system(f"cp {out_mlir} {timeout_report}")
            return self.checksum_timeout
        if has_crash_message(out_mlir):
            crash_report = f"{self.report_dir}/{self.case_name}.checksum.crash.txt"
            os.system(f"cp {out_mlir} {crash_report}")
            return self.checksum_crash
        if not is_mlir(out_mlir):
            failed_report = f"{self.report_dir}/{self.case_name}.checksum.failed.txt"
            os.system(f"cp {out_mlir} {failed_report}")
            return self.checksum_failed
        return self.checksum_success

    def compile_mlir_step_by_step(self, lowering_case: MLIRLoweringCase, opts: list):
        """
        Compile the mlir file with given optimization sequence
        If timeout/failure occurs, we ignore them
        If crash occurs, we terminate the compile process and return self.compile_crash

        valid mlir program always hold in lowering_case.mlir_path

        Parameters:
            lowering_case: MLIRLoweringCase, the lowering information structure
            opts: list, the optimization sequence

        Returns:
            { self.compile_crash, self.compile_success }
        """
        for o in opts:
            res = self.compile_mlir(lowering_case, [o])
            if res == self.compile_crash:
                return res
        return self.compile_success

    def compile_mlir(self, in_lowering_case: MLIRLoweringCase, opts: list):
        """
        Compile the in_lowering_case with given optimizations,
        generate the out_mlir,
        and record the optimizations into opt_recorder file.
        If any crash/compile failure/timeout occurs during this process, the original mlir file is not changed.

        Parameters:
            in_lowering_case: MLIRLoweringCase
            opts: list

        Return:
            predefined status
            self.compile_timeout, self.compile_crash_dir, self.compile_failed, and self.compile_success
        """
        for opt in opts:
            put_file_content(in_lowering_case.opt_recorder, opt + " \\" + "\n")
        opt_str = ' '.join(opts)
        out_mlir = self.test_dir + os.sep + "tmp" + os.sep + "tmp" + ".mlir"
        tmp_crash_log = f"{os.getpid()}.crash.log"

        cmd = f"{args.mlir_opt} {opt_str} {in_lowering_case.mlir_path} -o {out_mlir} 2>{tmp_crash_log}"
        timeout = execmd_limit_time(cmd, args.timeout)
        if timeout:
            # remain original exception handling logic
            put_file_content(self.failed_log, cmd + "\n")
            put_file_content(in_lowering_case.opt_recorder, "[timeout]" + cmd + "\n")
            txt = get_file_content(tmp_crash_log)
            put_file_content(in_lowering_case.opt_recorder, txt + "\n")
            timeout_pass_file = self.test_dir.split("test")[0] + "timeout_pass.txt"
            put_file_content(timeout_pass_file, opt_str + "\n")
            # new exception handling logic
            timeout_log = f"{self.report_dir}/{in_lowering_case.mlir_name}.compile.timeout.txt"
            put_file_content(timeout_log, cmd + '\n' + txt)
            return self.compile_timeout
        if has_crash_message(tmp_crash_log):
            # original exception handling logic
            os.system("cp " + out_mlir + " " + self.test_dir + os.sep + "crashlog" + os.sep + "crash.txt")
            put_file_content(self.failed_log, cmd + "\n")
            # new exception handling logic
            crash_log = f"{self.report_dir}/{in_lowering_case.mlir_name}.compile.crash.txt"
            os.system(f"cp {tmp_crash_log} {crash_log}")
            return self.compile_crash
        if not os.path.exists(out_mlir) or not is_mlir(out_mlir):
            put_file_content(self.failed_log, cmd + "\n")
            put_file_content(in_lowering_case.opt_recorder, "[optimize failed]" + cmd + "\n")
            txt = get_file_content(tmp_crash_log)
            put_file_content(in_lowering_case.opt_recorder, txt + "\n")
            optimize_failed_pass_file = self.test_dir.split("test")[0] + "optimize_failed_pass.txt"
            put_file_content(optimize_failed_pass_file, opt_str + "\n")
            # useless branch for me, but remain the old one.
            return self.compile_failed
        os.system(f"cp {out_mlir} {in_lowering_case.mlir_path}")
        return self.compile_success

    def translate(self, in_lowering_case: MLIRLoweringCase, ll_out: str):
        """
        Translate the input mlir program to llvm ir

        Parameters:
            in_lowering_case: MLIRLoweringCase, the input mlir program
            ll_out: str, path for output llvm ir.

        Return:
            status,
            { self.translate_timeout, self.translate_crash, self.translate_success }
        """
        mlir_translate = f"{args.mlir_opt.split('mlir-opt')[0]}/mlir-translate"
        translate_crash_log = f"{self.report_dir}/{in_lowering_case.mlir_name}.translate.crash.txt"
        cmd = f"{mlir_translate} --mlir-to-llvmir {in_lowering_case.mlir_path} > {ll_out} 2>{translate_crash_log}"
        timeout = execmd_limit_time(cmd, args.timeout)
        if timeout:
            translate_timeout_log = f"{self.report_dir}/{in_lowering_case.mlir_name}.translate.timeout.txt"
            put_file_content(translate_timeout_log, cmd)
            return self.translate_timeout
        if not os.path.exists(ll_out) or os.path.getsize(ll_out) == 0:
            put_file_content(translate_crash_log, '\n' + cmd)
            return self.translate_crash
        return self.translate_success

    def modify_func_name(self, mlir_name: str, in_llvm: str, out_llvm: str):
        """
        Replace function name ``func1'' to ``main'' to satisfy C program requirement.

        Parameters:
            mlir_name: str, path of mlir_name, which is the identifier of report files.
            in_llvm: str, path of input llvm ir
            out_llvm: str, path of output llvm ir

        Return:
            status,
            { self.sed_timeout, self.sed_crash, self.sed_success }
        """
        sed_crash_log = f"{self.report_dir}/{mlir_name}.sed.crash.txt"
        sed_timeout_log = f"{self.report_dir}/{mlir_name}.sed.timeout.txt"

        cmd = f"sed 's/@func1/@main/' {in_llvm} > {out_llvm} 2>{sed_crash_log}"
        timeout = execmd_limit_time(cmd, args.timeout)
        if timeout:
            put_file_content(sed_timeout_log, cmd)
            return self.sed_timeout
        if not os.path.exists(out_llvm) or os.path.getsize(out_llvm) == 0:
            put_file_content(sed_crash_log, cmd)
            return self.sed_crash
        os.system(f"rm -f {sed_crash_log}")
        return self.sed_success

    def compile_to_executable(self, mlir_name: str, llvm_ir: str, executable: str):
        """
        Compile the given llvm ir to executable.

        Parameters:
            mlir_name: str, prefix of report file name
            llvm_ir: str, given llvm ir
            executable: str, output executable file path.

        Return:
            state
            { self.clang_timeout, self.clang_crash, self.clang_success }
        """
        build_path = args.mlir_opt.split("bin")[0]
        clang = f"{build_path}/bin/clang"
        c_runner_utils_lib = f"{build_path}/lib/libmlir_c_runner_utils.so"
        float16_utils_lib = f"{build_path}/lib/libmlir_float16_utils.so"
        runner_utils_lib = f"{build_path}/lib/libmlir_runner_utils.so"
        cpp_std_lib = f"/usr/lib/gcc/x86_64-linux-gnu/9/libstdc++.so"

        crash_log = f"{self.report_dir}/{mlir_name}.clang.crash.txt"
        timeout_log = f"{self.report_dir}/{mlir_name}.clang.timeout.txt"

        cmd = " ".join([clang, "-lm", llvm_ir,
                        c_runner_utils_lib, float16_utils_lib, runner_utils_lib, cpp_std_lib,
                        "-o", executable, "2>", crash_log])

        timeout = execmd_limit_time(cmd, args.timeout)
        if timeout:
            put_file_content(timeout_log, cmd)
            return self.clang_timeout
        if not os.path.exists(executable) or os.path.getsize(executable) == 0:
            crash_message = get_file_content(crash_log)
            if "__extendhfsf2" in crash_message or "__truncsfhf2" in crash_message:
                cmd = " ".join([clang, "-lm", "-mf16c", llvm_ir,
                                c_runner_utils_lib, float16_utils_lib, runner_utils_lib, cpp_std_lib,
                                "-o", executable, "2>", crash_log])
                timeout = execmd_limit_time(cmd, args.timeout)
                if timeout:
                    put_file_content(timeout_log, cmd)
                    return self.clang_timeout
                if not os.path.exists(executable) or os.path.getsize(executable) == 0:
                    return self.clang_crash
            else:
                return self.clang_crash
        os.system(f"rm -f {crash_log}")
        return self.clang_success

    def re_lowering_and_execute(self, lowering_case: MLIRLoweringCase):
        """
        Eliminate non-terminate loops with canonicalize pass,
        re-lowering, optimize, translate, compile to executable and execute.

        Parameters:
            lowering_case: MLIRLoweringCase, the mlir program to be lowered.

        Return:
            state
            { self.compile_crash,
              self.translate_timeout, self.translate_crash,
              self.lowering_selector_error, self.lowering_failed,
              self.sed_timeout, self.sed_crash,
              self.clang_timeout, self.clang_crash,
              self.execute_timeout, self.execute_crash,
              self.re_lowering_success }
        """
        res = self.compile_mlir(lowering_case, ["--canonicalize"])
        if res == self.compile_crash:
            self.handle_exception(res)
            return res

        finished = False
        last_failed_passes = []
        for iterate_time in range(args.lowering_iter_limit):
            dyn_pass_selector = DynamicPassSelector(lowering_case, last_failed_passes)
            selected_pass = dyn_pass_selector.main()
            if selected_pass == dyn_pass_selector.lowering_finished:
                finished = True
                break
            if selected_pass == dyn_pass_selector.selector_error:
                selector_error_log = f"{self.report_dir}/{lowering_case.mlir_name}.relowering.selector.error.txt"
                put_file_content(selector_error_log,
                                 lowering_case.mlir_name + '\n' + lowering_case.mlir_path)
                self.handle_exception(self.lowering_selector_error)
                return self.lowering_selector_error

            before_program = get_file_content(lowering_case.mlir_path)
            res = self.compile_mlir(lowering_case, [selected_pass])
            after_program = get_file_content(lowering_case.mlir_path)
            if before_program == after_program:
                last_failed_passes.append(selected_pass)
            else:
                last_failed_passes = []
            if res == self.compile_crash:
                self.handle_exception(res)
                return res

        if not finished:
            lowering_failed_log = f"{self.report_dir}/{lowering_case.mlir_name}.relowering.failed.txt"
            put_file_content(lowering_failed_log,
                             lowering_case.mlir_name + '\n' + lowering_case.mlir_path)
            self.handle_exception(self.lowering_failed)
            return self.lowering_failed

        # translate
        ll_out = f"{self.test_dir}/tmp/{lowering_case.mlir_name}.ll"
        res = self.translate(lowering_case, ll_out)
        if res != self.translate_success:
            self.handle_exception(res)
            return res

        # change file name
        ll_out_final = f"{self.test_dir}/tmp/{lowering_case.mlir_name}.final.ll"
        res = self.modify_func_name(lowering_case.mlir_name, ll_out, ll_out_final)
        if res != self.sed_success:
            self.handle_exception(res)
            return res

        # compile to executable
        executable = f"{self.test_dir}/out/{lowering_case.mlir_name}.out"
        res = self.compile_to_executable(lowering_case.mlir_name, ll_out_final, executable)
        if res != self.clang_success:
            self.handle_exception(res)
            return res

        # execute
        executable = f"{self.test_dir}/out/{lowering_case.mlir_name}.out"
        result_file = f"{self.test_dir}/result/{lowering_case.mlir_name}.txt"
        res = self.execute(lowering_case.mlir_name, executable, result_file)
        if res != self.execute_success:
            self.handle_exception(res)
            return res

        return self.re_lowering_success

    def compile_and_lowering(self, lowering_case: MLIRLoweringCase, have_opt=True):
        """
        Conduct optimization and lowering on the given lowering_case.

        Parameters:
            lowering_case: MLIRLoweringCase, the MLIR program to be lowered.
            have_opt: whether conduct optimization before lowering, default is True.

        Return:
            state
            { self.compile_crash,
              self.lowering_selector_error,
              self.lowering_failed,
              self.lowering_success }
        """
        # assert len(lowering_cases) == 1
        # lowering_case = lowering_cases[0]

        lowering_finished = False
        last_lowering_failed_pass = []

        iterate_time = 0

        while not lowering_finished:
            iterate_time += 1

            # optimization
            if have_opt:
                dialects = collect_dialects(lowering_case.mlir_path)
                opts = recommend_opts(dialects)
                res = self.compile_mlir_step_by_step(lowering_case, opts)
                if res == self.compile_crash:
                    self.handle_exception(res)  # here only records compile crash
                    return res

            # lowering
            dyn_pass_selector = DynamicPassSelector(lowering_case, last_lowering_failed_pass)
            lowering_pass = dyn_pass_selector.main()
            print(f"[compile_and_lowering] lowering passes: {lowering_pass}")
            print(f"[compile_and_lowering] lowering failed passes: {last_lowering_failed_pass}")
            if lowering_pass == dyn_pass_selector.lowering_finished:
                lowering_finished = True
                continue
            if lowering_pass == dyn_pass_selector.selector_error:
                selector_error_log = f"{self.report_dir}/{lowering_case.mlir_name}.lowering.selector.error.txt"
                put_file_content(selector_error_log, lowering_case.mlir_name + '\n' + lowering_case.mlir_path)
                self.handle_exception(self.lowering_selector_error)
                return self.lowering_selector_error

            before_program = get_file_content(lowering_case.mlir_path)
            res = self.compile_mlir(lowering_case, [lowering_pass])
            after_program = get_file_content(lowering_case.mlir_path)
            # this check mainly targets on the compile_timeout and compile_failed.
            if before_program == after_program:
                last_lowering_failed_pass.append(lowering_pass)
            else:
                last_lowering_failed_pass = []
            print(f"[compile_and_lowering] lowering compile res: {res}")
            if res == self.compile_crash:
                self.handle_exception(res)
                return res

            if iterate_time == args.lowering_iter_limit:
                lowering_failed_log = f"{self.report_dir}/{lowering_case.mlir_name}.lowering.failed.txt"
                put_file_content(lowering_failed_log, lowering_case.mlir_name + '\n' + lowering_case.mlir_path)
                self.handle_exception(self.lowering_failed)
                return self.lowering_failed

        return self.lowering_success

    def execute(self, mlir_name: str, executable: str, result_file: str):
        """
        Execute the executable

        Parameters:
            mlir_name: str, the report name prefix
            executable: str, the path of executable
            result_file: str, the result file

        Return:
            state
            { self.execute_timeout, self.execute_crash, self.execute_success }
        """
        crash_log = f"{self.report_dir}/{mlir_name}.execute.crash.txt"
        timeout_log = f"{self.report_dir}/{mlir_name}.execute.timeout.txt"
        cmd = f"{executable} > {result_file} 2>{crash_log}"
        timeout = execmd_limit_time(cmd, args.timeout)
        if timeout:
            put_file_content(timeout_log, cmd)
            return self.execute_timeout
        if not os.path.exists(result_file) or os.path.getsize(result_file) == 0:
            put_file_content(crash_log, '\n' + cmd)
            return self.execute_crash
        return self.execute_success

    def fix_and_lowering(self):
        """
        This function conduct the following process:
        1. [fixub]
            fix the undefined behavior of the given MLIR test program.
        2. [checksum]
            generate checksum for the given MLIR test program.
        3. [compile_and_lowering]
            conduct the lowering-optimization process.
        4. [translate, function_rename, compile_to_executable]
            translate: translate the mlir file to llvm ir
            function rename: rename func1 to main function, which satisfies C program.
            compile to executable: use clang to convert llvm ir to executable
        5. [execute]
            execute the executable
        5. [compare result and re-lowering]
            compare the result, and re-lowering using canonicalize pass to eliminate some non-terminate loops.

        Effects of this function:
        1. conduct the above steps
        2. record the exception record(crash and timeout for each step)
        3. copy the exception test to the report directory(global)

        Return:
            state
        """
        # fixub
        self.fix_ub_mlir = f"{self.test_dir}/fixub.mlir"
        res = self.fix_ub(self.mlir, self.fix_ub_mlir)
        if res != self.fixub_success:
            self.handle_exception(res)
            return res

        # checksum
        self.checksum_mlir = f"{self.test_dir}/checksum.mlir"
        res = self.checksum(self.fix_ub_mlir, self.checksum_mlir)
        if res != self.checksum_success:
            self.handle_exception(res)
            return res

        # lowering
        self.lowering_cases = []
        # original case
        lowering_case = MLIRLoweringCase(self.test_dir, 0, self.case_name)
        os.system(f"cp {self.checksum_mlir} {lowering_case.mlir_path}")
        self.lowering_cases.append(lowering_case)
        res = self.compile_and_lowering(lowering_case, have_opt=False)
        if res != self.lowering_success:
            return res
        # differential cases
        for diff_cnt in range(args.diff_num):
            lowering_case = MLIRLoweringCase(self.test_dir, diff_cnt+1, self.case_name)
            os.system(f"cp {self.checksum_mlir} {lowering_case.mlir_path}")
            self.lowering_cases.append(lowering_case)
            res = self.compile_and_lowering(lowering_case)
            if res != self.lowering_success:
                return res

        for lowering_case in self.lowering_cases:
            # translate
            ll_out = f"{self.test_dir}/tmp/{lowering_case.mlir_name}.ll"
            res = self.translate(lowering_case, ll_out)
            if res != self.translate_success:
                self.handle_exception(res)
                return res

            # change file name
            ll_out_final = f"{self.test_dir}/tmp/{lowering_case.mlir_name}.final.ll"
            res = self.modify_func_name(lowering_case.mlir_name, ll_out, ll_out_final)
            if res != self.sed_success:
                self.handle_exception(res)
                return res

            # compile to executable
            executable = f"{self.test_dir}/out/{lowering_case.mlir_name}.out"
            res = self.compile_to_executable(lowering_case.mlir_name, ll_out_final, executable)
            if res != self.clang_success:
                self.handle_exception(res)
                return res

        # execute
        for lowering_case in self.lowering_cases:
            executable = f"{self.test_dir}/out/{lowering_case.mlir_name}.out"
            result_file = f"{self.test_dir}/result/{lowering_case.mlir_name}.txt"
            res = self.execute(lowering_case.mlir_name, executable, result_file)
            if res == self.execute_crash:
                self.handle_exception(res)
                return res

        # compare wrongcode
        for lowering_case in self.lowering_cases:
            result_file1 = f"{self.test_dir}/result/{lowering_case.mlir_name}.txt"
            result_file2 = f"{self.test_dir}/result/{self.lowering_cases[0].mlir_name}.txt"
            if not file_cmp(result_file1, result_file2):
                if get_file_content(result_file1).startswith(get_file_content(result_file2)):
                    mlir_file1 = f"{self.test_dir}/mlir/{self.lowering_cases[0].mlir_name}.mlir"
                    mlir_opt_file1 = f"{self.test_dir}/opt/{self.lowering_cases[0].mlir_name}.txt"
                    os.system(f"rm -f {mlir_opt_file1}")
                    os.system(f"cp {self.checksum_mlir} {mlir_file1}")
                    res = self.re_lowering_and_execute(self.lowering_cases[0])  # this function contains exception handling
                    if res != self.re_lowering_success:
                        return res
                    if file_cmp(result_file1, result_file2):
                        continue
                self.handle_exception(self.wrongcode)
                return self.wrongcode
        return self.success


class Coverage:
    COVERAGE = set()

    def __init__(self, mlir_path):
        self.mlir_path = mlir_path
        pass

    def collect_cov(self):
        cmd = ' '.join([args.collector, self.mlir_path, "--depth", str(args.cov_depth), args.cov_file])
        print(f'[Coverage.collect_cov] {cmd}')
        os.system(cmd)
        cov = set()
        if os.path.exists(args.cov_file):
            cov = set(get_file_lines(args.cov_file))
        os.system(f"rm -f {args.cov_file}")
        return cov

    def collect_and_compare(self):
        lines_before = len(Coverage.COVERAGE)
        cov = self.collect_cov()
        Coverage.COVERAGE |= cov
        lines_after = len(Coverage.COVERAGE)
        assert lines_after >= lines_before
        update = lines_before < lines_after
        print("[Coverage.collect_and_compare] coverage length=" + str(len(Coverage.COVERAGE)))
        return update

        # return True


class Mutator:
    def __init__(self, in_seed: Seed):
        self.in_seed = in_seed
        self.out = None
        self.mutation_timeout = "mutation_timeout"
        self.mutation_crash = "mutation_crash"
        self.mutation_success = "mutation_success"
        self.mutation_failed = "mutation_failed"
        self.mutators = [self.replace_data_edge, self.replace_control_edge,
                         self.delete_node, self.create_node]
        # self.mutators = [self.replace_data_edge, self.replace_control_edge,
        #                  self.delete_node]
        # self.mutators = [self.replace_data_edge, self.replace_control_edge, self.create_node]
        # self.mutators = [self.replace_control_edge, self.delete_node, self.create_node]
        # self.mutators = [self.replace_data_edge, self.delete_node, self.create_node]

    def replace_data_edge(self):
        random_name = random_file_prefix() + '.mlir'
        new_mlir = args.seed_dir + os.sep + random_name
        self.out = new_mlir
        cmd = ' '.join([args.replace_data_edge, "--operand-prob", str(args.operand_prob), self.in_seed.path, new_mlir])
        timeout = execmd_limit_time(cmd, args.timeout)
        if timeout:
            return self.mutation_timeout
        if not os.path.exists(new_mlir) or get_line_number(new_mlir) < 100:
            return self.mutation_crash
        return self.mutation_success

    def replace_control_edge(self):
        random_name = random_file_prefix() + '.mlir'
        new_mlir = args.seed_dir + os.sep + random_name
        self.out = new_mlir
        cmd = ' '.join([args.replace_control_edge, self.in_seed.path, new_mlir])
        timeout = execmd_limit_time(cmd, args.timeout)
        if timeout:
            return self.mutation_timeout
        if not os.path.exists(new_mlir) or not_mlir(new_mlir):
            return self.mutation_crash
        return self.mutation_success

    def delete_node(self):
        random_name = random_file_prefix() + '.mlir'
        new_mlir = args.seed_dir + os.sep + random_name
        self.out = new_mlir
        cmd = ' '.join([args.delete_node, self.in_seed.path, new_mlir])
        timeout = execmd_limit_time(cmd, args.timeout)
        if timeout:
            return self.mutation_timeout
        if not os.path.exists(new_mlir) or not_mlir(new_mlir):
            return self.mutation_crash
        return self.mutation_success

    def create_node(self):
        random_name = random_file_prefix() + '.mlir'
        new_mlir = args.seed_dir + os.sep + random_name
        self.out = new_mlir
        cmd = ' '.join([args.create_node, self.in_seed.path, new_mlir])
        timeout = execmd_limit_time(cmd, args.timeout)
        if timeout:
            return self.mutation_timeout
        if not os.path.exists(new_mlir) or not_mlir(new_mlir):
            return self.mutation_crash
        return self.mutation_success

    def mutate(self):
        mutation = self.mutators[random.randint(0, len(self.mutators) - 1)]
        print(f'[Mutator.mutate] Selected mutator: {mutation}')
        res = mutation()
        print(f'[Mutator.mutate] Mutation result: {res}')
        self.in_seed.cnt += 1
        return res


def main():
    print('[main] Start fuzzing process')
    seed_pool = SeedPool()
    print('[main] Start pool initialization')
    seed_pool.init()

    while not seed_pool.empty():
        print(f'[main] seed pool size: {len(seed_pool.pool)}')

        mutate_success = False
        new_mlir = None
        new_seed = None
        old_seed = None
        while not mutate_success:
            # seed selection
            s = seed_pool.select()
            old_seed = s
            print(f'[main] Seed selection: {s}')
            if not s.tested:
                print(f'[main] Seed execution: {s}')
                test_batch = TestBatch(s.path)
                res = test_batch.fix_and_lowering()
                s.tested = True
                print(f"[main] test result: {[s.path, res]}")
                os.system(f"rm -rf {test_batch.test_dir}")
            print('[main] *************************************')

            # seed mutation
            print(f'[main] Seed mutation')
            m = Mutator(s)
            res = m.mutate()
            if res != m.mutation_success:
                continue
            else:
                mutate_success = True
            new_seed = Seed(m.out)
            new_mlir = m.out
            print('[main] *************************************')

        assert new_mlir is not None
        assert new_seed is not None

        print(f'[main] Seed retention')
        if len(diff(new_seed.path, old_seed.path)) > 10:
            c1 = Coverage(new_mlir)
            if c1.collect_and_compare():
                seed_pool.add(new_seed)
            else:
                os.system(f'rm -f {new_mlir}')
                print(f'[main] new seed does not have new coverage')
                continue
            print('[main] *************************************')

            if not new_seed.tested:
                print(f'[main] Seed execution: {new_seed}')
                test_batch = TestBatch(new_seed.path)
                res = test_batch.fix_and_lowering()
                new_seed.tested = True
                print(f"[main] test result: {[new_seed.path, res]}")
                os.system(f"rm -rf {test_batch.test_dir}")

        seed_pool.clean()


"""
Usage:
python3 MLIRod_FIXUB.py \
--seed existing \
--existing_seed_dir existing_seeds \
--mlir_opt /data2/src/llvm-project/build/bin/mlir-opt \
--mlirsmith /data2/src/llvm-dev/FIXUB-dev/build/bin/toyc-ch5_dynamicLowering \
--fixub /data2/src/llvm-dev/FIXUB-dev/build/bin/FixUBnew \
--checksum /data2/src/llvm-dev/FIXUB-dev/build/bin/CheckSum \
--replace_data_edge /data2/src/llvm-dev/FIXUB-dev/build/bin/ReplaceDataEdge \
--replace_control_edge /data2/src/llvm-dev/FIXUB-dev/build/bin/ReplaceControlEdge \
--delete_node /data2/src/llvm-dev/FIXUB-dev/build/bin/DeleteNode \
--create_node /data2/src/llvm-dev/FIXUB-dev/build/bin/CreateNode \
--collector /data2/src/llvm-dev/FIXUB-dev/build/bin/CollectCoverage \
--clean True >log.log 2>error.log
"""
if __name__ == '__main__':
    random.seed(time.time())
    parser = argparse.ArgumentParser()
    # OOA arguments
    parser.add_argument("--timeout", type=int, help="Size of seed pool.", default=10)
    parser.add_argument("--clean", type=bool, help="Clean", default=False)
    parser.add_argument("--lowering_iter_limit", type=int, help="max lowering iteration count.", default=50)
    parser.add_argument("--diff_num", type=int, help="the number of differential executables.", default=1)
    parser.add_argument("--opt_num", type=int, help="the number of optimization passes at each step.", default=1)
    # arguments in seed pool initialization
    parser.add_argument("--init_size", type=int, help="Size of seed pool.", default=1000)
    parser.add_argument("--seed_dir", type=str, help="Seed directory.", default="seeds")
    parser.add_argument("--seed", type=str, help="Method to initialize the seed pool", default="mlirsmith")
    parser.add_argument("--existing_seed_dir", type=str, help="The path of existing seed directory", default="seeds")
    # Seed pool initialization from mlirsmith.
    parser.add_argument("--fixub", type=str, help="Path of fixub", default="")
    parser.add_argument("--mlirsmith", type=str, help="Path of mlirsmith", default="")
    parser.add_argument("--checksum", type=str, help="Path of checksum", default="")
    parser.add_argument("--conf", type=str, help="Path of configuration file", default="cov.json")
    parser.add_argument("--mlir_template", type=str, help="Path of mlir template", default="template.mlir")
    parser.add_argument("--mlir_opt", type=str, help="Path of mlir_opt", default="")
    # pass files
    parser.add_argument("--opt_passes_file", type=str, help="Path of optimization passes file",
                        default="fuzzingopts/optpasses.txt")
    parser.add_argument("--opt_file", type=str, help="Path of optimization file", default="fuzzingopts/opt.txt")
    parser.add_argument("--lowermap_file", type=str, help="Path of lowering pass map file", default="lowermap.txt")
    parser.add_argument("--prerequisite_passes_file", type=str, help="Path of prerequisite passes file",
                        default="prerequisitepasses.txt")
    parser.add_argument("--regexp_file", type=str, help="Path of regular expression file", default="regexp.txt")
    # test path parameters
    parser.add_argument("--report_dir", type=str, help="Path of report", default="report")
    parser.add_argument("--test_dir", type=str, help="Path of test", default="test")
    # arguments in seed mutation
    parser.add_argument("--replace_data_edge", type=str, help="Path of mutator data edge modification", default="")
    parser.add_argument("--replace_control_edge", type=str, help="Path of mutator control edge modification",
                        default="")
    parser.add_argument("--delete_node", type=str, help="Path of mutator node deletion", default="")
    parser.add_argument("--create_node", type=str, help="Path of mutator node insertion", default="")
    parser.add_argument("--operand_prob", type=float, help="Probability for operands to be selected to mutate",
                        default=0.1)
    parser.add_argument("--mutation_cnt", type=int, help="The mutation count for each seed.",
                        default=10)
    # arguments in seed retention
    parser.add_argument("--collector", type=str, help="OD Coverage collector", default="")
    parser.add_argument("--cov_file", type=str, help="temporary OD Coverage file.", default="coverage.txt")
    parser.add_argument("--cov_depth", type=int, help="OD Coverage steps, the parameter d in parer.", default=2)
    parser.add_argument("--max_selected", type=int, help="Max selected count for each seed.", default=10)
    args = parser.parse_args()

    assert args.seed in ["mlirsmith", "existing"]
    if args.seed == "existing":
        os.path.exists(args.existing_seed_dir)
    assert os.path.exists(args.fixub)
    assert os.path.exists(args.mlir_opt)
    assert os.path.exists(args.checksum)
    assert os.path.exists(args.mlir_template)
    assert os.path.exists(args.opt_passes_file)
    assert os.path.exists(args.opt_file)
    assert os.path.exists(args.lowermap_file)
    assert os.path.exists(args.prerequisite_passes_file)
    assert os.path.exists(args.regexp_file)

    DynamicPassSelector.init_dics(args.lowermap_file, args.prerequisite_passes_file, args.regexp_file)
    if args.clean:
        os.system(f'rm -rf {args.report_dir} {args.test_dir} {args.seed_dir}')

    os.system(f'mkdir -p {args.seed_dir}')
    os.system(f'mkdir -p {args.report_dir}')

    DIALECT_OPTS = get_optimization(args.opt_passes_file)
    print("[main] dialect opts")
    print(DIALECT_OPTS)

    main()
