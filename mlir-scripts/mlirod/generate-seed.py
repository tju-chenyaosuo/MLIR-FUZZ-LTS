import os
import argparse


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


def get_file_lines(file_path):
    f = open(file_path)
    lines = f.readlines()
    return lines


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


def not_mlir(file):
    cmd = f"file -r {file}"
    ret = execmd(cmd)
    print(ret)
    return ": ASCII text" not in ret


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
        # cmd = f"{args.mlirsmith} -emit=mlir-affine -dfile={args.conf} {args.mlir_template} 2>{mlir_path}"
        cmd = f"{args.mlirsmith} 2>{mlir_path}"
        timeout = execmd_limit_time(cmd, args.timeout)
        if timeout:
            return self.mlir_timeout
        mlir_file_len = get_line_number(mlir_path)
        if mlir_file_len < 100:
            return self.mlir_crash
        cmd = f"{args.opt} {mlir_path} 2>xxx"
        timeout = execmd_limit_time(cmd, args.timeout)
        if timeout:
            return self.mlir_timeout
        if get_line_number('xxx') >= 1:
            return self.mlir_crash
        return self.mlir_success


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


def main():
    factory = MLIRSmithSeedFactory()
    for _ in range(50):
        s = factory.gen_seed()
        print(s)


"""
python3 generate-seed.py \
--mlirsmith /data3/2025.04.07.liangjingjing/bin/llvm-13c6abfac84fca4bc55c0721d1853ce86a385678/build/bin/mlirsmith \
--opt /data3/2025.04.07.liangjingjing/bin/llvm-13c6abfac84fca4bc55c0721d1853ce86a385678/build/bin/mlir-opt
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_dir", type=str, help="Seed directory.", default="seeds")
    parser.add_argument("--max_selected", type=int, help="Max selected count for each seed.", default=10)
    parser.add_argument("--mlirsmith", type=str, help="Path of mlirsmith", default="")
    parser.add_argument("--opt", type=str, help="Path of mlirsmith", default="")
    parser.add_argument("--conf", type=str, help="Path of configuration file", default="cov.json")
    parser.add_argument("--timeout", type=int, help="Size of seed pool.", default=10)
    parser.add_argument("--mlir_template", type=str, help="Path of mlir template", default="template.mlir")
    args = parser.parse_args()

    os.system(f"mkdir {args.seed_dir}")
    main()
