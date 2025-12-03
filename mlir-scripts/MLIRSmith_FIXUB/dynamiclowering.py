import re
import os
import argparse
import random
import math
import time
import filecmp


def get_file_content(file_path):
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


def fil_cmp(path1, path2):
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


class Testmlir:
    # 输入为测试目录，和mlir文件名
    def __init__(self, mlirpath, opttxtpath):
        self.mlirpath = mlirpath
        self.opttxtpath = opttxtpath


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
    including same operation that requires different lowering path eg : {"scf\.while .+ tensor" : "scf.while2"}.
    These regular expressions are used to recognize specific operations that deal with some specific types.
    For exmaple, ''scf.while .+ tensor'' means that scf.while operation with tensor operands.
    """
    regexp = {}

    # special judges for continuous passes
    @staticmethod
    def specialJudge(_pass):
        if _pass == "--one-shot-bufferize=\"dialect-filter=linalg\"":
            return " --convert-linalg-to-loops"
        if _pass == "--expand-strided-metadata":
            return " --func-bufferize"
        if _pass == "--one-shot-bufferize=\"dialect-filter=arith\"":
            return " --convert-arith-to-llvm"
        if _pass == "--convert-vector-to-scf":
            return " --convert-vector-to-llvm"
        return "not special case"

    @staticmethod
    def init_dics(lowermapfile, prerequisitepassesfile, regexpfile):
        """

        :param lowermapfile: lowermap.txt
        :param prerequisitepassesfile: prerequisitepasses.txt
        :param regexpfile: regexp.txt
        :return:
        """

        two_file = [lowermapfile, prerequisitepassesfile, regexpfile]
        for file in two_file:
            lines = get_file_lines(file)
            for line in lines:
                key = line.split(":")[0].strip()
                values = line.split(":")[1].strip().split(" ")
                passlist = []

                # TODO this code can be wrong
                tocontinue = 0
                for value in values:
                    if tocontinue:
                        tocontinue = 0
                        continue

                    # special judges for continuous passes
                    judgeres = DynamicPassSelector.specialJudge(value)  # why not directly use the replace method.
                    if judgeres != "not special case":
                        passlist.append(value + judgeres)
                        tocontinue = 1
                        continue

                    passlist.append(value)
                if file == lowermapfile:
                    DynamicPassSelector.lower_map[key] = passlist
                elif file == regexpfile:
                    DynamicPassSelector.regexp[key] = passlist
                else:

                    # TODO: change to setdefault function.
                    if key in DynamicPassSelector.prerequisite_passes:
                        for _pass in passlist:
                            DynamicPassSelector.prerequisite_passes[key].append(_pass)
                    else:
                        DynamicPassSelector.prerequisite_passes[key] = passlist
        return

    def __init__(self, mlir, lastfailed):
        self.mlirpath = mlir.mlirpath      # the path of mlir program
        self.opttxtpath = mlir.opttxtpath  # the path of optimization record of this mlir program
        self.possible_passes = {}          # ???
        self.operations = {}               # the operations in MLIR file.
        self.lastfailed = lastfailed       # last selected lowering passes that failed to lower the MLIR program.
        self.lowering_finished = "lowering_finished"
        self.selector_error = "selector_error"
        return

    def getOperation(self):
        file_lines = get_file_lines(self.mlirpath)
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

                # TODO: "self.operations[m[0]] =" is redudant, because this is the initialization code rather than
                #  modification code.
                self.operations[m[0]] = self.operations.setdefault(m[0], 1)
        return

    def calPassIn(self):
        """
        Collect all passes
        :return:
        """
        for operation in self.operations:
            if operation not in DynamicPassSelector.lower_map:
                continue
            passes = DynamicPassSelector.lower_map[operation]
            first = 1

            # TODO: refine the usage of setdefault function
            for _pass in passes:
                if first == 1:
                    first = 0
                    self.possible_passes[_pass] = self.possible_passes.setdefault(_pass, 0)
                    continue
                else:
                    self.possible_passes[_pass] = self.possible_passes.setdefault(_pass, 0)
                    self.possible_passes[_pass] = self.possible_passes[_pass] + 1
        return

    def addPrerequisite(self):
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

        for failed_pass in self.lastfailed:
            self.possible_passes[failed_pass] = self.possible_passes[failed_pass] + 1
        return

    def getToExcutePass(self):
        """
        TODO: This is a very strange function.
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
        self.getOperation()
        self.calPassIn()
        self.addPrerequisite()
        print(self.possible_passes)
        return self.getToExcutePass()


def excute(testmlir, to_excute, tmp_path):
    opt = f"{args.mlir_opt} {to_excute} {testmlir.mlirpath} > {tmp_path}"
    # print(opt)
    os.system(opt)
    if len(get_file_content(tmp_path)) == 0:
        return "failed"
    if get_file_content(mlir_path) == get_file_content(tmp_path):
        return "failed"
    os.system(f"cp {tmp_path} {mlir_path}")
    put_file_content(testmlir.opttxtpath, f"{to_excute} \n")

    return "success"


"""
python3 dynamiclowering.py \
--mlir_opt /data/tmp/v0717/llvm-project/build/bin/mlir-opt \
--lowermapfile=/data/senbai_doc_11_15/dynamic-lower-pass/lowermap.txt \
--prerequisitepassesfile=/data/senbai_doc_11_15/dynamic-lower-pass/prerequisitepasses.txt \
--regexpfile=/data/senbai_doc_11_15/dynamic-lower-pass/regexp.txt
"""

if __name__ == '__main__':

    random.seed(time.time())
    parser = argparse.ArgumentParser()
    # OOA arguments
    parser.add_argument("--lowermapfile", type=str, help="Path of lowering pass map file", default="")
    parser.add_argument("--prerequisitepassesfile", type=str, help="Path of prerequisite passes file", default="")
    parser.add_argument("--regexpfile", type=str, help="Path of regular expression file", default="")
    parser.add_argument("--mlir_opt", type=str, help="Path of mlir_opt", default="")
    args = parser.parse_args()

    DynamicPassSelector.init_dics(args.lowermapfile, args.prerequisitepassesfile, args.regexpfile)

    ori_mlir_path = "/data/senbai_doc_11_15/dynamic-lower-pass/checksum.mlir"
    opt_path = "/data/senbai_doc_11_15/dynamic-lower-pass/optchecksum.txt"
    mlir_path = "/data/senbai_doc_11_15/dynamic-lower-pass/result.mlir"
    tmp_path = "/data/senbai_doc_11_15/dynamic-lower-pass/tmp.mlir"
    os.system(f"cp {ori_mlir_path} {mlir_path}")
    os.system(f"rm -rf {opt_path}")
    max_iterate_time = 40

    lastfailed = []
    for i in range(max_iterate_time):
        testmlir = Testmlir(mlir_path, opt_path)
        selector = DynamicPassSelector(testmlir, lastfailed)
        to_excute = selector.main()
        print("\n")
        print(to_excute)
        if to_excute == "lowering finished!" or to_excute == "error!":
            break
        status = excute(testmlir, to_excute, tmp_path)
        if status == "failed":
            lastfailed.append(to_excute)
        else:
            lastfailed = []
        print("\nlowering times: ")
        print(i)
        print("\n")
