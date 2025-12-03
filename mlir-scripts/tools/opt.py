"""
Extract all of the optimization from MLIR online documentation.
"""

import os
import time
import requests
from typing import List
from bs4 import BeautifulSoup

MLIR_OPT = "/data1/llvm-project/build/bin/mlir-opt"

def extract_opts(url: str) -> List[str]:
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        nav_tag = soup.find('nav', id='TableOfContents')
        code_tags = nav_tag.find_all('code')

        opts = []
        for code_tag in code_tags:
            opt = code_tag.get_text(strip=True)
            if opt:
                opts.append(opt)

        return opts

    except Exception as e:
        print(f"Error: {e}")
        return []

def test(opts):
    content = "func.func nested @func1() -> () { \n %0 = arith.constant true \n return \n }"
    if os.path.exists('a.mlir'):
        os.remove("a.mlir")
    f = open('a.mlir', 'a+')
    f.write(content)
    f.close()
    
    valid_opts = []
    for o in opts:
        cmd = f'{MLIR_OPT} a.mlir 1>/dev/null 2>/dev/null'
        if not os.system(cmd):
            valid_opts.append(o)
    return valid_opts

if __name__ == '__main__':
    url = "https://mlir.llvm.org/docs/Passes/"  # Replace with actual URL
    opts = extract_opts(url)
    valid_opts = test(opts)

    if os.path.exists("opt.txt"):
        os.remove("opt.txt")
    f = open("opt.txt", "a+")
    f.write('\n'.join(valid_opts))
    f.close()
