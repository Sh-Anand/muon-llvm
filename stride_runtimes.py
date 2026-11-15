#!/usr/bin/env python3
"""Build libc.a, libm.a and libclang_rt.builtins.a for the lane-interleaved stack.

-mllvm -riscv-stack-word-stride=N changes the stack ABI, so everything linked into a program
built with it must be too.  This recompiles the objects of the three archives exactly as the
runtimes build compiled them, plus the stride flag, into a separate directory; the normal
runtimes are left alone.

usage: stride_runtimes.py <llvm build dir> <stride> <output dir>
"""
import json, os, shlex, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

build, stride, out = os.path.realpath(sys.argv[1]), sys.argv[2], os.path.realpath(sys.argv[3])
sub = f"{build}/runtimes/runtimes-riscv32-unknown-elf-bins"
commands = {os.path.relpath(c["output"], sub): c for c in json.load(open(f"{sub}/compile_commands.json"))}

# the objects of each archive, from its rule in the runtimes build
archives = {}
for line in open(f"{sub}/build.ninja"):
    for name in ("libc.a", "libm.a", "libclang_rt.builtins.a"):
        if line.startswith("build ") and line.split(":")[0].endswith("/" + name):
            archives[name] = [o for o in line.split(":", 1)[1].split("|")[0].split()[1:] if o.endswith(".o")]

def compile(obj):
    c = commands[obj]
    dst = f"{out}/obj/{obj}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    args = shlex.split(c["command"])
    args[args.index("-o") + 1] = dst
    if not c["file"].endswith(".S"):
        args += ["-mllvm", f"-riscv-stack-word-stride={stride}"]
    subprocess.run(args, cwd=c["directory"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dst

with ThreadPoolExecutor(os.cpu_count()) as pool:
    for name, objs in archives.items():
        built = list(pool.map(compile, objs))
        if os.path.exists(f"{out}/{name}"):
            os.remove(f"{out}/{name}")
        subprocess.run([f"{build}/bin/llvm-ar", "rcs", f"{out}/{name}"] + built, check=True)
        print(f"{out}/{name}: {len(built)} objects")
