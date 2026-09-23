import taichi as ti

_arch = None # set by the first init(); None means Taichi not started yet

def init(backend):
    arch = "cuda" if backend == "cuda" else "cpu"
    global _arch
    if _arch is None: 
        ti.init(arch=ti.cuda if arch == "cuda" else ti.cpu)
        _arch = arch
    elif _arch != arch:
        raise RuntimeError(f"Cannot change backend from {_arch} to {arch}")
