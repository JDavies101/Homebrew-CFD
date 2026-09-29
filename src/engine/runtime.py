# one Taichi init per process, on cpu or cuda
import taichi as ti

# set by the first init(); None means Taichi not started yet
_arch = None

def init(backend):
    """
    Start Taichi on the requested backend once; later calls must ask for the same backend.
    """

    global _arch
    arch = "cuda" if backend == "cuda" else "cpu"

    if _arch is None:
        ti.init(arch=ti.cuda if arch == "cuda" else ti.cpu)
        _arch = arch
    elif _arch != arch:
        raise RuntimeError(f"Cannot change backend from {_arch} to {arch}")
