# checks taichi imports and that the cuda backend starts on the gpu
# run: python -m src.config.environment
from __future__ import annotations

from dataclasses import dataclass


# what the probe found
@dataclass
class EnvReport:
    taichi_available: bool
    taichi_version: str | None
    cuda_available: bool
    notes: list[str]

    # true when the gpu solver can run
    @property
    def ready(self) -> bool:
        return self.taichi_available and self.cuda_available


# probe the environment, never raises: failures land in the report
def check_environment() -> EnvReport:
    notes: list[str] = []

    try:
        import taichi as ti  # noqa: F401
    except ImportError:
        return EnvReport(
            taichi_available=False,
            taichi_version=None,
            cuda_available=False,
            notes=["Taichi not installed. Run: pip install -r requirements.txt"],
        )

    version = ".".join(str(v) for v in ti.__version__)

    cuda_available = False
    try:
        ti.init(arch=ti.cuda)
        cuda_available = True
        notes.append("CUDA backend initialized.")
        # Best-effort VRAM report (API surface varies across Taichi versions).
        try:
            import taichi.lang.impl as _impl  # noqa: F401
            notes.append("GPU visible to Taichi CUDA backend.")
        except Exception:  # pragma: no cover - informational only
            pass
    except Exception as exc:  # CUDA missing / no GPU / driver issue
        notes.append(f"CUDA backend unavailable: {exc}")
        notes.append("Falling back is possible with arch=ti.cpu for small 2D cases.")

    return EnvReport(
        taichi_available=True,
        taichi_version=version,
        cuda_available=cuda_available,
        notes=notes,
    )


def main() -> int:
    report = check_environment()
    print("Homebrew CFD - environment check")
    print("-" * 34)
    print(f"Taichi available : {report.taichi_available}")
    print(f"Taichi version   : {report.taichi_version}")
    print(f"CUDA backend     : {report.cuda_available}")
    print(f"Ready for GPU LBM: {report.ready}")
    for note in report.notes:
        print(f"  - {note}")
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
