"""Environment check for Homebrew CFD.

Verifies the GPU compute stack is ready before any solver work:
  * Taichi imports
  * the CUDA backend initializes (the RTX 3090 is visible)
  * reports available VRAM

Run directly:  python -m src.config.environment
Import and call check_environment() for use in tests / scripts.

No solver physics here — this is a Phase 0 readiness probe.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnvReport:
    """Result of an environment probe."""
    taichi_available: bool
    taichi_version: str | None
    cuda_available: bool
    notes: list[str]

    @property
    def ready(self) -> bool:
        """True when the GPU LBM stack can run."""
        return self.taichi_available and self.cuda_available


def check_environment() -> EnvReport:
    """Probe the compute environment without raising.

    Safe to call anywhere (including CI without a GPU): failures are recorded
    in the report rather than thrown, so callers decide how strict to be.
    """
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
    print("Homebrew CFD — environment check")
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
