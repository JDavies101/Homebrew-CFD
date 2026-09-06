"""Run configuration loading for Homebrew CFD.

A run is described by a YAML file (see cases/). Phase 0 keeps this deliberately
small: load YAML into a validated dataclass with sane defaults. Physics fields
will grow as solver phases land.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunConfig:
    """Minimal run configuration.

    Fields are intentionally generic for Phase 0; each solver phase extends this.
    """
    name: str = "unnamed"
    dimensions: int = 2            # 2 (Phase 1) or 3 (Phase 2+)
    resolution: list[int] = field(default_factory=lambda: [256, 128])
    reynolds: float = 100.0        # target Reynolds number
    max_steps: int = 10_000
    backend: str = "cuda"          # "cuda" or "cpu"

    def validate(self) -> None:
        if self.dimensions not in (2, 3):
            raise ValueError(f"dimensions must be 2 or 3, got {self.dimensions}")
        if len(self.resolution) != self.dimensions:
            raise ValueError(
                f"resolution needs {self.dimensions} entries, got {self.resolution}"
            )
        if any(r <= 0 for r in self.resolution):
            raise ValueError(f"resolution entries must be positive, got {self.resolution}")
        if self.reynolds <= 0:
            raise ValueError(f"reynolds must be positive, got {self.reynolds}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.backend not in ("cuda", "cpu"):
            raise ValueError(f"backend must be 'cuda' or 'cpu', got {self.backend}")


def from_dict(data: dict[str, Any]) -> RunConfig:
    """Build a validated RunConfig from a plain dict."""
    known = RunConfig.__dataclass_fields__.keys()
    filtered = {k: v for k, v in data.items() if k in known}
    cfg = RunConfig(**filtered)
    cfg.validate()
    return cfg


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a YAML run configuration."""
    import yaml

    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping, got {type(data).__name__}")
    return from_dict(data)
