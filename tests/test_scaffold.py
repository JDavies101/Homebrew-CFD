"""Phase 0 scaffold tests.

These assert the project skeleton is wired up correctly. No solver physics yet;
physics validation (Tier B) arrives with each solver phase.
"""
from pathlib import Path

import pytest

from src.config import environment, loader

CASES_DIR = Path(__file__).resolve().parent.parent / "cases"


def test_packages_import():
    """Core packages import without error."""
    import src.lbm  # noqa: F401
    import src.geometry  # noqa: F401
    import src.turbulence  # noqa: F401
    import src.post  # noqa: F401


def test_environment_check_runs():
    """Environment probe returns a report and never raises (CI-safe)."""
    report = environment.check_environment()
    assert isinstance(report.ready, bool)
    assert isinstance(report.notes, list)


def test_config_defaults_valid():
    """Default RunConfig passes its own validation."""
    cfg = loader.RunConfig()
    cfg.validate()  # should not raise


def test_config_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        loader.from_dict({"dimensions": 4, "resolution": [1, 2, 3, 4]})


def test_config_rejects_resolution_mismatch():
    with pytest.raises(ValueError):
        loader.from_dict({"dimensions": 2, "resolution": [128, 64, 32]})


def test_example_case_loads():
    """The shipped example YAML parses and validates."""
    cfg = loader.load_config(CASES_DIR / "example_poiseuille.yaml")
    assert cfg.name == "poiseuille_2d"
    assert cfg.dimensions == 2
    assert cfg.resolution == [256, 64]
