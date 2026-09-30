# scaffold checks: packages import, env probe runs, config validates
import subprocess
import sys
from pathlib import Path
import pytest
from src.config import loader

cases_directory = Path(__file__).resolve().parent.parent / "cases"
# test 1: core packages import without error
def test_packages_import():

    import src.lbm  # noqa: F401
    import src.geometry  # noqa: F401
    import src.turbulence  # noqa: F401
    import src.post  # noqa: F401

# test 2: environment probe runs in its own process (its ti.init(cuda) must not re-init Taichi under the test session)
def test_environment_check_runs():

    code = (
        "from src.config.environment import check_environment\n"
        "result = check_environment()\n"
        "assert isinstance(result.ready, bool) and isinstance(result.notes, list)\n"
    )
    process = subprocess.run([sys.executable, "-c", code], cwd=cases_directory.parent, capture_output=True, text=True)

    assert process.returncode == 0, process.stderr

# test 3: default RunConfig passes its own validation (must not raise)
def test_config_defaults_valid():

    config = loader.RunConfig()
    config.validate()

# test 4: 4 dimensions are rejected
def test_config_rejects_bad_dimensions():

    with pytest.raises(ValueError):
        loader.from_dict({"dimensions": 4, "resolution": [1, 2, 3, 4]})

# test 5: resolution length must match dimensions
def test_config_rejects_resolution_mismatch():

    with pytest.raises(ValueError):
        loader.from_dict({"dimensions": 2, "resolution": [128, 64, 32]})

# test 6: the shipped example YAML parses and validates
def test_example_case_loads():

    config = loader.load_config(cases_directory / "example_poiseuille.yaml")

    assert config.name == "poiseuille_2d"
    assert config.dimensions == 2
    assert config.resolution == [256, 64]
