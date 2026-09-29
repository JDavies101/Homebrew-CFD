# run log: row append, derived fields, schema migration, field health
import csv
import numpy as np
import pytest
from src.post.run_log import log_run, field_health, RunRecord, FIELDS

def _read_rows(path):
    """
    Read every row of a run-log csv.

    Returns a list of dicts.
    """

    with open(path, encoding="utf-8") as file:
        return list(csv.DictReader(file))

# test 1: rows append with sequential ids, mlups derived only when cells, steps and time are given
def test_log_run_appends_and_derives(tmp_path):

    path = str(tmp_path / "log.csv")
    first_id = log_run(path=path, case="x", cells=1000000, steps=1000, wall_time_s=10)
    second_id = log_run(path=path, case="y")
    rows = _read_rows(path)

    assert (first_id, second_id) == (1, 2)
    assert rows[0]["mlups"] == "100.0"
    assert rows[0]["status"] == "unreviewed"
    assert rows[1]["mlups"] == ""

# test 2: unknown field names are rejected
def test_log_run_rejects_unknown_field(tmp_path):

    with pytest.raises(KeyError):
        log_run(path=str(tmp_path / "log.csv"), not_a_field=1)

# test 3: an old header is migrated, old values kept
def test_old_header_is_migrated(tmp_path):

    path = tmp_path / "log.csv"
    path.write_text("run_id,case,value\n1,old,0.5\n", encoding="utf-8")
    log_run(path=str(path), case="new")
    rows = _read_rows(str(path))

    assert list(rows[0].keys()) == FIELDS
    assert rows[0]["case"] == "old" and rows[0]["value"] == "0.5"
    assert rows[1]["run_id"] == "2"

# test 4: hot cells are located in the right face layer or the interior
def test_field_health_locates_hot_cells():

    velocity = np.zeros((3, 20, 10, 12))
    velocity[0] = 0.05
    velocity[0, 1, 5, 6] = 0.3  # inside the x- boundary band
    velocity[0, 10, 5, 6] = 0.2  # interior
    health = field_health(velocity, np.ones((20, 10, 12)), None, 0.05)

    assert health["max_u"] == 0.3
    assert health["hot_cells"] == 2
    assert "x-:1" in health["hot_where"] and "interior:1" in health["hot_where"]

# test 5: RunRecord with numpy arrays fills grid, cells, error and health
def test_run_record_with_arrays(tmp_path):

    path = str(tmp_path / "log.csv")
    velocity = np.full((2, 8, 8), 0.01)
    run = RunRecord("demo", None, steps=10, u_ref=0.1, path=path, Re=5)
    run.stop()
    run.finish(u=velocity, rho=np.ones((8, 8)), metric="m", value=1.1, reference=1.0)
    row = _read_rows(path)[0]

    assert row["grid"] == "8x8" and row["cells"] == "64"
    assert row["err_pct"] == "10.0"
    assert row["hot_cells"] == "0"
