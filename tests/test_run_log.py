# run log: row append, derived fields, schema migration, field health
import csv
import numpy as np
import pytest
from src.post.run_log import log_run, field_health, RunRecord, FIELDS


def _rows(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def test_log_run_appends_and_derives(tmp_path):
    p = str(tmp_path / "log.csv")
    a = log_run(path=p, case="x", cells=1000000, steps=1000, wall_time_s=10)
    b = log_run(path=p, case="y")
    rows = _rows(p)
    assert (a, b) == (1, 2)
    assert rows[0]["mlups"] == "100.0"
    assert rows[0]["status"] == "unreviewed"
    assert rows[1]["mlups"] == ""


def test_log_run_rejects_unknown_field(tmp_path):
    with pytest.raises(KeyError):
        log_run(path=str(tmp_path / "log.csv"), not_a_field=1)


def test_old_header_is_migrated(tmp_path):
    p = tmp_path / "log.csv"
    p.write_text("run_id,case,value\n1,old,0.5\n", encoding="utf-8")
    log_run(path=str(p), case="new")
    rows = _rows(str(p))
    assert list(rows[0].keys()) == FIELDS
    assert rows[0]["case"] == "old" and rows[0]["value"] == "0.5"
    assert rows[1]["run_id"] == "2"


def test_field_health_locates_hot_cells():
    u = np.zeros((3, 20, 10, 12))
    u[0] = 0.05
    u[0, 1, 5, 6] = 0.3          # inside the x- boundary band
    u[0, 10, 5, 6] = 0.2         # interior
    h = field_health(u, np.ones((20, 10, 12)), None, 0.05)
    assert h["max_u"] == 0.3
    assert h["hot_cells"] == 2
    assert "x-:1" in h["hot_where"] and "interior:1" in h["hot_where"]


def test_run_record_with_arrays(tmp_path):
    p = str(tmp_path / "log.csv")
    u = np.full((2, 8, 8), 0.01)
    run = RunRecord("demo", None, steps=10, u_ref=0.1, path=p, Re=5)
    run.stop()
    run.finish(u=u, rho=np.ones((8, 8)), metric="m", value=1.1, reference=1.0)
    r = _rows(p)[0]
    assert r["grid"] == "8x8" and r["cells"] == "64"
    assert r["err_pct"] == "10.0"
    assert r["hot_cells"] == "0"
