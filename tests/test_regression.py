# tests for the regression-suite comparison
import math
from src.run.regression import compare

entry = {"name": "case", "case_hash": "abc123", "checks": [{"part": "wing", "axis": "y", "reference": -0.3368, "tolerance": 0.0004}]}

def result(status="finished", value=-0.3368, case_hash="abc123", part="wing"):
    """
    A result.json-shaped dictionary with one part.

    Returns the dictionary.
    """

    return {"status": status, "case_hash": case_hash, "parts": {part: {"y": {"mean": value}}}}

# test 1: a run inside the tolerance passes every row
def test_within_tolerance_passes():

    rows = compare(entry, result(value=-0.3371))

    assert len(rows) == 3
    assert all(row[4] for row in rows)

# test 2: just outside the tolerance fails
def test_outside_tolerance_fails():

    rows = compare(entry, result(value=-0.3373))

    assert rows[2][4] is False

# test 3: a missing part reads as NaN and fails
def test_missing_part_fails():

    rows = compare(entry, result(part="renamed"))

    assert math.isnan(rows[2][1])
    assert rows[2][4] is False

# test 4: an unfinished run fails on status even with the right value
def test_blow_up_fails():

    rows = compare(entry, result(status="blow_up"))

    assert rows[0][4] is False
    assert rows[2][4] is True

# test 5: a changed case fails on the pinned hash
def test_case_hash_mismatch_fails():

    rows = compare(entry, result(case_hash="def456"))

    assert rows[1][4] is False