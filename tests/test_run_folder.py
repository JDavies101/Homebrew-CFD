# run folders: case hash, folder creation, atomic JSON writes
import dataclasses
import json
from src.run.case import Flow, Domain, Turbulence, Timing
from src.run.case_file import CaseFile
from src.run.run_folder import case_hash, create_run_folder, write_json_atomic

def small_case_file(**flow_settings):
    """
    A minimal case file; flow keyword settings override Flow defaults.

    Returns the CaseFile.
    """

    return CaseFile(name="hash_test", flow=Flow(**flow_settings), domain=Domain(), turbulence=Turbulence(), timing=Timing())

# test 1: the hash is stable for equal cases and changes when any setting changes
def test_case_hash_tracks_settings():

    first = case_hash(small_case_file())
    second = case_hash(small_case_file())
    edited = case_hash(small_case_file(reynolds_number=5001.0))
    renamed = case_hash(dataclasses.replace(small_case_file(), name="other"))

    assert first == second
    assert edited != first
    assert renamed != first
    assert len(first) == 64

# test 2: run folders are new directories named after the case
def test_create_run_folder(tmp_path):

    folder = create_run_folder(tmp_path, "wing_ground")

    assert folder.is_dir()
    assert folder.parent == tmp_path
    assert folder.name.endswith("_wing_ground")

# test 3: atomic writes round-trip, overwrite, and leave no temp file behind
def test_write_json_atomic(tmp_path):

    path = tmp_path / "progress.json"
    write_json_atomic(path, {"step": 1})
    write_json_atomic(path, {"step": 2, "status": "running"})

    assert json.loads(path.read_text()) == {"step": 2, "status": "running"}
    assert not (tmp_path / "progress.json.tmp").exists()