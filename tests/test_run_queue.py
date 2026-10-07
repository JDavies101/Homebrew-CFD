# tests for the run queue model
from pathlib import Path
from src.run.case_file import load_case_file
from app.run_queue import RunQueue

case_file = load_case_file("cases/templates/sphere.json")

# test 1: add snapshots the case and later edits do not reach the snapshot
def test_add_snapshots_case(tmp_path):

    queue = RunQueue(tmp_path)
    entry = queue.add(case_file, tmp_path / "runs")
    original_name = case_file.name
    case_file.name = "edited"
    snapshot = load_case_file(entry["case"])
    case_file.name = original_name

    assert entry["state"] == "pending"
    assert snapshot.name == original_name

# test 2: next_pending skips entries that are not pending
def test_next_pending_order(tmp_path):

    queue = RunQueue(tmp_path)
    first = queue.add(case_file, tmp_path)
    second = queue.add(case_file, tmp_path)
    queue.mark(first["id"], "finished")

    assert queue.next_pending()["id"] == second["id"]

# test 3: a running entry reloads as interrupted and is not picked up again
def test_reload_running_is_interrupted(tmp_path):

    queue = RunQueue(tmp_path)
    entry = queue.add(case_file, tmp_path)
    queue.mark(entry["id"], "running")
    reloaded = RunQueue(tmp_path)

    assert reloaded.find(entry["id"])["state"] == "interrupted"
    assert reloaded.next_pending() is None

# test 4: move reorders and clamps at the ends
def test_move(tmp_path):

    queue = RunQueue(tmp_path)
    first = queue.add(case_file, tmp_path)
    second = queue.add(case_file, tmp_path)
    queue.move(second["id"], -1)
    queue.move(second["id"], -1)

    assert [entry["id"] for entry in queue.entries] == [second["id"], first["id"]]

# test 5: remove deletes the snapshot but never a running entry
def test_remove(tmp_path):

    queue = RunQueue(tmp_path)
    pending = queue.add(case_file, tmp_path)
    running = queue.add(case_file, tmp_path)
    queue.mark(running["id"], "running")
    queue.remove(pending["id"])
    queue.remove(running["id"])

    assert len(queue.entries) == 1
    assert queue.entries[0]["id"] == running["id"]
    assert not (tmp_path / f"{pending['id']}_sphere.json").exists()

# test 6: move passes over finished and running entries, which keep their place
def test_move_skips_inactive(tmp_path):

    queue = RunQueue(tmp_path)
    first = queue.add(case_file, tmp_path)
    done = queue.add(case_file, tmp_path)
    second = queue.add(case_file, tmp_path)
    queue.mark(done["id"], "finished")
    queue.move(second["id"], -1)
    queue.move(done["id"], 1)

    assert [entry["id"] for entry in queue.entries] == [second["id"], done["id"], first["id"]]

# test 7: study and value are stored and study_entries returns that study in queue order
def test_study_entries(tmp_path):

    queue = RunQueue(tmp_path)
    study_path = str((tmp_path / "studies" / "sweep").resolve())
    other_path = str((tmp_path / "other" / "sweep").resolve())
    first = queue.add(case_file, tmp_path, study=study_path, value=1.5)
    queue.add(case_file, tmp_path)
    queue.add(case_file, tmp_path, study=other_path, value=9.0)
    second = queue.add(case_file, tmp_path, study=study_path, value=2.5)
    reloaded = RunQueue(tmp_path)

    assert [entry["id"] for entry in reloaded.study_entries(study_path)] == [first["id"], second["id"]]
    assert reloaded.find(second["id"])["value"] == 2.5
    assert reloaded.find(first["id"])["study"] == study_path

# test 8: project_entries returns only the entries of that runs folder, in queue order
def test_project_entries(tmp_path):

    queue = RunQueue(tmp_path)
    first = queue.add(case_file, tmp_path / "project_a" / "runs")
    queue.add(case_file, tmp_path / "project_b" / "runs")
    second = queue.add(case_file, tmp_path / "project_a" / "runs")

    assert [entry["id"] for entry in queue.project_entries(tmp_path / "project_a" / "runs")] == [first["id"], second["id"]]
    assert len(queue.project_entries(tmp_path / "project_c" / "runs")) == 0

# test 9: next_pending with a runs folder skips pending entries of other projects
def test_next_pending_by_runs_directory(tmp_path):

    queue = RunQueue(tmp_path)
    queue.add(case_file, tmp_path / "project_a" / "runs")
    wanted = queue.add(case_file, tmp_path / "project_b" / "runs")

    assert queue.next_pending(tmp_path / "project_b" / "runs")["id"] == wanted["id"]
    assert queue.next_pending(tmp_path / "project_c" / "runs") is None

# test 10: clear_pending drops this project's pending entries and snapshots; running, finished and other projects stay;
# active_entries lists only pending and running
def test_clear_pending_and_active_entries(tmp_path):

    queue = RunQueue(tmp_path)
    runs_directory = tmp_path / "project_a" / "runs"
    pending = queue.add(case_file, runs_directory)
    done = queue.add(case_file, runs_directory)
    running = queue.add(case_file, runs_directory)
    other = queue.add(case_file, tmp_path / "project_b" / "runs")
    queue.mark(done["id"], "finished")
    queue.mark(running["id"], "running")
    active_before = [entry["id"] for entry in queue.active_entries(runs_directory)]
    queue.clear_pending(runs_directory)

    assert active_before == [pending["id"], running["id"]]
    assert [entry["id"] for entry in queue.entries] == [done["id"], running["id"], other["id"]]
    assert not Path(pending["case"]).exists()
    assert Path(done["case"]).exists()
    assert Path(running["case"]).exists()
    assert Path(other["case"]).exists()
