# regression suite: run each listed case in its own solver 
# process and compare its result to the logged reference
import argparse
import json
import subprocess
import sys
from pathlib import Path
from src.run.run_folder import new_run_folder_path, write_json_atomic

def load_suite(path):
    """
    Read the suite manifest.

    Returns the list of case entries.
    """

    suite = json.loads(Path(path).read_text())

    return suite["cases"]

def run_entry(entry, suite_folder, backend):
    """
    Solve one case in a fresh solver process (one Taichi runtime per process) and read its result.

    Returns the result.json contents, or a crashed placeholder when the solver left none.
    """

    folder = suite_folder / entry["name"]
    command = [sys.executable, "-m", "src.run", entry["case"], "--backend", backend, "--run-folder", str(folder)]
    subprocess.run(command)

    result_path = folder / "result.json"
    result = {"status": "crashed", "case_hash": "", "parts": {}}
    if result_path.exists():
        result = json.loads(result_path.read_text())

    return result

def compare(entry, result):
    """
    Check one run against its entry: finished status, pinned case hash (if any) and every reference value.

    Returns a list of (label, value, reference, tolerance, passed).
    """

    rows = [("status", result["status"], "finished", None, result["status"] == "finished")]
    if "case_hash" in entry:
        rows.append(("case", result["case_hash"][:12], entry["case_hash"][:12], None, result["case_hash"] == entry["case_hash"]))

    # a missing part or axis reads as NAN which fails every comparison
    for check in entry["checks"]:
        value = result["parts"].get(check["part"], {}).get(check["axis"], {}).get("mean", float("nan"))
        passed = abs(value - check["reference"]) <= check["tolerance"]
        rows.append((f"{check['part']} C_{check['axis']}", value, check["reference"], check["tolerance"], passed))

    return rows

def main():
    """
    Run the suite for one backend, print a pass / fail line per check, write report.json, exit 1 on any failure.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="cases/regression/suite.json")
    parser.add_argument("--backend", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--runs", default="runs/regression") # parent of the dated suite folder
    parser.add_argument("--only", default=None) # run one case by name
    options = parser.parse_args()

    entries = [entry for entry in load_suite(options.suite) if entry["backend"] == options.backend]
    if options.only is not None:
        entries = [entry for entry in entries if entry["name"] == options.only]

    suite_folder = new_run_folder_path(options.runs, options.backend)
    suite_folder.mkdir(parents=True)

    # each case in its own process then its checks
    report = []
    for entry in entries:
        result = run_entry(entry, suite_folder, options.backend)
        rows = compare(entry, result)
        report.append({"name": entry["name"], "reference_run": entry["reference_run"], "case_hash": result["case_hash"], "checks": rows, "passed": all(row[4] for row in rows)})
        for label, value, reference, tolerance, passed in rows:
            print(f"{'PASS' if passed else 'FAIL'} {entry['name']} {label}: {value} (reference {reference}, tolerance {tolerance})")

    write_json_atomic(suite_folder / "report.json", report)
    failed = [item["name"] for item in report if not item["passed"]]
    print(f"{len(report) - len(failed)} of {len(report)} passed" + (f"; failed: {','.join(failed)}" if failed else ""))

    sys.exit(1 if failed or not report else 0)

if __name__ == "__main__":
    main()