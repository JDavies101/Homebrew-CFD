# projects: one folder per case in the workspace (case file, geometry, runs, exports), portable as a whole
import filecmp
import re
import shutil
from pathlib import Path

project_subfolders = ("geometry", "runs", "exports")

def check_project_name(name):
    """
    Raise if a project name cannot be a folder name everywhere (letters, digits, space, _ - . only; not empty).
    """

    if not re.fullmatch(r"[A-Za-z0-9 _.\-]+", name) or name.strip(" .") == "":
        raise ValueError(f"project name {name!r}: use letters, digits, space, _ - . only")
    
def create_project(workspace, name):
    """
    Make a new project folder with its subfolders; an existing name is refused, never merged.

    Returns the case file path inside it (<workspace>/<name>/<name>.json).
    """

    check_project_name(name)
    folder = Path(workspace) / name
    if folder.exists():
        raise FileExistsError(f"a project named {name!r} already exists in {workspace}")
    
    for subfolder in project_subfolders:
        (folder / subfolder).mkdir(parents=True)

    return folder / f"{name}.json"

def is_project(case_path):
    """
    Returns True when the case file sits in a project folder (a runs folder beside it).
    """

    return case_path is not None and (Path(case_path).parent / "runs").is_dir()

def runs_directory_for(case_path, workspace):
    """
    Where runs of a case go: its project's runs folder, or the workspace Runs folder for a loose case file.

    Returns the Path.
    """
    
    if is_project(case_path):
        return Path(case_path).parent / "runs"
    
    return Path(workspace) / "Runs"

def adopt_geometry(case_file, case_path):
    """
    Copy every STL from outside the project into its geometry folder and point the spec at the copy;
    an identical file already there is reused, a different one with the same name gets a numbered name.

    Returns the list of (source, copy) pairs made.
    """

    geometry_folder = (Path(case_path).parent / "geometry").resolve()
    copies = []
    for spec in case_file.geometry:
        if spec.kind != "stl" or not spec.path:
            continue

        source = Path(spec.path).resolve()
        if source.is_relative_to(geometry_folder):
            continue

        # same name, different content: number the copy instead of overwriting
        target = geometry_folder / source.name
        number = 2
        while target.exists() and not filecmp.cmp(source, target, shallow=False):
            target = geometry_folder / f"{source.stem}_{number}{source.suffix}"
            number += 1

        if not target.exists():
            shutil.copy2(source, target)
            copies.append((str(source), str(target)))

        spec.path = str(target)

    return copies