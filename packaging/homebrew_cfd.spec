# PyInstaller build: one windowed app folder; engine and Taichi kept as .py source because Taichi reads kernel source at run time
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

root = Path(SPECPATH).parent
taichi_datas, taichi_binaries, taichi_hidden_imports = collect_all("taichi")

analysis = Analysis([str(root / "packaging" / "launch.py")],
                    pathex=[str(root)],
                    binaries=taichi_binaries,
                    datas=taichi_datas + [(str(root / "cases" / "templates"), "cases/templates"), (str(root / "app" / "icon.png"), "app")],
                    hiddenimports=taichi_hidden_imports + collect_submodules("src") + collect_submodules("app"),
                    module_collection_mode={"src": "py", "taichi": "py"},
                    excludes=["pytest"])
pyz = PYZ(analysis.pure)
exe = EXE(pyz, analysis.scripts, [], exclude_binaries=True, name="HomebrewCFD", console=False, icon=str(root / "packaging" / "icon.ico"))
COLLECT(exe, analysis.binaries, analysis.datas, name="HomebrewCFD")