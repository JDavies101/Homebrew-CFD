# build the app folder (PyInstaller) and the installer (Inno Setup); version comes from src/__init__.py
$ErrorActionPreference = "Stop"
$version = python -c "from src import __version__; print(__version__)"
pyinstaller --noconfirm --clean packaging/homebrew_cfd.spec
$inno_compiler = if ($env:INNO_COMPILER) { $env:INNO_COMPILER } else { "C:\Program Files\Inno Setup 7\ISCC.exe" }
& $inno_compiler "/DAppVersion=$version" packaging/installer.iss
Write-Host "Built HomebrewCFD $version -> dist\HomebrewCFD and dist\installer"