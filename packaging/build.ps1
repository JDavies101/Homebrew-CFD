# build the app folder (PyInstaller) and/or the installer (Inno Setup); version comes from src/__init__.py
param([ValidateSet("all", "app", "installer")] [string]$stage = "all")
$ErrorActionPreference = "Stop"
$version = python -c "from src import __version__; print(__version__)"
if ($stage -ne "installer") {
    pyinstaller --noconfirm --clean packaging/homebrew_cfd.spec
}
if ($stage -ne "app") {
    $inno_compiler = if ($env:INNO_COMPILER) { $env:INNO_COMPILER } else { "C:\Program Files\Inno Setup 7\ISCC.exe" }
    & $inno_compiler "/DAppVersion=$version" packaging/installer.iss
}
Write-Host "Built HomebrewCFD $version ($stage)"