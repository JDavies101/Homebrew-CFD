# build the app folder (PyInstaller) and the installer (Inno Setup); version comes from src/__init__.py
$ErrorActionPreference = "Stop"
$version = python -c "from src import __version__; print(__version__)"
pyinstaller --noconfirm --clean packaging/homebrew_cfd.spec
& "C:\Program Files\Inno Setup 7\ISCC.exe" "/DAppVersion=$version" packaging/installer.iss
Write-Host "Built HomebrewCFD $version -> dist\HomebrewCFD and dist\installer"