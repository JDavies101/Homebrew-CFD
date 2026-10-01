; Inno Setup installer for the PyInstaller app folder; version passed in by build.ps1 (/DAppVersion=...)
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppName=Homebrew CFD
AppVersion={#AppVersion}
DefaultDirName={autopf}\Homebrew CFD
DefaultGroupName=Homebrew CFD
OutputDir=..\dist\installer
OutputBaseFilename=HomebrewCFD-{#AppVersion}-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\HomebrewCFD.exe

[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "..\dist\HomebrewCFD\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\Homebrew CFD"; Filename: "{app}\HomebrewCFD.exe"
Name: "{autodesktop}\Homebrew CFD"; Filename: "{app}\HomebrewCFD.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\HomebrewCFD.exe"; Description: "Launch Homebrew CFD"; Flags: nowait postinstall skipifsilent