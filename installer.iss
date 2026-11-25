#define AppName "VC Translator"
#define AppVer GetEnv("APP_VERSION")

[Setup]
; Keep this AppId the same forever: it is what lets a new installer update the old one in place.
AppId={{B6F1D6C2-3E57-4D9B-9A55-6C1D2E8F7A01}
AppName={#AppName}
AppVersion={#AppVer}
DefaultDirName={autopf}\VCTranslator
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=dist_installer
OutputBaseFilename=VCTranslator-Setup
Compression=lzma2
SolidCompression=yes
CloseApplications=yes
UninstallDisplayName={#AppName}

[Files]
Source: "dist\VCTranslator\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\VCTranslator.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\VCTranslator.exe"

[Run]
; silent update (started by the app): relaunch automatically
Filename: "{app}\VCTranslator.exe"; Flags: nowait runasoriginaluser; Check: WizardSilent
; normal install: offer to launch
Filename: "{app}\VCTranslator.exe"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
