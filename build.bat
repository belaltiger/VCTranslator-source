@echo off
cd /d "%~dp0"
echo === Installing build tools ===
python -m pip install -r requirements.txt
if errorlevel 1 goto err
for /f %%v in ('python -c "from version import __version__ as v;print(v)"') do set APP_VERSION=%%v
echo === Building version %APP_VERSION% ===
python -m PyInstaller --noconfirm --clean --onedir --noconsole --name VCTranslator --hidden-import pyaudiowpatch --collect-all customtkinter --collect-all faster_whisper --collect-all ctranslate2 --collect-all onnxruntime --hidden-import comtypes --collect-all av --collect-all pycaw --collect-all comtypes --collect-all proctap app.py
if errorlevel 1 goto err
set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
"%ISCC%" installer.iss
if errorlevel 1 goto err
echo.
echo DONE: dist_installer\VCTranslator-Setup.exe
pause
exit /b 0
:err
echo.
echo BUILD FAILED - copy the messages above and send them to Claude.
pause
exit /b 1
