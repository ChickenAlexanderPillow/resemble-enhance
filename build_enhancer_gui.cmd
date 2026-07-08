@echo off
setlocal

set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe

echo Ensuring PyInstaller is installed...
"%PY%" -m pip show pyinstaller >nul 2>&1 || "%PY%" -m pip install -q pyinstaller

echo Ensuring ffmpeg binaries are vendored...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0installer\download_ffmpeg.ps1"
if errorlevel 1 (
  echo Failed to vendor ffmpeg.
  exit /b 1
)

echo Building ResembleEnhanceGUI app bundle...
"%PY%" -m PyInstaller --noconfirm ResembleEnhanceGUI.spec
if errorlevel 1 exit /b 1

echo Build complete. See dist\ResembleEnhanceGUI
pause

endlocal
