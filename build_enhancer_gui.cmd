@echo off
setlocal

set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe

echo Ensuring PyInstaller is installed...
"%PY%" -m pip show pyinstaller >nul 2>&1 || "%PY%" -m pip install -q pyinstaller

echo Building ResembleEnhanceGUI.exe ...
"%PY%" -m PyInstaller --noconfirm ResembleEnhanceGUI.spec

echo Build complete. See the dist folder.
pause

endlocal
