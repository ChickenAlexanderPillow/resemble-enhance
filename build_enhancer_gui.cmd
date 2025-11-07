@echo off
setlocal

set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe

echo Ensuring PyInstaller is installed...
"%PY%" -m pip show pyinstaller >nul 2>&1 || "%PY%" -m pip install -q pyinstaller

echo Building ResembleEnhanceGUI.exe ...
REM Include model repo so the app works offline
set DATAS=resemble_enhance\model_repo;resemble_enhance\model_repo

"%PY%" -m PyInstaller --noconsole --onefile --name ResembleEnhanceGUI ^
  --add-data "%DATAS%" ^
  --hidden-import torchaudio._extension._load_lib ^
  --hidden-import torchaudio._extension._lib ^
  enhancer_gui.py

echo Build complete. See the dist folder.
pause

endlocal
