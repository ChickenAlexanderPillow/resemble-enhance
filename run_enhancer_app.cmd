@echo off
setlocal EnableDelayedExpansion

REM Ensure we run from the repo root even if launched from Explorer
pushd %~dp0
title Resemble Enhance App
echo Bootstrapping Resemble Enhance App launcher...

set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe
echo Using Python: %PY%

REM Ensure Flask is available; install quietly if missing
"%PY%" -c "import flask" 1>nul 2>nul
if errorlevel 1 (
  echo Installing Flask into environment...
  "%PY%" -m pip install --disable-pip-version-check -q Flask>=2.3.0
)

REM Pick a free port, starting at 5137 and incrementing until available
set BASEPORT=5137
set PORT=%BASEPORT%
set ATTEMPTS=0
echo Detecting free port starting at %BASEPORT%...
:checkport
REM Look only for LISTENING sockets on this port
netstat -ano | findstr /R ":%PORT%[ ]" | findstr /C:"LISTENING" >nul 2>&1
if %ERRORLEVEL%==0 (
  set /a PORT+=1
  set /a ATTEMPTS+=1
  if !ATTEMPTS! GEQ 20 (
    echo Could not find a free port near %BASEPORT% after !ATTEMPTS! tries. Proceeding with port %PORT%.
    goto startapp
  )
  goto checkport
)

:startapp

set ENHANCER_APP_PORT=%PORT%
echo Starting Resemble Enhance App on http://127.0.0.1:%PORT%
echo (Tip) Health check: http://127.0.0.1:%PORT%/api/ping
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
if not exist enhancer_app\app.py (
  echo ERROR: enhancer_app\app.py not found. Are you running from the project root?
  pause
  goto :eof
)
echo Launching server...
"%PY%" -u enhancer_app\app.py
set EXITCODE=%ERRORLEVEL%
if not %EXITCODE%==0 (
  echo.
  echo The app exited with code %EXITCODE%.
  echo If this window closes too fast, run this file from a terminal.
  pause
)

popd
endlocal
