@echo off
setlocal enabledelayedexpansion

REM Resemble Enhance runner (cmd.exe)
REM Uses input_audio -> output_audio by default.

set IN=%CD%\input_audio
set OUT=%CD%\output_audio
set DEVICE=cuda

REM Prefer local venv python if present
set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe

echo Running Resemble Enhance...
echo IN:  %IN%
echo OUT: %OUT%

"%PY%" -m resemble_enhance.enhancer "%IN%" "%OUT%" --denoise_only --device %DEVICE% %*

set EXITCODE=%ERRORLEVEL%
echo.
echo Done. Exit code: %EXITCODE%
exit /b %EXITCODE%

