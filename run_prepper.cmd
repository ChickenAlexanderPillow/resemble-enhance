@echo off
setlocal enabledelayedexpansion

REM Premiere Prepper runner (cmd.exe)
REM Edit these three lines to match your shoot layout.
set INPUT=%CD%\Shoot\2025_11_05
set OUTPUT=%CD%\Shoot\PREPPED
set CAMS=camA camB
set MICS=audio

REM Defaults
set TIMEBASE=25
set DEVICE=cuda
set MAX_OFFSET=2.0
set VAD_AGGR=2
set MIN_SPEECH=0.25
set MIN_SILENCE=0.80
set HEAD_PAD=0.60
set TAIL_PAD=1.00

echo Running Premiere Prepper...
echo INPUT:  %INPUT%
echo OUTPUT: %OUTPUT%
echo CAMS:   %CAMS%
echo MICS:   %MICS%

python premiere_prepper.py ^
  --input "%INPUT%" ^
  --out "%OUTPUT%" ^
  --cams %CAMS% ^
  --mics %MICS% ^
  --timebase %TIMEBASE% ^
  --keep-scratch ^
  --write-mp4 ^
  --device %DEVICE% ^
  --trim-on ^
  --vad-aggr %VAD_AGGR% ^
  --min-speech %MIN_SPEECH% ^
  --min-silence %MIN_SILENCE% ^
  --head-pad %HEAD_PAD% ^
  --tail-pad %TAIL_PAD% ^
  --force-clean ^
  --strict-clean ^
  --global-search ^
  --drift-correct ^
  --max-offset %MAX_OFFSET%

set EXITCODE=%ERRORLEVEL%
echo.
echo Done. Exit code: %EXITCODE%
exit /b %EXITCODE%

