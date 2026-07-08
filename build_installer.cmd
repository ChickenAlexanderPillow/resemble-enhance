@echo off
setlocal

set "ROOT=%~dp0"
set "DIST_DIR=%ROOT%dist\ResembleEnhanceGUI"
set "ISS_FILE=%ROOT%installer\inno\ResembleEnhanceGUI.iss"
set "OUTPUT_DIR=%ROOT%dist\installer"
set "ISCC="

if not exist "%DIST_DIR%\ResembleEnhanceGUI.exe" (
  echo Missing app bundle: %DIST_DIR%\ResembleEnhanceGUI.exe
  echo Build it first with build_enhancer_gui.cmd
  exit /b 1
)

if not exist "%ISS_FILE%" (
  echo Missing Inno Setup script: %ISS_FILE%
  exit /b 1
)

for %%P in (
  "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
  "%ProgramFiles%\Inno Setup 6\ISCC.exe"
  "%ProgramFiles(x86)%\Inno Setup 5\ISCC.exe"
  "%ProgramFiles%\Inno Setup 5\ISCC.exe"
) do (
  if exist "%%~P" set "ISCC=%%~P"
)

if not defined ISCC (
  for /f "delims=" %%P in ('where ISCC.exe 2^>nul') do (
    if not defined ISCC set "ISCC=%%P"
  )
)

if not defined ISCC (
  echo Inno Setup compiler not found.
  echo Install Inno Setup 6 from https://jrsoftware.org/isinfo.php and rerun this script.
  exit /b 1
)

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
del /q "%OUTPUT_DIR%\ResembleEnhanceGUI-Setup*.exe" >nul 2>&1
del /q "%OUTPUT_DIR%\ResembleEnhanceGUI-Setup*.bin" >nul 2>&1
del /q "%OUTPUT_DIR%\~ResembleEnhanceGUI-Setup.DDF" >nul 2>&1

for /f "usebackq delims=" %%S in (`powershell -NoProfile -Command "$s=(Get-ChildItem '%DIST_DIR%' -Recurse -File | Measure-Object Length -Sum).Sum; [math]::Round($s/1GB,2)"`) do set "DIST_GB=%%S"
echo App bundle size: %DIST_GB% GB
echo Inno disk spanning is enabled. Large builds will produce one setup EXE plus one or more BIN files.

echo Building installer with:
echo %ISCC%
"%ISCC%" "%ISS_FILE%"
if errorlevel 1 (
  echo Inno Setup packaging failed.
  exit /b 1
)

echo.
echo Installer output created in:
echo %OUTPUT_DIR%
echo.
echo Send every ResembleEnhanceGUI-Setup*.exe and ResembleEnhanceGUI-Setup*.bin file together.
echo The user launches the EXE; the BIN files must stay beside it.
exit /b 0
