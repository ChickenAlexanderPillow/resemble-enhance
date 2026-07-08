@echo off
setlocal

set "APP_NAME=Resemble Enhance"
set "APP_EXE=ResembleEnhanceGUI.exe"
set "INSTALL_DIR=%LOCALAPPDATA%\ResembleEnhanceGUI"

if not exist "%~dp0%APP_EXE%" (
  echo Missing payload: %APP_EXE%
  exit /b 1
)

if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
copy /Y "%~dp0%APP_EXE%" "%INSTALL_DIR%\%APP_EXE%" >nul
if errorlevel 1 (
  echo Failed to copy app executable.
  exit /b 1
)

set "PS1=%TEMP%\resemble_shortcuts_%RANDOM%.ps1"
>"%PS1%" echo $ErrorActionPreference = 'Stop'
>>"%PS1%" echo $target = Join-Path $env:LOCALAPPDATA 'ResembleEnhanceGUI\ResembleEnhanceGUI.exe'
>>"%PS1%" echo $shell = New-Object -ComObject WScript.Shell
>>"%PS1%" echo $desktop = [Environment]::GetFolderPath('Desktop')
>>"%PS1%" echo $startMenu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
>>"%PS1%" echo foreach($lnkPath in @((Join-Path $desktop 'Resemble Enhance.lnk'), (Join-Path $startMenu 'Resemble Enhance.lnk'))){ $s = $shell.CreateShortcut($lnkPath); $s.TargetPath = $target; $s.WorkingDirectory = Split-Path $target; $s.Save() }
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" >nul 2>&1
del /q "%PS1%" >nul 2>&1

start "" "%INSTALL_DIR%\%APP_EXE%"
exit /b 0
