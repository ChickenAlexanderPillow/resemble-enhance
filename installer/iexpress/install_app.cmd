@echo off
setlocal

set "APP_NAME=Resemble Enhance"
set "APP_ID=ResembleEnhanceGUI"
set "APP_EXE=ResembleEnhanceGUI.exe"
set "INSTALL_DIR=%LOCALAPPDATA%\ResembleEnhanceGUI"
set "PAYLOAD_ZIP=%~dp0ResembleEnhanceGUI.zip"
set "UNINSTALL_SRC=%~dp0uninstall_app.cmd"
set "UNINSTALL_DST=%INSTALL_DIR%\uninstall.cmd"
set "START_MENU_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Resemble Enhance"

if not exist "%PAYLOAD_ZIP%" (
  echo Missing payload: %PAYLOAD_ZIP%
  exit /b 1
)

if not exist "%UNINSTALL_SRC%" (
  echo Missing uninstaller template: %UNINSTALL_SRC%
  exit /b 1
)

if exist "%INSTALL_DIR%" (
  taskkill /IM "%APP_EXE%" /F >nul 2>&1
  rmdir /S /Q "%INSTALL_DIR%" >nul 2>&1
)

mkdir "%INSTALL_DIR%" >nul 2>&1
if errorlevel 1 (
  echo Failed to create install directory.
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -LiteralPath '%PAYLOAD_ZIP%' -DestinationPath '%INSTALL_DIR%' -Force" >nul
if errorlevel 1 (
  echo Failed to extract application payload.
  exit /b 1
)

copy /Y "%UNINSTALL_SRC%" "%UNINSTALL_DST%" >nul
if errorlevel 1 (
  echo Failed to install uninstaller.
  exit /b 1
)

set "PS1=%TEMP%\resemble_install_%RANDOM%.ps1"
>"%PS1%" echo $ErrorActionPreference = 'Stop'
>>"%PS1%" echo $target = Join-Path $env:LOCALAPPDATA 'ResembleEnhanceGUI\ResembleEnhanceGUI.exe'
>>"%PS1%" echo $uninstall = Join-Path $env:LOCALAPPDATA 'ResembleEnhanceGUI\uninstall.cmd'
>>"%PS1%" echo $desktop = [Environment]::GetFolderPath('Desktop')
>>"%PS1%" echo $startMenuDir = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\Resemble Enhance'
>>"%PS1%" echo New-Item -ItemType Directory -Force -Path $startMenuDir ^| Out-Null
>>"%PS1%" echo $shell = New-Object -ComObject WScript.Shell
>>"%PS1%" echo foreach($lnkPath in @((Join-Path $desktop 'Resemble Enhance.lnk'), (Join-Path $startMenuDir 'Resemble Enhance.lnk'))){ $s = $shell.CreateShortcut($lnkPath); $s.TargetPath = $target; $s.WorkingDirectory = Split-Path $target; $s.Save() }
>>"%PS1%" echo $u = $shell.CreateShortcut((Join-Path $startMenuDir 'Uninstall Resemble Enhance.lnk')); $u.TargetPath = $uninstall; $u.WorkingDirectory = Split-Path $uninstall; $u.Save()
>>"%PS1%" echo $key = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\ResembleEnhanceGUI'
>>"%PS1%" echo New-Item -Path $key -Force ^| Out-Null
>>"%PS1%" echo Set-ItemProperty -Path $key -Name DisplayName -Value 'Resemble Enhance'
>>"%PS1%" echo Set-ItemProperty -Path $key -Name Publisher -Value 'Resemble Enhance'
>>"%PS1%" echo Set-ItemProperty -Path $key -Name InstallLocation -Value (Join-Path $env:LOCALAPPDATA 'ResembleEnhanceGUI')
>>"%PS1%" echo Set-ItemProperty -Path $key -Name DisplayIcon -Value $target
>>"%PS1%" echo Set-ItemProperty -Path $key -Name UninstallString -Value ('cmd.exe /c ""' + $uninstall + '""')
>>"%PS1%" echo Set-ItemProperty -Path $key -Name QuietUninstallString -Value ('cmd.exe /c ""' + $uninstall + '""')
>>"%PS1%" echo Set-ItemProperty -Path $key -Name NoModify -Type DWord -Value 1
>>"%PS1%" echo Set-ItemProperty -Path $key -Name NoRepair -Type DWord -Value 1
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" >nul 2>&1
del /q "%PS1%" >nul 2>&1

start "" "%INSTALL_DIR%\%APP_EXE%"
exit /b 0
