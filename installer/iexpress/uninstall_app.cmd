@echo off
setlocal

set "APP_EXE=ResembleEnhanceGUI.exe"
set "INSTALL_DIR=%LOCALAPPDATA%\ResembleEnhanceGUI"
set "START_MENU_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Resemble Enhance"
set "DESKTOP_LNK=%USERPROFILE%\Desktop\Resemble Enhance.lnk"
set "UNINSTALL_KEY=HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\ResembleEnhanceGUI"

taskkill /IM "%APP_EXE%" /F >nul 2>&1
del /q "%DESKTOP_LNK%" >nul 2>&1
rmdir /S /Q "%START_MENU_DIR%" >nul 2>&1
reg delete "%UNINSTALL_KEY%" /f >nul 2>&1

set "CLEANUP_CMD=%TEMP%\resemble_cleanup_%RANDOM%.cmd"
>"%CLEANUP_CMD%" echo @echo off
>>"%CLEANUP_CMD%" echo ping 127.0.0.1 -n 3 ^>nul
>>"%CLEANUP_CMD%" echo rmdir /S /Q "%INSTALL_DIR%"
>>"%CLEANUP_CMD%" echo del /Q "%%~f0"

start "" /min cmd /c "%CLEANUP_CMD%"
exit /b 0
