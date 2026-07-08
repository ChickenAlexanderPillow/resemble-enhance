@echo off
setlocal

set "ROOT=%~dp0"
set "DIST_EXE=%ROOT%dist\ResembleEnhanceGUI.exe"
set "IEXPRESS_DIR=%ROOT%installer\iexpress"
set "PAYLOAD_EXE=%IEXPRESS_DIR%\ResembleEnhanceGUI.exe"
set "SED_FILE=%IEXPRESS_DIR%\ResembleEnhanceGUI.sed"
set "OUTPUT_SETUP=%ROOT%dist\installer\ResembleEnhanceGUI-Setup.exe"

if not exist "%DIST_EXE%" (
  echo Missing dist executable: %DIST_EXE%
  echo Build it first with build_enhancer_gui.cmd
  exit /b 1
)

if not exist "%IEXPRESS_DIR%" mkdir "%IEXPRESS_DIR%"
if not exist "%ROOT%dist\installer" mkdir "%ROOT%dist\installer"

copy /Y "%DIST_EXE%" "%PAYLOAD_EXE%" >nul
if errorlevel 1 (
  echo Failed to stage payload executable.
  exit /b 1
)

(
echo [Version]
echo Class=IEXPRESS
echo SEDVersion=3
echo [Options]
echo PackagePurpose=InstallApp
echo ShowInstallProgramWindow=0
echo HideExtractAnimation=1
echo UseLongFileName=1
echo InsideCompressed=0
echo CAB_FixedSize=0
echo CAB_ResvCodeSigning=0
echo RebootMode=N
echo InstallPrompt=
echo DisplayLicense=
echo FinishMessage=Resemble Enhance has been installed.
echo TargetName=%OUTPUT_SETUP%
echo FriendlyName=Resemble Enhance Setup
echo AppLaunched=install_app.cmd
echo PostInstallCmd=^<None^>
echo AdminQuietInstCmd=install_app.cmd
echo UserQuietInstCmd=install_app.cmd
echo SourceFiles=SourceFiles
echo [SourceFiles]
echo SourceFiles0=%IEXPRESS_DIR%\
echo [SourceFiles0]
echo %%FILE0%%=
echo %%FILE1%%=
echo [Strings]
echo FILE0=install_app.cmd
echo FILE1=ResembleEnhanceGUI.exe
) > "%SED_FILE%"

iexpress /N /Q "%SED_FILE%"
if errorlevel 1 (
  echo IExpress packaging failed.
  exit /b 1
)

echo Installer created:
echo %OUTPUT_SETUP%
exit /b 0
