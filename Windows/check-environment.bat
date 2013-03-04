@Echo off

:: Set the default file directories
set CURRENT=%CD%
set DOWNLOADS=%CD%\..\Downloads
set DEPENDS=%CD%\..\Dependencies
set INCLUDES=%CD%\..\Includes

:: Ensure we're running in a VS environment
msbuild /? > nul
if NOT ERRORLEVEL 0 (
	echo.
	echo Please run this script from a Visual Studio Command Prompt.
	exit /b -1
)

:: Ensure the wget utility is available
set WGET="%CD%\Support\bin\wget.exe"
%WGET% --version 1> nul 2>&1
if ERRORLEVEL 9009 (
	echo.
	echo Cannot execute wget. Aborting.
	exit /b -1
)

:: Ensure the unzip utility is available
set UNZIPBIN="%CD%\Support\bin\7za.exe"
%UNZIPBIN% > nul
if ERRORLEVEL 9009 (
	echo.
	echo Cannot execute unzip. Aborting.
	exit /b -1
)

:: Allow the user to specify extraction mode options if not already set
if defined FORCE_EXTRACT goto EndExtractChoice
:ForceExtractChoice
set FORCE_EXTRACT=0
echo.
echo **************************************************************************
echo *                         Extract Option                                 *
echo **************************************************************************
echo.
echo Should all sources be decompressed regardless of whether they have already
echo been extracted? If you have issues building the downloaded dependencies,
echo you may wish to select yes here to ensure the issue is not with a corrupt
echo or outdated file.
echo.
echo 0. No (default)
echo 1. Yes
echo.
set /p FORCE_EXTRACT="Selection? "
if %FORCE_EXTRACT% EQU 0 goto EndExtractChoice
if %FORCE_EXTRACT% EQU 1 goto EndExtractChoice
echo Invalid choice
goto ForceExtractChoice
:EndExtractChoice