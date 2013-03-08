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

:: Allow the user to specify platform if not already set
if defined BUILD_PLATFORM goto EndPlatformChoice
:BuildPlatformChoice
set BUILD_PLATFORM=0
echo.
echo **************************************************************************
echo *                         Platform Option                                *
echo **************************************************************************
echo.
echo Are you building 32 or 64 bit binaries? (Note this depends on whether you
echo are running a standard or x64 Visual Studio command prompt)
echo.
echo 0. x86 (default)
echo 1. x64
echo.
set /p BUILD_PLATFORM="Selection? "
if %BUILD_PLATFORM% EQU 0 ( 
    set BUILD_PLATFORM=x86
	goto EndPlatformChoice 
	)
if %BUILD_PLATFORM% EQU 1 (
    set BUILD_PLATFORM=x64
    goto EndPlatformChoice
	)
echo Invalid choice
goto BuildPlatformChoice
:EndPlatformChoice