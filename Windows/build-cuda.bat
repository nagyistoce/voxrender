@Echo off

:: Issue a message about the project and the use of 
:: wget for fetching the project source+dependencies
echo.
echo **************************************************************************
echo *                             Startup                                    *
echo **************************************************************************
echo.
echo NOTICE: You are about to download and download CUDA version 5.0
echo         from the NVidia website at http://developer.download.nvidia.com
echo.
echo This script will use the following pre-built binary to help build the project:
echo  1: GNU wget.exe    http://gnuwin32.sourceforge.net/packages/wget.htm
echo.

:: Set execution environment
call check-environment.bat

:: Prompt the user for some download information
echo.
echo **************************************************************************
echo *                        NVidia's CUDA SDK                               *
echo **************************************************************************
:Choice
echo.
echo Please select which Cuda SDK you wish to use:
echo.
echo 1. NVIDIA CUDA ToolKit for Windows 8.1
echo 2. NVIDIA CUDA ToolKit for Windows Vista,7,8.0
echo N. I have already installed an NVIDIA CUDA 5.5 Toolkit
echo.
set CUDA_CHOICE=0
set /p CUDA_CHOICE="Selection? "
IF %CUDA_CHOICE% EQU 1 goto CUDA_81
IF %CUDA_CHOICE% EQU 2 goto CUDA_80
IF /i "%CUDA_CHOICE%" == "N" goto Finished
echo Invalid choice
goto Choice

:CUDA_80
set CUDA_NAME=NVIDIA CUDA ToolKit 5.5 for Win Vista,7,8.0
set CUDA_URL=http://developer.download.nvidia.com/compute/cuda/5_5/rel/installers/
set CUDA_PKG=cuda_5.5.31_win8.1_general_x64.exe
goto Install

:CUDA_81
set CUDA_NAME=NVIDIA CUDA ToolKit 5.5 for Win 8.1
set CUDA_URL=http://developer.download.nvidia.com/compute/cuda/5_5/rel/installers/
set CUDA_PKG=cuda_5.5.20_winvista_win7_win8_general_64.exe
goto Install

:Install
if not exist %DOWNLOADS%\%CUDA_PKG% (
	echo.
	echo **************************************************************************
	echo *               Downloading %CUDA_NAME%                                  *
	echo **************************************************************************
	%WGET% %CUDA_URL%%CUDA_PKG% -O %DOWNLOADS%\%CUDA_PKG%
	if errorlevel 1 (
		echo.
		echo Download failed. Are you connected to the internet?
		exit /b -1
	)
)
echo.

:Finished
echo.
echo **************************************************************************
echo *                             Finished                                   *
echo **************************************************************************

if not defined CUDA_PKG goto eof
echo.
echo  The SDK installer will now be launched. You may need to perform a system
echo  restart in order to ensure environment variables are available for use
echo  by CMake.
echo.

start /WAIT "" %DOWNLOADS%\%CUDA_PKG%
echo Waiting for installer.

:eof