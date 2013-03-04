@Echo off

:: This flag identifies the version of CUDA to
:: be downloaded and built, you may change it but
:: it could cause issues with altered ABIs, etc 
:: NOTICE: This version flag does not alter the 
::         download URL due to the odd archive 
::         structure on the NVidia site. 
set CUDA_VER=4.1.28

:: Issue a message about the project and the use of 
:: wget for fetching the project source+dependencies
echo.
echo **************************************************************************
echo *                             Startup                                    *
echo **************************************************************************
echo.
echo NOTICE: You are about to download and download CUDA version %CUDA_VER%
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
echo 1. NVIDIA CUDA ToolKit 4.1 for Win 32 bit
echo 2. NVIDIA CUDA ToolKit 4.1 for Win 64 bit (also contains 32bit libs)
echo N. I have already installed an NVIDIA CUDA 4.1 Toolkit
echo.
set CUDA_CHOICE=0
set /p CUDA_CHOICE="Selection? "
IF %CUDA_CHOICE% EQU 1 goto CUDA_32
IF %CUDA_CHOICE% EQU 2 goto CUDA_64
IF /i "%CUDA_CHOICE%" == "N" goto Finished
echo Invalid choice
goto Choice

:CUDA_32
set CUDA_NAME=NVIDIA CUDA ToolKit 4.1 for Win 32 bit
set CUDA_URL=http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/
set CUDA_PKG=cudatoolkit_4.1.28_win_32.msi
goto Install

:CUDA_64
set CUDA_NAME=NVIDIA CUDA ToolKit 4.1 for Win 64 bit
set CUDA_URL=http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/
set CUDA_PKG=cudatoolkit_4.1.28_win_64.msi
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