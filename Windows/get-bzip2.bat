@Echo off

:: This flag identifies the version of bzip to
:: be downloaded, you may change it but
:: it could cause issues with altered ABIs, etc 
set BZIP_VER=1.0.6

:: Issue a message about the project and the use of 
:: wget for fetching the project source+dependencies
echo.
echo **************************************************************************
echo *                             Startup                                    *
echo **************************************************************************
echo.
echo NOTICE: You are about to download and bzip2 version %BZIP_VER% from
echo         from the source link at http://sourceforge.net/. 
echo.
echo This script will use the following pre-built binaries to help build the project:
echo  1: GNU wget.exe    http://gnuwin32.sourceforge.net/packages/wget.htm
echo  2: 7za.exe (7-zip) http://7-zip.org/download.html
echo.

:: Set execution environment
call check-environment.bat

:: Download the source binary
IF NOT EXIST %DOWNLOADS%\bzip2-%BZIP_VER%.tar.gz (
	echo.
	echo **************************************************************************
	echo *                        Downloading bzip                                *
	echo **************************************************************************
	%WGET% http://www.bzip.org/%BZIP_VER%/bzip2-%BZIP_VER%.tar.gz -O %DOWNLOADS%\bzip2-%BZIP_VER%.tar.gz
	if ERRORLEVEL 1 (
		echo.
		echo Download failed. Are you connected to the internet?
		exit /b -1
	)
)

:: Extract the bzip source directory to the dependencies folder
set EXTRACT_BZIP=1
IF EXIST %DEPENDS%\bzip2-%BZIP_VER% IF %FORCE_EXTRACT% NEQ 1 set EXTRACT_BZIP=0
IF %EXTRACT_BZIP% EQU 1 (
	echo.
	echo **************************************************************************
	echo *                       Extracting bzip                                  *
	echo **************************************************************************
	%UNZIPBIN% x -y %DOWNLOADS%\bzip2-%BZIP_VER%.tar.gz > nul
	%UNZIPBIN% x -y bzip2-%BZIP_VER%.tar -o%DEPENDS% > nul
	:: Delete intermediary tar, let user decide on full .tar.gz as specified in docs
	del bzip2-%BZIP_VER%.tar
)

:: Set an environment variable for boost build script
set VOX_BZIP_SRC_DIR=%DEPENDS%\bzip2-%BZIP_VER%

echo.
echo **************************************************************************
echo *                             Finished                                   *
echo **************************************************************************
echo.
echo  bzip is now configured for linking with the boost libraries.
echo.