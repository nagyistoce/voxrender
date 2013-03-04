@Echo off

:: This flag identifies the version of zlib to
:: be downloaded, you may change it but
:: it could cause issues with altered ABIs, etc 
set ZLIB_VER_P=1.2.3
set ZLIB_VER_N=123

:: Issue a message about the project and the use of 
:: wget for fetching the project source+dependencies
echo.
echo **************************************************************************
echo *                             Startup                                    *
echo **************************************************************************
echo.
echo NOTICE: You are about to download and zlib version %ZLIB_VER_P% from
echo         from the source link at http://sourceforge.net/. 
echo.
echo This script will use the following pre-built binaries to help build the project:
echo  1: GNU wget.exe    http://gnuwin32.sourceforge.net/packages/wget.htm
echo  2: 7za.exe (7-zip) http://7-zip.org/download.html
echo.

:: Set execution environment
call check-environment.bat

:: Download the source binary
IF NOT EXIST %DOWNLOADS%\zlib%ZLIB_VER_N%.zip (
	echo.
	echo **************************************************************************
	echo *                        Downloading zlib                                *
	echo **************************************************************************
	%WGET% http://sourceforge.net/projects/libpng/files/zlib/%ZLIB_VER_P%/zlib%ZLIB_VER_N%.zip/download -O %DOWNLOADS%\zlib%ZLIB_VER_N%.zip
	if ERRORLEVEL 1 (
		echo.
		echo Download failed. Are you connected to the internet?
		exit /b -1
	)
)

:: Extract the zlib source directory to the dependencies folder
set EXTRACT_ZLIB=1
IF EXIST %DEPENDS%\zlib-%ZLIB_VER_P% IF %FORCE_EXTRACT% NEQ 1 set EXTRACT_ZLIB=0
IF %EXTRACT_ZLIB% EQU 1 (
	echo.
	echo **************************************************************************
	echo *                         Extracting zlib                                *
	echo **************************************************************************
	%UNZIPBIN% x -y %DOWNLOADS%\zlib%ZLIB_VER_N%.zip -o%DEPENDS%\zlib-%ZLIB_VER_P% > nul
)

:: Set an environment variable for boost build script
set VOX_ZLIB_SRC_DIR=%DEPENDS%\zlib-%ZLIB_VER_P%

echo.
echo **************************************************************************
echo *                             Finished                                   *
echo **************************************************************************
echo.
echo  zlib is now configured for linking with the boost libraries.
echo.