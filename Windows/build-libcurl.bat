@Echo off

:: This flag identifies the version of libcurl to
:: be downloaded and built, you may change it but
:: it could cause issues with altered ABIs, etc 
set CURL_VER=7.29.0

:: Issue a message about the project and the use of 
:: wget for fetching the project source+dependencies
echo.
echo **************************************************************************
echo *                             Startup                                    *
echo **************************************************************************
echo.
echo NOTICE: You are about to download and build libcurl version %CURL_VER%
echo         from the curl website at http://curl.haxx.se/download.html
echo         libcurl is by default compiled for dynamic linking. See the
echo         readme file for more information.
echo.
echo This script will use the following pre-built binaries to help build the project:
echo  1: GNU wget.exe    http://gnuwin32.sourceforge.net/packages/wget.htm
echo  2: 7za.exe (7-zip) http://7-zip.org/download.html
echo.

:: Set execution environment
call check-environment.bat

:: Import the vox environment so we can determine whether 
:: additional dependencies need to be installed (zlib, openSSL, etc)
:: :TODO:

:: Download the source binary
IF NOT EXIST %DOWNLOADS%\curl-%CURL_VER%.zip (
	echo.
	echo **************************************************************************
	echo *                       Downloading libcurl                              *
	echo **************************************************************************
	echo.
	%WGET% http://curl.haxx.se/download/curl-%CURL_VER%.zip -O %DOWNLOADS%\curl-%CURL_VER%.zip
	if ERRORLEVEL 1 (
		echo.
		echo Download failed. You should check your internet connection and verify that the
		echo source link http://curl.haxx.se/download/curl-%CURL_VER%.zip is still valid.
		exit /b -1
	)
)

:: Extract the libcurl source directory to the temp folder
set EXTRACT_CURL=1
IF EXIST %DEPENDS%\curl-%CURL_VER% IF %FORCE_EXTRACT% NEQ 1 set EXTRACT_CURL=0
IF %EXTRACT_CURL% EQU 1 (
	echo.
	echo **************************************************************************
	echo *                       Extracting libcurl                               *
	echo **************************************************************************
	echo.
	%UNZIPBIN% x -y %DOWNLOADS%\curl-%CURL_VER%.zip -o%DEPENDS% > nul
)

:: Execute the VC makefile and build libcurl
echo.
echo **************************************************************************
echo *                          Building libcurl                              *
echo **************************************************************************
echo.

cd %DEPENDS%/curl-%CURL_VER%/winbuild
nmake /f Makefile.vc VC=11 GEN_PDB=yes mode=dll MACHINE=%BUILD_PLATFORM% /S > %CURRENT%\Reports\curl-%CURL_VER%_%BUILD_PLATFORM%_build.txt
cd %CURRENT%

rmdir %INCLUDES%\%BUILD_PLATFORM%\curl > %CURRENT%\Reports\tmp.txt
mklink /J %INCLUDES%\%BUILD_PLATFORM%\curl %DEPENDS%\curl-%CURL_VER%\builds\libcurl-vc10-%BUILD_PLATFORM%-release-dll-ipv6-sspi-spnego-winssl > %CURRENT%\Reports\tmp.txt

echo.
echo **************************************************************************
echo *                             Finished                                   *
echo **************************************************************************
echo.
echo  libcurl has now been built and configured for linking with the Vox libraries.
echo  If a previous version of libcurl was configured before, you may need to 
echo  remove it manually and reconfigure the CMake include directory as outlined 
echo  in the project readme files.
echo.