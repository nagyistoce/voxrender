@Echo off

:: Issue a message about the project and the use of 
:: wget for fetching the project source+dependencies
echo.
echo **************************************************************************
echo *                             Startup                                    *
echo **************************************************************************
echo.
echo NOTICE: You are about to download and build the dependencies for the 
echo         Vox library. See the readme file for more information.
echo.
echo This script will use the following pre-built binary to help build the project:
echo  1: GNU wget.exe    http://gnuwin32.sourceforge.net/packages/wget.htm
echo  2: 7za.exe (7-zip) http://7-zip.org/download.html
echo.

:: Set execution environment
call check-environment.bat

:: Execute dependency build scripts
call build-cuda.bat
call build-qt.bat
call build-boost.bat
call build-libcurl.bat

echo.
echo **************************************************************************
echo *                             Finished                                   *
echo **************************************************************************
echo.
echo  Dependencies have now been built and configured for linking with the Vox libraries.
echo  For information on how to build VoxRender, see the project readme.
echo.