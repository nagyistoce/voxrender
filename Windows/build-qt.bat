@Echo off

:: This flag identifies the version of qt to
:: be downloaded and built, you may change it but
:: it could cause issues with altered ABIs, etc 
set QT_VER=5.2.0

:: Issue a message about the project and the use of 
:: wget for fetching the project source+dependencies
echo.
echo **************************************************************************
echo *                             Startup                                    *
echo **************************************************************************
echo.
echo NOTICE: You are about to download and build qt version %QT_VER%
echo         from the qt website at http://qt.nokia.com/
echo.
echo         WARNING: QT is a large library and will require some time to build.
echo         In addition, unique copies of the build tree are made for each 
echo         platform and the disk usage may exceed several gigabytes. QT is only
echo         needed for building the VoxRender application component and not the 
echo         VoxLib C++ Library.
echo.
echo This script will use the following pre-built binaries to help build the project:
echo  1: GNU wget.exe    http://gnuwin32.sourceforge.net/packages/wget.htm
echo  2: 7za.exe (7-zip) http://7-zip.org/download.html
echo.

:: Set execution environment
call check-environment.bat

:: Download the source binary
IF NOT EXIST %DOWNLOADS%\qt-everywhere-opensource-src-%QT_VER%.zip (
	echo.
	echo **************************************************************************
	echo *                          Downloading QT                                *
	echo **************************************************************************
	echo.
	%WGET% http://download.qt-project.org/official_releases/qt/5.2/%QT_VER%/single/qt-everywhere-opensource-src-%QT_VER%.zip -O %DOWNLOADS%\qt-everywhere-opensource-src-%QT_VER%.zip
	if ERRORLEVEL 1 (
		echo.
		echo Download failed. You should check your internet connection and verify that the
		echo source link http://get.qt.nokia.com/qt/source/qt-everywhere-opensource-src-%QT_VER%.zip 
		echo is still valid.
		exit /b -1
	)
)

:: Extract the qt source directory to the temp folder
echo.
echo **************************************************************************
echo *                           Extracting QT                                *
echo **************************************************************************
rmdir /s /q %INCLUDES%\%BUILD_PLATFORM%\qt-everywhere-opensource-src\ > %CURRENT%\Reports\tmp.txt
%UNZIPBIN% x -y %DOWNLOADS%\qt-everywhere-opensource-src-%QT_VER%.zip -o%INCLUDES%\%BUILD_PLATFORM% > nul
ren %INCLUDES%\%BUILD_PLATFORM%\qt-everywhere-opensource-src-%QT_VER% qt-everywhere-opensource-src

echo.
echo **************************************************************************
echo *                           Building QT                                  *
echo **************************************************************************
echo.

cd /d %INCLUDES%/%BUILD_PLATFORM%/qt-everywhere-opensource-src
echo Cleaning Qt, this may take a few moments...
nmake confclean 1>nul 2>nul
echo.

configure -opensource -release -mp -plugin-manifests -nomake -nomake examples -no-phonon -no-phonon-backend -no-audio-backend -no-webkit -no-script -no-scripttools -no-sse2
nmake
cd %CURRENT%

echo.
echo **************************************************************************
echo *                             Finished                                   *
echo **************************************************************************
echo.
echo  QT has now been built and configured for linking with the Vox libraries.
echo.