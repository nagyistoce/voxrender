@Echo off

:: This flag identifies the version of qt to
:: be downloaded and built, you may change it but
:: it could cause issues with altered ABIs, etc 
set QT_VER=4.8.0

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
	%WGET% ftp://ftp.qt-project.org/qt/source/qt-everywhere-opensource-src-%QT_VER%.zip -O %DOWNLOADS%\qt-everywhere-opensource-src-%QT_VER%.zip
	if ERRORLEVEL 1 (
		echo.
		echo Download failed. You should check your internet connection and verify that the
		echo source link http://get.qt.nokia.com/qt/source/qt-everywhere-opensource-src-%QT_VER%.zip 
		echo is still valid.
		exit /b -1
	)
)

:: Extract the qt source directory to the temp folder
set EXTRACT_QT=1
IF EXIST %DEPENDS%\qt-everywhere-opensource-src-%QT_VER% IF %FORCE_EXTRACT% NEQ 1 set EXTRACT_QT=0
IF %EXTRACT_QT% EQU 1 ( 
	echo.
	echo **************************************************************************
	echo *                           Extracting QT                                *
	echo **************************************************************************
	%UNZIPBIN% x -y %DOWNLOADS%\qt-everywhere-opensource-src-%QT_VER%.zip -o%DEPENDS% > nul
)	

echo.
echo **************************************************************************
echo *                           Building QT                                  *
echo **************************************************************************
echo.

cd /d %DEPENDS%/qt-everywhere-opensource-src-%QT_VER%
echo Cleaning Qt, this may take a few moments...
nmake confclean 1>nul 2>nul
echo.

del "bin\syncqt"
del "bin\syncqt.bat"
configure -opensource -release -fast -mp -plugin-manifests -nomake demos -nomake examples -no-multimedia -no-phonon -no-phonon-backend -no-audio-backend -no-webkit -no-script -no-scripttools -no-sse2
nmake > %CURRENT%\Reports\qt-everywhere-opensource-src-%QT_VER%_x86_build.txt
cd %CURRENT%
echo d | xcopy %DEPENDS%\qt-everywhere-opensource-src-%QT_VER%\lib %INCLUDES%\x86\qt-everywhere-opensource-src\lib /E /D /Y > %CURRENT%\Reports\tmp.txt
echo d | xcopy %DEPENDS%\qt-everywhere-opensource-src-%QT_VER%\bin %INCLUDES%\x86\qt-everywhere-opensource-src\bin /E /D /Y > %CURRENT%\Reports\tmp.txt
echo d | xcopy %DEPENDS%\qt-everywhere-opensource-src-%QT_VER%\include %INCLUDES%\x86\qt-everywhere-opensource-src\include /E /D /Y > %CURRENT%\Reports\tmp.txt

echo.
echo **************************************************************************
echo *                             Finished                                   *
echo **************************************************************************
echo.
echo  QT has now been built and configured for linking with the Vox libraries.
echo.