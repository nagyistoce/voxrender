@Echo off

set BOOST_VER_U=1_49_0
set BOOST_VER_P=1.49.0

set PYTHON2_VER=2.7.1
set PYTHON3_VER=3.2

set ZLIB_VER_P=1.2.3
set ZLIB_VER_N=123

set FREEIMAGE_VER_P=3.14.1
set FREEIMAGE_VER_N=3141

set QT_VER=4.8.0

set CURL_VER=7.29.0

set GLEW_VER=1.5.5


echo.
echo **************************************************************************
echo * Startup                                                                *
echo **************************************************************************
echo.
echo We are going to download and extract these libraries:
echo   Boost %BOOST_VER_P%                     http://www.boost.org/
echo   QT %QT_VER%                         http://qt.nokia.com/
echo   zlib %ZLIB_VER_P%                       http://www.zlib.net/
echo   bzip 1.0.6                       http://www.bzip.org/
echo   FreeImage %FREEIMAGE_VER_P%                 http://freeimage.sf.net/
echo   Python %PYTHON2_VER% ^& Python %PYTHON3_VER%        http://www.python.org/
echo   freeglut 2.6.0                   http://freeglut.sourceforge.net/
echo   NVIDIA CUDA ToolKit 4.1          http://developer.nvidia.com/cuda-toolkit-42
echo   NVIDIA OPTIX SDK 3.0 Beta        http://developer.nvidia.com/optix
echo   libcurl %CURL_VER%                 http://curl.haxx.se/libcurl/
echo.
echo Downloading and extracting all this source code will require several gigabytes,
echo and building it will require a lot more. Make sure you have plenty of space
echo available on this drive, at least 5GB.
echo.
echo This script will use 2 pre-built binaries to download and extract source
echo code from the internet:
echo  1: GNU wget.exe       from http://gnuwin32.sourceforge.net/packages/wget.htm
echo  2: 7za.exe (7-zip)    from http://7-zip.org/download.html
echo.
echo If you do not wish to execute these binaries for any reason, PRESS CTRL-C NOW
pause


echo.
echo **************************************************************************
echo * Checking environment                                                   *
echo **************************************************************************
set WGET="%CD%\Support\bin\wget.exe"
%WGET% --version 1> nul 2>&1
if ERRORLEVEL 9009 (
	echo.
	echo Cannot execute wget. Aborting.
	exit /b -1
)
set UNZIPBIN="%CD%\Support\bin\7za.exe"
%UNZIPBIN% > nul
if ERRORLEVEL 9009 (
	echo.
	echo Cannot execute unzip. Aborting.
	exit /b -1
)

set DOWNLOADS="%CD%\..\Downloads"
set DEPSROOT=%CD%\..\Includes

:ConfigDepsDir
:: resolve relative path
FOR %%G in (%DOWNLOADS%) do (
	set DOWNLOADS="%%~fG"
)

for %%G in (%DEPSROOT%) do (
	set DEPSROOT=%%~fG
)

set D32="%DEPSROOT%\x86"
::FOR %%G in (%D32%) do (
::	set D32="%%~fG"
::)
set D32R=%D32:"=%

set D64="%DEPSROOT%\x64"
::FOR %%G in (%D64%) do (
::	set D64="%%~fG"
::)
set D64R=%D64:"=%

echo.
echo Downloads will be stored in %DOWNLOADS%
echo Dependencies will be extracted to "%DEPSROOT%"
echo.
echo Change these locations?
echo.
echo 0. No (default)
echo 1. Yes
echo.
set /P CHANGE_DEPSROOT="Selection? "
IF %CHANGE_DEPSROOT% EQU 0 GOTO DepsRootAccepted
IF %CHANGE_DEPSROOT% EQU 1 GOTO ChangeDepsRoot
echo Invalid selection
GOTO ConfigDepsDir


:ChangeDepsRoot
set /P DOWNLOADS="Enter path for downloads: "
set /P DEPSROOT="Enter path for dependencies: "
GOTO ConfigDepsDir

:DepsRootAccepted
mkdir %DOWNLOADS% 2> nul
mkdir %D32% 2> nul
mkdir %D64% 2> nul

::echo %DOWNLOADS%
::echo %D32%
::echo %D64%
::echo OK


set FORCE_EXTRACT=0
:ForceExtractChoice
echo.
echo **************************************************************************
echo * Extract Option                                                         *
echo **************************************************************************
echo.
echo Should all sources be decompressed regardless of whether they have already
echo been extracted ?
echo.
echo 0. No (default)
echo 1. Yes
echo.
set /p FORCE_EXTRACT="Selection? "
IF %FORCE_EXTRACT% EQU 0 GOTO CreateBuildVars
IF %FORCE_EXTRACT% EQU 1 GOTO CreateBuildVars
echo Invalid choice
goto ForceExtractChoice


:CreateBuildVars
echo @Echo off > build-vars.bat
set VOX_WINDOWS_BUILD_ROOT="%CD%"
echo set VOX_WINDOWS_BUILD_ROOT="%CD%">> build-vars.bat

:Cuda
echo.
echo **************************************************************************
echo * NVidia's GPU Computing SDK                                             *
echo **************************************************************************
:OpenCLChoice
echo.
echo Please select which Cuda SDK you wish to use:
echo.
echo 1. NVIDIA CUDA ToolKit 4.1 for Win 32 bit
echo 2. NVIDIA CUDA ToolKit 4.1 for Win 64 bit (also contains 32bit libs)
echo N. I have already installed an NVIDIA CUDA 4.1 Toolkit
echo.
set CUDA_CHOICE=0
set /p CUDA_CHOICE="Selection? "
IF %CUDA_CHOICE% EQU 1 GOTO CUDA_32
IF %CUDA_CHOICE% EQU 2 GOTO CUDA_64
IF /i "%CUDA_CHOICE%" == "N" GOTO SetCUDAVars
echo Invalid choice
GOTO OpenCLChoice

:CUDA_32
set CUDA_VARS=SetCUDAVars
set CUDA_NAME=NVIDIA CUDA ToolKit 4.1 for Win 32 bit
set CUDA_URL=http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/
set CUDA_PKG=cudatoolkit_4.1.28_win_32.msi
GOTO CudaInstall

:CUDA_64
set CUDA_VARS=SetCUDAVars
set CUDA_NAME=NVIDIA CUDA ToolKit 3.1 for Win 64 bit
set CUDA_URL=http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/
set CUDA_PKG=cudatoolkit_4.1.28_win_64.msi
GOTO CudaInstall

:CudaInstall
IF NOT EXIST %DOWNLOADS%\%CUDA_PKG% (
	echo.
	echo **************************************************************************
	echo * Downloading %CUDA_NAME%
	echo **************************************************************************
	%WGET% %CUDA_URL%%CUDA_PKG% -O %DOWNLOADS%\%CUDA_PKG%
	if ERRORLEVEL 1 (
		echo.
		echo Download failed. Are you connected to the internet?
		exit /b -1
	)
)
echo.
echo The SDK installer will now be launched. You can install anywhere you like, but it
echo is required that  the install path not contain any spaces if you intend to use the
echo build automation batch files.

start /WAIT "" %DOWNLOADS%\%OPENCL_PKG%
echo Waiting for installer. When finished,
pause
goto %OPENCL_VARS%

:SetCUDAVars
:: Use another cmd instance to get new env vars expanded
cmd /C echo set VOX_X86_CUDA_LIBS="%CUDA_LIB_PATH%\..\lib\">> build-vars.bat
cmd /C echo set VOX_X86_CUDA_INCLUDE="%CUDA_INC_PATH%">> build-vars.bat
cmd /C echo set VOX_X64_CUDA_LIBS="%CUDA_LIB_PATH%">> build-vars.bat
cmd /C echo set VOX_X64_CUDA_INCLUDE="%CUDA_INC_PATH%">> build-vars.bat

goto CudaFinished

:CudaFinished

:boost
IF NOT EXIST %DOWNLOADS%\boost_%BOOST_VER_U%.zip (
	echo.
	echo **************************************************************************
	echo * Downloading Boost                                                      *
	echo **************************************************************************
	%WGET% http://sourceforge.net/projects/boost/files/boost/%BOOST_VER_P%/boost_%BOOST_VER_U%.zip/download -O %DOWNLOADS%\boost_%BOOST_VER_U%.zip
	if ERRORLEVEL 1 (
		echo.
		echo Download failed. Are you connected to the internet?
		exit /b -1
	)
)

set EXTRACT_BOOST=1
IF EXIST %D32%\boost_%BOOST_VER_U% IF %FORCE_EXTRACT% NEQ 1 set EXTRACT_BOOST=0
IF %EXTRACT_BOOST% EQU 1 (
	echo.
	echo **************************************************************************
	echo * Extracting Boost                                                       *
	echo **************************************************************************
	%UNZIPBIN% x -y %DOWNLOADS%\boost_%BOOST_VER_U%.zip -o%D32% > nul
	%UNZIPBIN% x -y %DOWNLOADS%\boost_%BOOST_VER_U%.zip -o%D64% > nul
)
echo set VOX_X86_BOOST_ROOT=%D32%\boost_%BOOST_VER_U%>> build-vars.bat
echo set VOX_X64_BOOST_ROOT=%D64%\boost_%BOOST_VER_U%>> build-vars.bat


:qt
IF NOT EXIST %DOWNLOADS%\qt-everywhere-opensource-src-%QT_VER%.zip (
	echo.
	echo **************************************************************************
	echo * Downloading QT                                                         *
	echo **************************************************************************
	%WGET% http://get.qt.nokia.com/qt/source/qt-everywhere-opensource-src-%QT_VER%.zip -O %DOWNLOADS%\qt-everywhere-opensource-src-%QT_VER%.zip
	if ERRORLEVEL 1 (
		echo.
		echo Download failed. Are you connected to the internet?
		exit /b -1
	)
)


:curl
IF NOT EXIST %DOWNLOADS%\curl-%CURL_VER%.zip (
	echo.
	echo **************************************************************************
	echo * Downloading libcurl                                                    *
	echo **************************************************************************
	%WGET% http://curl.haxx.se/download/curl-%CURL_VER%.zip -O %DOWNLOADS%\curl-%CURL_VER%.zip
	if ERRORLEVEL 1 (
		echo.
		echo Download failed. Are you connected to the internet?
		exit /b -1
	)
)

set EXTRACT_CURL=1
IF EXIST %D32%\curl-%CURL_VER% IF %FORCE_EXTRACT% NEQ 1 set EXTRACT_CURL=0
IF %EXTRACT_CURL% EQU 1 (
	echo.
	echo **************************************************************************
	echo * Extracting libcurl                                                     *
	echo **************************************************************************
	%UNZIPBIN% x -y %DOWNLOADS%\curl-%CURL_VER%.zip -o%D32% > nul
	%UNZIPBIN% x -y %DOWNLOADS%\curl-%CURL_VER%.zip -o%D64% > nul
)
echo set VOX_X86_CURL_ROOT=%D32%\curl-%CURL_VER%>> build-vars.bat
echo set VOX_X64_CURL_ROOT=%D64%\curl-%CURL_VER%>> build-vars.bat


set EXTRACT_QT=1
IF EXIST %D32%\qt-everywhere-opensource-src-%QT_VER% IF %FORCE_EXTRACT% NEQ 1 set EXTRACT_QT=0
IF %EXTRACT_QT% EQU 1 ( 
	echo.
	echo **************************************************************************
	echo * Extracting QT                                                          *
	echo **************************************************************************
	%UNZIPBIN% x -y %DOWNLOADS%\qt-everywhere-opensource-src-%QT_VER%.zip -o%D32% > nul
	%UNZIPBIN% x -y %DOWNLOADS%\qt-everywhere-opensource-src-%QT_VER%.zip -o%D64% > nul
)	
echo set VOX_X86_QT_ROOT=%D32%\qt-everywhere-opensource-src-%QT_VER%>> build-vars.bat
echo set VOX_X64_QT_ROOT=%D64%\qt-everywhere-opensource-src-%QT_VER%>> build-vars.bat


:zlib
IF NOT EXIST %DOWNLOADS%\zlib%ZLIB_VER_N%.zip (
	echo.
	echo **************************************************************************
	echo * Downloading zlib                                                       *
	echo **************************************************************************
	%WGET% http://sourceforge.net/projects/libpng/files/zlib/%ZLIB_VER_P%/zlib%ZLIB_VER_N%.zip/download -O %DOWNLOADS%\zlib%ZLIB_VER_N%.zip
	if ERRORLEVEL 1 (
		echo.
		echo Download failed. Are you connected to the internet?
		exit /b -1
	)
)

set EXTRACT_ZLIB=1
IF EXIST %D32%\zlib-%ZLIB_VER_P% IF %FORCE_EXTRACT% NEQ 1 set EXTRACT_ZLIB=0
IF %EXTRACT_ZLIB% EQU 1 (
	echo.
	echo **************************************************************************
	echo * Extracting zlib                                                        *
	echo **************************************************************************
	%UNZIPBIN% x -y %DOWNLOADS%\zlib%ZLIB_VER_N%.zip -o%D32%\zlib-%ZLIB_VER_P% > nul
	%UNZIPBIN% x -y %DOWNLOADS%\zlib%ZLIB_VER_N%.zip -o%D64%\zlib-%ZLIB_VER_P% > nul
)
echo set VOX_X86_ZLIB_ROOT=%D32%\zlib-%ZLIB_VER_P%>> build-vars.bat
echo set VOX_X64_ZLIB_ROOT=%D64%\zlib-%ZLIB_VER_P%>> build-vars.bat


:bzip
IF NOT EXIST %DOWNLOADS%\bzip2-1.0.6.tar.gz (
	echo.
	echo **************************************************************************
	echo * Downloading bzip                                                       *
	echo **************************************************************************
	%WGET% http://www.bzip.org/1.0.6/bzip2-1.0.6.tar.gz -O %DOWNLOADS%\bzip2-1.0.6.tar.gz
	if ERRORLEVEL 1 (
		echo.
		echo Download failed. Are you connected to the internet?
		exit /b -1
	)
)

set EXTRACT_BZIP=1
IF EXIST %D32%\bzip2-1.0.6 IF %FORCE_EXTRACT% NEQ 1 set EXTRACT_BZIP=0
IF %EXTRACT_BZIP% EQU 1 (
	echo.
	echo **************************************************************************
	echo * Extracting bzip                                                        *
	echo **************************************************************************
	%UNZIPBIN% x -y %DOWNLOADS%\bzip2-1.0.6.tar.gz > nul
	%UNZIPBIN% x -y bzip2-1.0.6.tar -o%D32% > nul
	%UNZIPBIN% x -y bzip2-1.0.6.tar -o%D64% > nul
	del bzip2-1.0.5.tar
)
echo set VOX_X86_BZIP_ROOT=%D32%\bzip2-1.0.6>> build-vars.bat
echo set VOX_X64_BZIP_ROOT=%D64%\bzip2-1.0.6>> build-vars.bat


:freeimage
IF NOT EXIST %DOWNLOADS%\FreeImage%FREEIMAGE_VER_N%.zip (
	echo.
	echo **************************************************************************
	echo * Downloading FreeImage                                                  *
	echo **************************************************************************
	%WGET% http://downloads.sourceforge.net/freeimage/FreeImage%FREEIMAGE_VER_N%.zip -O %DOWNLOADS%\FreeImage%FREEIMAGE_VER_N%.zip
	if ERRORLEVEL 1 (
		echo.
		echo Download failed. Are you connected to the internet?
		exit /b -1
	)
)

set EXTRACT_FREEIMAGE=1
IF EXIST %D32%\FreeImage%FREEIMAGE_VER_N% IF %FORCE_EXTRACT% NEQ 1 set EXTRACT_FREEIMAGE=0
IF %EXTRACT_FREEIMAGE% EQU 1 (
	echo.
	echo **************************************************************************
	echo * Extracting FreeImage                                                   *
	echo **************************************************************************
	%UNZIPBIN% x -y %DOWNLOADS%\FreeImage%FREEIMAGE_VER_N%.zip -o%D32%\FreeImage%FREEIMAGE_VER_N% > nul
	%UNZIPBIN% x -y %DOWNLOADS%\FreeImage%FREEIMAGE_VER_N%.zip -o%D64%\FreeImage%FREEIMAGE_VER_N% > nul
)
echo set VOX_X86_FREEIMAGE_ROOT=%D32%\FreeImage%FREEIMAGE_VER_N%>> build-vars.bat
echo set VOX_X64_FREEIMAGE_ROOT=%D64%\FreeImage%FREEIMAGE_VER_N%>> build-vars.bat


:python2
IF NOT EXIST %DOWNLOADS%\Python-%PYTHON2_VER%.tgz (
	echo.
	echo **************************************************************************
	echo * Downloading Python 2                                                   *
	echo **************************************************************************
	%WGET% http://python.org/ftp/python/%PYTHON2_VER%/Python-%PYTHON2_VER%.tgz -O %DOWNLOADS%\Python-%PYTHON2_VER%.tgz
	if ERRORLEVEL 1 (
		echo.
		echo Download failed. Are you connected to the internet?
		exit /b -1
	)
)

set EXTRACT_PYTHON2=1
IF EXIST %D32%\Python-%PYTHON2_VER% IF %FORCE_EXTRACT% NEQ 1 set EXTRACT_PYTHON2=0
IF %EXTRACT_PYTHON2% EQU 1 (
	echo.
	echo **************************************************************************
	echo * Extracting Python 2                                                    *
	echo **************************************************************************
	%UNZIPBIN% x -y %DOWNLOADS%\Python-%PYTHON2_VER%.tgz > nul
	%UNZIPBIN% x -y Python-%PYTHON2_VER%.tar -o%D32% > nul
	%UNZIPBIN% x -y Python-%PYTHON2_VER%.tar -o%D64% > nul
	del Python-%PYTHON2_VER%.tar
)
echo set VOX_X86_PYTHON2_ROOT=%D32%\Python-%PYTHON2_VER%>> build-vars.bat
echo set VOX_X64_PYTHON2_ROOT=%D64%\Python-%PYTHON2_VER%>> build-vars.bat


:python3
IF NOT EXIST %DOWNLOADS%\Python-%PYTHON3_VER%.tgz (
	echo.
	echo **************************************************************************
	echo * Downloading Python 3                                                   *
	echo **************************************************************************
	%WGET% http://python.org/ftp/python/%PYTHON3_VER%/Python-%PYTHON3_VER%.tgz -O %DOWNLOADS%\Python-%PYTHON3_VER%.tgz
	if ERRORLEVEL 1 (
		echo.
		echo Download failed. Are you connected to the internet?
		exit /b -1
	)
)

set EXTRACT_PYTHON3=1
IF EXIST %D32%\Python-%PYTHON3_VER% IF %FORCE_EXTRACT% NEQ 1 set EXTRACT_PYTHON3=0
IF %EXTRACT_PYTHON3% EQU 1 (
	echo.
	echo **************************************************************************
	echo * Extracting Python 3                                                    *
	echo **************************************************************************
	%UNZIPBIN% x -y %DOWNLOADS%\Python-%PYTHON3_VER%.tgz > nul
	%UNZIPBIN% x -y Python-%PYTHON3_VER%.tar -o%D32% > nul
	%UNZIPBIN% x -y Python-%PYTHON3_VER%.tar -o%D64% > nul
	del Python-%PYTHON3_VER%.tar
)
echo set VOX_X86_PYTHON3_ROOT=%D32%\Python-%PYTHON3_VER%>> build-vars.bat
echo set VOX_X64_PYTHON3_ROOT=%D64%\Python-%PYTHON3_VER%>> build-vars.bat


set EXTRACT_FREEGLUT=1
IF EXIST %D32%\freeglut-2.6.0 IF %FORCE_EXTRACT% NEQ 1 set EXTRACT_FREEGLUT=0
IF %EXTRACT_FREEGLUT% EQU 1 (
	echo.
	echo **************************************************************************
	echo * Extracting freeglut                                                    *
	echo **************************************************************************
	%UNZIPBIN% x -y "%VOX_WINDOWS_BUILD_ROOT%"\support\freeglut-2.6.0.7z -o%D32%\ > nul
	%UNZIPBIN% x -y "%VOX_WINDOWS_BUILD_ROOT%"\support\freeglut-2.6.0.7z -o%D64%\ > nul
)
echo set VOX_X86_GLUT_INCLUDE=%D32%\freeglut-2.6.0\include>> build-vars.bat
echo set VOX_X64_GLUT_INCLUDE=%D64%\freeglut-2.6.0\include>> build-vars.bat
echo set VOX_X86_GLUT_LIBS=%D32%\freeglut-2.6.0\VisualStudio2008Static\Win32\Release>> build-vars.bat
echo set VOX_X64_GLUT_LIBS=%D64%\freeglut-2.6.0\VisualStudio2008Static\x64\Release>> build-vars.bat


IF %SKIP_GLEW% EQU 0 (
	IF NOT EXIST %DOWNLOADS%\glew-%GLEW_VER%_x86.zip (
		echo.
		echo **************************************************************************
		echo * Downloading GLEW 32 bit                                                *
		echo **************************************************************************
		rem %WGET% http://sourceforge.net/projects/glew/files/glew/%GLEW_VER%/glew-%GLEW_VER%-win32.zip/download -O %DOWNLOADS%\glew-%GLEW_VER%-win32.zip
		%WGET% http://www.luxrender.net/release/luxrender/dev/win/libs/glew-%GLEW_VER%_x86.zip -O %DOWNLOADS%\glew-%GLEW_VER%_x86.zip
		if ERRORLEVEL 1 (
			echo.
			echo Download failed. Are you connected to the internet?
			exit /b -1
		)
	)
	IF NOT EXIST %DOWNLOADS%\glew-%GLEW_VER%_x64.zip (
		echo.
		echo **************************************************************************
		echo * Downloading GLEW 64 bit                                                *
		echo **************************************************************************
		rem %WGET% http://sourceforge.net/projects/glew/files/glew/%GLEW_VER%/glew-%GLEW_VER%-win64.zip/download -O %DOWNLOADS%\glew-%GLEW_VER%-win64.zip
		%WGET% http://www.luxrender.net/release/luxrender/dev/win/libs/glew-%GLEW_VER%_x64.zip -O %DOWNLOADS%\glew-%GLEW_VER%_x64.zip
		if ERRORLEVEL 1 (
			echo.
			echo Download failed. Are you connected to the internet?
			exit /b -1
		)
	)
	echo.
	echo **************************************************************************
	echo * Extracting GLEW                                                        *
	echo **************************************************************************
	%UNZIPBIN% x -y %DOWNLOADS%\glew-%GLEW_VER%_x86.zip -o%D32%\ > nul
	%UNZIPBIN% x -y %DOWNLOADS%\glew-%GLEW_VER%_x64.zip -o%D64%\ > nul
	
	echo set VOX_X86_GLEW_INCLUDE=%D32%\glew-%GLEW_VER%\include>> build-vars.bat
	echo set VOX_X64_GLEW_INCLUDE=%D64%\glew-%GLEW_VER%\include>> build-vars.bat
	echo set VOX_X86_GLEW_LIBS=%D32%\glew-%GLEW_VER%\lib>> build-vars.bat
	echo set VOX_X64_GLEW_LIBS=%D64%\glew-%GLEW_VER%\lib>> build-vars.bat
	echo set VOX_X86_GLEW_BIN=%D32%\glew-%GLEW_VER%\bin>> build-vars.bat
	echo set VOX_X64_GLEW_BIN=%D64%\glew-%GLEW_VER%\bin>> build-vars.bat
	
	echo set VOX_X86_GLEW_LIBNAME=glew32s>> build-vars.bat
	echo set VOX_X64_GLEW_LIBNAME=glew64s>> build-vars.bat

) ELSE (
	
        IF "%AMDAPPSDKSAMPLESROOT%"=="" (
               SET SAMPLESROOT=%ATISTREAMSDKSAMPLESROOT%
        ) ELSE (
               SET SAMPLESROOT=%AMDAPPSDKSAMPLESROOT%
        )

	cmd /C echo set VOX_X86_GLEW_INCLUDE="%SAMPLESROOT%\include">> build-vars.bat
	cmd /C echo set VOX_X64_GLEW_INCLUDE="%SAMPLESROOT%\include">> build-vars.bat
	cmd /C echo set VOX_X86_GLEW_LIBS="%SAMPLESROOT%\lib\x86">> build-vars.bat
	cmd /C echo set VOX_X64_GLEW_LIBS="%SAMPLESROOT%\lib\x86_64">> build-vars.bat
	cmd /C echo set VOX_X86_GLEW_BIN="%SAMPLESROOT%\bin\x86">> build-vars.bat
	cmd /C echo set VOX_X64_GLEW_BIN="%SAMPLESROOT%\bin\x86_64">> build-vars.bat
	
	echo set VOX_X86_GLEW_LIBNAME=glew32>> build-vars.bat
	echo set VOX_X64_GLEW_LIBNAME=glew64>> build-vars.bat
	
)



echo.
echo **************************************************************************
echo * DONE                                                                   *
echo **************************************************************************
echo.

echo To build dependencies for x86 you can now run build-deps-x86.bat from a
echo Visual Studio Command Prompt for x86 window.
echo.

echo To build dependencies for x64 you can now run build-deps-x64.bat from a
echo Visual Studio Command Prompt for x64 window.
echo.

echo For further instructions, see the Compile.rtf file.
echo.
pause
