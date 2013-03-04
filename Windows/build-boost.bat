@Echo off

:: This flag identifies the version of boost to
:: be downloaded and built, you may change it but
:: it could cause issues with altered ABIs
set BOOST_VER=1.49.0
set BOOST_VER_U=1_49_0

:: Issue a message about the project and the use of 
:: wget for fetching the project source+dependencies
echo.
echo **************************************************************************
echo *                             Startup                                    *
echo **************************************************************************
echo.
echo NOTICE: You are about to download and build boost version %BOOST_VER%
echo         from the boost website at http://www.boost.org/. Only a portion
echo         of the boost libraries are actually compiled here. See the
echo         readme file for more information.
echo.
echo This script will use the following pre-built binaries to help build the project:
echo  1: GNU wget.exe    http://gnuwin32.sourceforge.net/packages/wget.htm
echo  2: 7za.exe (7-zip) http://7-zip.org/download.html
echo.

:: Set execution environment
call check-environment.bat

:: Get the bzip and zlib sources
call get-bzip2.bat
call get-zlib.bat

:: Download the source binary
IF NOT EXIST %DOWNLOADS%\boost_%BOOST_VER_U%.zip (
	echo.
	echo **************************************************************************
	echo *                         Downloading Boost                              *
	echo **************************************************************************
	%WGET% http://sourceforge.net/projects/boost/files/boost/%BOOST_VER%/boost_%BOOST_VER_U%.zip/download -O %DOWNLOADS%\boost_%BOOST_VER_U%.zip
	if ERRORLEVEL 1 (
		echo.
		echo Download failed. You should check your internet connection and verify that the
		echo source link http://sourceforge.net/projects/boost/files/boost/%BOOST_VER%/boost_%BOOST_VER_U%.zip/download 
		echo is still valid.
		exit /b -1
	)
)

set EXTRACT_BOOST=1
IF EXIST %DEPENDS%\boost_%BOOST_VER_U% IF %FORCE_EXTRACT% NEQ 1 set EXTRACT_BOOST=0
IF %EXTRACT_BOOST% EQU 1 (
	echo.
	echo **************************************************************************
	echo *                       Extracting Boost                                 *
	echo **************************************************************************
	echo.
	%UNZIPBIN% x -y %DOWNLOADS%\boost_%BOOST_VER_U%.zip -o%DEPENDS% > nul
)

echo.
echo **************************************************************************
echo *                           Building Boost                               *
echo **************************************************************************
echo.

echo.
echo **************************************************************************
echo *                           Building BJam                                *
echo **************************************************************************
cd /d %DEPENDS%\boost_%BOOST_VER_U%
call bootstrap.bat
SET BOOST_JOBS=8

echo.
echo **************************************************************************
echo *                     Building Boost::IOStreams                          *
echo **************************************************************************
set BUILD_DEBUG=0
IF %BUILD_DEBUG% EQU 1 ( bjam.exe -j%BOOST_JOBS% toolset=msvc-10.0 variant=debug link=static threading=multi runtime-link=shared -a -sZLIB_SOURCE=%VOX_ZLIB_SRC_DIR% -sBZIP2_SOURCE=%VOX_BZIP_SRC_DIR% --with-iostreams --stagedir=stage/boost --build-dir=bin/boost debug stage )
bjam.exe -j%BOOST_JOBS% toolset=msvc-10.0 variant=release link=static threading=multi runtime-link=shared -a -sZLIB_SOURCE=%VOX_ZLIB_SRC_DIR% -sBZIP2_SOURCE=%VOX_BZIP_SRC_DIR% --with-iostreams --stagedir=%INCLUDES%/x86/boost --build-dir=bin/boost stage > %CURRENT%\Reports\boost-%BOOST_VER%_io_x86_build.txt

echo.
echo **************************************************************************
echo * Building Boost::FileSystem                                             *
echo *          Boost::Program_Options                                        *
echo *          Boost::Regex                                                  *
echo *          Boost::Serialization                                          *
echo *          Boost::Thread                                                 *
echo *          Boost::Date_Time                                              *
echo *          Boost::Unit_Test_Framework                                    *
echo *          Boost::Chrono                                                 *
echo **************************************************************************
IF %BUILD_DEBUG% EQU 1 ( bjam.exe -j%BOOST_JOBS% toolset=msvc-10.0 variant=debug link=static threading=multi runtime-link=shared -a --with-test --with-date_time --with-filesystem --with-program_options --with-regex --with-serialization --with-thread --stagedir=stage/boost --build-dir=bin/boost debug stage ) 
bjam.exe -j%BOOST_JOBS% toolset=msvc-10.0 variant=release link=static threading=multi runtime-link=shared -a --with-test --with-date_time --with-filesystem --with-program_options --with-regex --with-serialization --with-thread --stagedir=%INCLUDES%/x86/boost --build-dir=bin/boost stage > %CURRENT%\Reports\boost-%BOOST_VER%_x86_build.txt

echo d | xcopy %DEPENDS%\boost_%BOOST_VER_U%\boost %INCLUDES%\x86\boost\include /E /D /Y > %CURRENT%\Reports\tmp.txt
cd %CURRENT%

echo.
echo **************************************************************************
echo *                             Finished                                   *
echo **************************************************************************
echo.
echo  Boost has now been built and configured for linking with the Vox libraries.
echo  See the project readme for more details.
echo.