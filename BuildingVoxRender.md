# Contents #



# Overview #

VoxRender uses the CMake build file generator available at the location listed in the Tools section below. The windows specific compilation tools require that the project directory (eg voxrender) structure is retained as is. If the directory structure is changed, the batch files may no longer work correctly.

Initial project directory structure:
```
        \voxrender
            \Binaries       _CMake output directory_
            \CMake          _CMake config files_
            \Documentation  _Doxygen build directory_
            \Dependencies   _Dependency source directory_
            \Downloads      _Temp directory for file download_
            \Includes       _Directory for dependency libs, etc_
            \Source         _VoxRender source code_
            \Windows        _Windows platform build tools_
            CMakeLists.txt  _CMake build file_
```
# Supported Compilers #
Although platform and compiler support may vary, the library has been built and tested for the following compiler and platform combinations.
```
    Visual Studio 2010       - Windows 7, 8
    Visual Studio 2012       - Windows 7, 8
```

# Required Libraries #

List of required libraries (latest version used):
```
	Boost C++:    http://www.boost.org/users/download/ (1.53.0)
	Digia's QT5:  http://qt.nokia.com/downloads (5.5)
	Cuda SDK:     http://developer.nvidia.com/cuda-downloads (5.2)
	Cuda Toolkit: http://developer.nvidia.com/cuda-downloads (5.2)
	BZip:         http://www.bzip.org/ (1.0.6)
	ZLib:         http://www.zlib.net/ (1.2.3)
```

Additional information on building and linking these libraries with VoxLib is provided in the Configuring Dependencies section below.

# Additional Libraries #

Many of the VoxRender application features, including the built in features, are implemented as binary plugins. To enable these features, the corresponding plugins must be built using additional libraries.

List of additional libraries (latest version used):
```
	libpng:  http://www.libpng.org/ (1.6.8)
	libjpeg: http://www.infai.org/jpeg/ (9a)
	libCurl: http://curl.haxx.se/libcurl/ (7.34.0)
```

# Required Tools #

List of required tools:
```
	Doxygen:    http://www.doxygen.org (optional - used for generating documentation)
	CMake:      http://www.cmake.org/cmake/resources/software.html
	QtDesigner: http://qt.nokia.com/downloads (optional - used to edit GUI form files)
```
# Building VoxRender #

For anyone familiar with the CMake build system, compiling the source code should be a simple matter of configuring the dependency include paths in the CMake/Dependencies.cmake source file and running cmake on the root directory. A more detailed process is described below, including some common issues that may be encountered while trying to build the project.


## (1) Configuring Dependencies ##

The first step in building any of the projects is to ensure that all of the 3rd party dependencies required are going to be found by the CMake scripts. For additional information, see http://www.cmake.org/Wiki/CMake:How_To_Find_Libraries. CMake is not a compiler or build tool and you will be required to install a compatible C++ compiler. You should ensure that the dependencies you use for building are compatible with your chosen compiler. I recommend GCC or the Visual Studio compiler (2010 or higher required).

### _Windows_ ###

If you are running Windows, then the batch files provided in the Windows subdirectory can be used to automatically download and build all of the required dependencies for Visual Studio 2010 (VC10). Simply run the build-all.bat file from a Visual Studio Command Prompt (2010) and follow the provided instructions. If you only need a subset of the dependencies, then you can launch any of the build .bat files to download and build that specific dependency.

  * If you encounter any issues using the windows bat files, the /Windows/Reports subdirectory keeps a collection of the build outputs for most dependencies.

  * The Visual Studio Command Prompt is a cmd launcher provided with Visual Studio which automatically configures an environment for building from the commandline. If you have difficulty locating it, try using the start menu search box.

After the batch files have completed, you should check the /Includes/ subdirectory and verify that there are directories present for each required dependency.

### _Other_ ###

By default, the CMake Find scripts are used to locate the project dependencies (see above.)  In addition to searching a variety of common locations for dependencies, the default includes directory "voxrender/Includes/%PLATFORM%" will be searched for most dependencies.

Each library's Find script will expect a directory in the !Includes folder containing a 'bin', 'lib', and 'include' folder which contains the respective dependencies build files. The names of each of the build dependencies default folders are listed below:
```
    Boost C++ Libraries: 'boost'
    QT Open Source SDK:  'qt-everywhere-opensource-src'
    LibCurl:             'curl'
```


#### Boost C++ Libraries ####

The boost libraries are a collection of multiple open source libraries updated on a regular basis. In addition, the IOStreams library must be built with the 'zlib' and 'bzip2' libraries. Fortunately the CMake scripts will automatically issue errors on the following conditions:

  * The boost version is not compatible with VoxRender

  * If the boost libraries required by VoxRender are not found

Detailed information on building boost libraries can be found on the official website at http://www.boost.org/.

#### QT SDK ####

The QT library is quite large and requires a significant amount of time to build. If you wish to build only the VoxLib library without the graphical application component, there is no need to download or build QT. The build time can also be reduced by not building examples and the webkit, audio, multimedia, and phonon libraries.

  * If you are symlinking to the QT include folder, you should take note that it contains relative references to the src directory. This means you must symlink to the src directory as well in the include folder.

  * Easy option is to download the prebuilt binaries for your system

## (2) Generating Build Files ##

After downloading CMake, launch the included GUI. You should be presented with a window similar to the one below. Don't worry that the text field is empty.

![https://voxrender.googlecode.com/svn/wiki/images/cmake-gui.png](https://voxrender.googlecode.com/svn/wiki/images/cmake-gui.png)

  * The top level build directory is specified within the text box labeled "Where is the source code:"  and should point to the root directory containing this readme and a CMakeLists.txt file.

  * The binary subdirectory /Binaries/ is where users new to CMake should build the binares. This is where cmake will generate all of the build files required for a given compiler.

Press configure and select your compiler from the popup menu.

  * You now have the opportunity to specify some options for building VoxRender. The ungrouped entities folder contains several key-value pairs which control whether libraries are shared or static as well as some additional options described in **(Optional)** sections. Unless you have reason to do otherwise, select shared for all of these. (Particularly VoxLib as it is required for centralized logging behavior)

Press generate and wait for the output window to display "Generation Complete"

!CMake should now have successfully generated build files for your compiler. These can be found in the directory you specified above for "Where to build the binaries". For Visual Studio, this will be a .sln file named VoxRender.

## (3) Compiling Targets ##

### Visual Studio ###

The CMake Visual Studio Solution file should build successfully without any need for changes. If you do not have Visual Studio, the 'express' edition can be downloaded from http://www.microsoft.com/visualstudio/eng/downloads.

It is necessary that the version of Visual Studio used to build VoxRender is the same version used to build to Boost C++ libraries.

If you are unfamiliar with Visual Studio:

  1. Open the VoxRender.sln file produced by CMake
  1. Press _F7_ or select _Build -> Build Solution_ from the the menu bar
  1. Wait for the build to complete (~ 10 minutes max on decent systems)
  1. Right click on the VoxRender project in the Solution Explorer on the left hand pane. If you cannot find it, try selecting _View -> Solution Explorer_ from the menu bar. From the drop-down menu, select _Set as Startup Project_
  1. Press F5 to launch the program, or CTRL-F5 to launch without the attached debugger

## (Optional) Generating Documentation Files ##

The GENERATE\_DOCUMENTATION flag in the !CMake options allows you to specify whether you want the capability to compile documentation for VoxRender. If checked, the build files generated will contain a Documentation target which will use !Doxygen and graphviz to generate documentation files in the source directory _/Documentation/html/_.

  * Doxygen can be downloaded at: http://www.doxygen.org
  * graphviz can be downloaded at: http://www.graphviz.org/

# Tips and Suggestions #

  * Build all targets and dependencies to link with the same CRT libraries. VoxLib utilizes standard library types extensively in interface classes.
  * There are a series of unit test targets generated by the CMake build scripts. If you are running on an untested system, you can use them to verify functionality of library components.

# Contact Information #

If you have any issues feel free to email me at LucasASherman@gmail.com.