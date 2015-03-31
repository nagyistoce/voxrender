# Introduction #

VoxRender is a C++ application designed to facilitate the development and testing of volume rendering algorithms.

# Features #

  * Volume rendering for 3D/4D datasets of less than 1-2 GB per 3D slice
  * A monte-carlo based volume scattering renderer using CUDA
  * Volume import/export support for PVM and RAW/Binary formats
  * Stereo 3D rendering
  * 1D/2D Transfer function editing
  * Animation sequencing and rendering with keyframes to several video formats
  * Dynamic lighting, clip plane selection, and scene configuration
  * C++ Plugin support for Volume import/export, IO protocols, AV codecs, and volume filters
  * (In Progress) A web application for remote rendering

# Program Structure #

The VoxRender application itself is essentially a hefty graphical interface based on the [LuxRender](http://www.luxrender.net/) interface's GUI form files. The core functional components of the application are comprised of a handful of core libraries:

| **Library** | **Dependencies** | **Overview** |
|:------------|:-----------------|:-------------|
| VoxLib | _none_ | A core library providing basic IO, Image, Plugin,  and Logging utilities as well as basic classes for computing with CUDA |
| VoxScene | VoxLib | A library built on VoxLib which provides definitions for scene graph elements and renderers |
| VoxVolt | VoxLib VoxScene | A library built on VoxScene which provides based volume filtering functionality and a runtime registration system for filtering plugins |
| VoxServer | VoxLib VoxScene VoxVolt | A library providing render server hosting using WebSockets protocol (for VoxWebApp) |

While the remaining components are essentially plugins for the core libraries that perform filtering, rendering, image/video/scene encoding and resource IO tasks. The primary plugins include:

| **Plugin** | **Overview** |
|:-----------|:-------------|
| StandardIO | Handles IO for URI using schemes support by LibCurl |
| FileIO | Handles _file://_ IO using boost filesystem |
| VoxSceneExporter | A high level Scene import/export module for xml format scene specifications which supports inline references to other documents of any scene formats registered in the system at runtime and over any supported URI protocol |
| RawVolumeImporter | A Scene import/export module for binary volume data files using zlib or bzip encoding schemes |
| PVMVolumeImporter | A Scene import module for PVM format volume files |
| StdVolumeFilter | Volume filtering modules for various convolution and linear transform operations |
| StandardImg | Image import/export modules for bmp, png, and jpeg image formats |
| StandardVid | An AV file import/export module built on ffmpeg (for exporting videos) |

# Binaries #

Windows binaries are posted on Google Docs when available:

[VoxRender Application for Windows x86 (5/15/2014)](https://drive.google.com/file/d/0ByknPmk3wl2nc3RZNU1uMnZiUWs/edit?usp=sharing)

The program is currently relatively stable but still in development. It was written to be platform generic but has not actually been tested on Linux or Mac so it may be awhile before any binaries are available.

# Compiling from Source #

The program build files can be configured and generated with [CMake](http://www.cmake.org/) and compiled with any of the supported compilers. In depth information on compiling the source code is available on the ["Building VoxRender"](https://code.google.com/p/voxrender/wiki/BuildingVoxRender) page.

# Licensing #

This program is licensed under the GNU Public License Version 3.

Because the VoxRender application interface is derived from LuxRender, a GPL licensed software product, the licensing of the interface components of the program are non-negotiable.

The VoxLib library and alternative program components however can be made available through a more permissive license on request.