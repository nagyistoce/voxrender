# VoxRender #
_(Click above for the wiki)_

This project is an open source implementation of an interactive, GPU based volume renderer. Although there are many existing, more well developed open source projects such as LuxRender and ExposureRender (and several components of this project derive from them) it may be of use to others for reference. The original motivation for the project was simply to implement an interactive volume rendering application as part of a directed study project at the University of Minnesota.

The application's core libraries, VoxLib and VoxVolt, are intended to serve as a standalone general purpose C++ APIs for rendering applications. It does not use LuxRender or QT code and can be utilized under a less restrictive license on request. This includes all of the plugins for resource IO and image import/export.

Models in a supported volume format (PVM) are available at http://www9.informatik.uni-erlangen.de/External/vollib/. The best formats to use are raw data files with standard compression types following the examples here: https://code.google.com/p/voxrender/source/browse/#svn%2Ftrunk%2FModels%2FExamples

# Web Application #

I recently started working on a web application version of VoxRender (Proof of Concept) which offloads the scene rendering to a server computer running a slightly different version of VoxRender without the UI. There is a test page below which works for 1 client at a time to provide rendering services. (Which may or may not be up depending on whether I have it running) The scene upload isn't working yet either but some example scenes are on the server.

[VoxRender Web App](http://lsherman.no-ip.org:61736)

![https://voxrender.googlecode.com/svn/wiki/images/webapp.jpg](https://voxrender.googlecode.com/svn/wiki/images/webapp.jpg)

# Renders #

![https://voxrender.googlecode.com/svn/wiki/images/renders.png](https://voxrender.googlecode.com/svn/wiki/images/renders.png)

# Videos #
| **Application Demo** | **Animation Widget** |
|:---------------------|:---------------------|
| <a href='http://www.youtube.com/watch?feature=player_embedded&v=xZysLCr3DyA' target='_blank'><img src='http://img.youtube.com/vi/xZysLCr3DyA/0.jpg' width='425' height=344 /></a> | <a href='http://www.youtube.com/watch?feature=player_embedded&v=v65bMKEpPho' target='_blank'><img src='http://img.youtube.com/vi/v65bMKEpPho/0.jpg' width='425' height=344 /></a> |

| **Rendered Animation** |
|:-----------------------|
| <a href='http://www.youtube.com/watch?feature=player_embedded&v=h3cyqfadYUA' target='_blank'><img src='http://img.youtube.com/vi/h3cyqfadYUA/0.jpg' width='425' height=344 /></a> |

# Download #

_Note: If the program terminates on startup, check the ./Logs folder. This is usually the result of an incompatible graphics card or CUDA installation_

**Latest Build**

[VoxRender Application for Windows x86 (5/15/2014)](https://drive.google.com/file/d/0ByknPmk3wl2nc3RZNU1uMnZiUWs/edit?usp=sharing)

[Include Dependencies for Windows x86 VC11](https://drive.google.com/file/d/0ByknPmk3wl2nUzNKSGhhX2xrdTA/edit?usp=sharing)

**Old Builds**

[Windows x86 (3/21/2014)](https://drive.google.com/file/d/0ByknPmk3wl2nU3JNRWRXMlVJYlk/edit?usp=sharing)

[Windows x86 (1/29/2014)](https://drive.google.com/file/d/0ByknPmk3wl2nU3JNRWRXMlVJYlk/edit?usp=sharing)

# Credits #

This project uses the following open-source software libraries: Boost C++ Libraries, libpng, QT5, libjpeg, libcurl, zlib, bzip, libav

And uses or references code from the following projects: LuxRender (UI form files), V^3 Volume Renderer (PVM format loader), ExposureRender (Rendering kernel code)

Data sets used for testing came from multiple sources:
  * http://www.osirix-viewer.com/datasets/
  * http://www9.informatik.uni-erlangen.de/External/vollib/
  * https://www.cg.tuwien.ac.at/research/vis/datasets/
  * http://www.sci.utah.edu/cgi-bin/NCRRdatasets.pl?id=volume-tooth

See the README files for more information.