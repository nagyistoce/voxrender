#===========================================================================#
#                                                                           #
#   Project: VoxRender - GPU Based Real-Time Volume Rendering               #
#                                                                           #
#   Description: Performs real-time rendering of volume data using volume   #
#       ray casting techniques.                                             #
#                                                                           #
#       Copyright (C) 2012 Lucas Sherman                                    #
#                                                                           #
#       Lucas Sherman, email: LucasASherman@gmail.com                       #
#                                                                           #
#   This program is free software: you can redistribute it and/or modify    #
#   it under the terms of the GNU General Public License as published by    #
#   the Free Software Foundation, either version 3 of the License, or       #
#   (at your option) any later version.                                     #
#                                                                           #
#   This program is distributed in the hope that it will be useful,         #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of          #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           #
#   GNU General Public License for more details.                            #
#                                                                           #
#   You should have received a copy of the GNU General Public License       #
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.   #
#                                                                           #
#===========================================================================#

MESSAGE(STATUS "Configuring VoxLib")

INCLUDE_DIRECTORIES(.)

#===============================================#
#                  HEADER FILES                 #
#===============================================#

Set(VoxLib_HEADERS_ACTION
    Action/Action.h
    Action/ActionManager.h
)

Set(VoxLib_HEADERS_BITMAP
    Bitmap/Bitmap.h
)

Set(VoxLib_HEADERS_CORE
    Core/Common.h
    Core/CudaCommon.h
    Core/Debug.h
    Core/Devices.h
    Core/Format.h
    Core/Functors.h
    Core/Logging.h
    Core/Preprocessor.h
    Core/System.h
    Core/Types.h
    Core/Version.h
    Core/Geometry.h
)

Set(VoxLib_HEADERS_CORE_GEOMETRY
    Core/Geometry/AABBox.h
    Core/Geometry/Color.h
    Core/Geometry/Image.h
    Core/Geometry/Image3D.h
    Core/Geometry/Matrix.h
    Core/Geometry/Ray.h
    Core/Geometry/Vector.h
)

Set(VoxLib_HEADERS_ERROR
    Error/CudaError.h
    Error/Error.h
    Error/ErrorCodes.h
    Error/FileError.h
    Error/PluginError.h
    Error/SystemError.h
)

Set(VoxLib_HEADERS_IO
    IO/MimeTypes.h
    IO/OptionSet.h
    IO/Resource.h
    IO/ResourceHelper.h
    IO/ResourceId.h
    IO/ResourceModule.h
)

Set(VoxLib_HEADERS_PLUGIN
    Plugin/Plugin.h
    Plugin/PluginContainer.h
    Plugin/PluginManager.h
)

Set(VoxLib_HEADERS_VIDEO
    Video/VidCodec.h
    Video/VidStream.h
)

Set(VoxLib_HEADERS_RENDERING
    Rendering/FrameBuffer.h
    Rendering/Renderer.h
    Rendering/RenderController.h
    Rendering/RenderThread.h
)

set(VoxLib_HEADERS
    ${VoxLib_HEADERS_ACTION}
    ${VoxLib_HEADERS_BITMAP}
    ${VoxLib_HEADERS_CORE}
    ${VoxLib_HEADERS_CORE_GEOMETRY}
    ${VoxLib_HEADERS_ERROR}
    ${VoxLib_HEADERS_IO}
    ${VoxLib_HEADERS_IMAGE}
    ${VoxLib_HEADERS_PLUGIN}
    ${VoxLib_HEADERS_VIDEO}
)

SOURCE_GROUP("Header Files\\Action"                         FILES ${VoxLib_HEADERS_ACTION})
SOURCE_GROUP("Header Files\\Bitmap"                         FILES ${VoxLib_HEADERS_BITMAP})
SOURCE_GROUP("Header Files\\Core"                           FILES ${VoxLib_HEADERS_CORE})
SOURCE_GROUP("Header Files\\Core\\Geometry"                 FILES ${VoxLib_HEADERS_CORE_GEOMETRY})
SOURCE_GROUP("Header Files\\Error"                          FILES ${VoxLib_HEADERS_ERROR})
SOURCE_GROUP("Header Files\\IO"                             FILES ${VoxLib_HEADERS_IO})
SOURCE_GROUP("Header Files\\Image"                          FILES ${VoxLib_HEADERS_IMAGE})
SOURCE_GROUP("Header Files\\Plugin"                         FILES ${VoxLib_HEADERS_PLUGIN})
SOURCE_GROUP("Header Files\\Video"                          FILES ${VoxLib_HEADERS_VIDEO})

#===============================================#
#                  SOURCE FILES                 #
#===============================================#

Set(VoxLib_SOURCES_ACTION
    Action/Action.cpp
    Action/ActionManager.cpp
)

Set(VoxLib_SOURCES_BITMAP
    Bitmap/Bitmap.cpp
)

Set(VoxLib_SOURCES_CORE
    Core/Devices.cpp
    Core/Functors.cpp
    Core/Logging.cpp
    Core/System.cpp
)

Set(VoxLib_SOURCES_CORE_GEOMETRY
    Core/Geometry/Color.cpp
)

Set(VoxLib_SOURCES_ERROR
    Error/Error.cpp
    Error/CudaError.cpp
)

Set(VoxLib_SOURCES_IO
    IO/MimeTypes.cpp
    IO/Resource.cpp
    IO/ResourceHelper.cpp
    IO/ResourceId.cpp
)

Set(VoxLib_SOURCES_PLUGIN
    Plugin/Plugin.cpp
    Plugin/PluginManager.cpp
)

Set(VoxLib_SOURCES_VIDEO
    Video/VidStream.cpp
)

set(VoxLib_SOURCES
    ${VoxLib_SOURCES_ACTION}
    ${VoxLib_SOURCES_BITMAP}
    ${VoxLib_SOURCES_CORE}
    ${VoxLib_SOURCES_CORE_GEOMETRY}
    ${VoxLib_SOURCES_ERROR}
    ${VoxLib_SOURCES_IO}
    ${VoxLib_SOURCES_IMAGE}
    ${VoxLib_SOURCES_PLUGIN}
    ${VoxLib_SOURCES_VIDEO}
)

SOURCE_GROUP("Source Files\\Action"                         FILES ${VoxLib_SOURCES_ACTION})
SOURCE_GROUP("Source Files\\Bitmap"                         FILES ${VoxLib_SOURCES_BITMAP})
SOURCE_GROUP("Source Files\\Core"                           FILES ${VoxLib_SOURCES_CORE})
SOURCE_GROUP("Source Files\\Core\\Geometry"                 FILES ${VoxLib_SOURCES_CORE_GEOMETRY})
SOURCE_GROUP("Source Files\\Plugin"                         FILES ${VoxLib_SOURCES_PLUGIN})
SOURCE_GROUP("Source Files\\Error"                          FILES ${VoxLib_SOURCES_ERROR})
SOURCE_GROUP("Source Files\\IO"                             FILES ${VoxLib_SOURCES_IO})
SOURCE_GROUP("Source Files\\Image"                          FILES ${VoxLib_SOURCES_IMAGE})
SOURCE_GROUP("Source Files\\Video"                          FILES ${VoxLib_SOURCES_VIDEO})

#===============================================#
#                  LIBRARY                      #
#===============================================#

CUDA_ADD_LIBRARY(VoxLib SHARED
                 ${VoxLib_SOURCES} 
                 ${VoxLib_HEADERS})

TARGET_LINK_LIBRARIES(VoxLib ${Boost_LIBRARIES})






