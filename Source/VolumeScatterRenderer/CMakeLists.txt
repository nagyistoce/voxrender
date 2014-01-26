#===========================================================================#
#                                                                           #
#   Project: VoxRender - Volume Scatter Renderer                            #
#                                                                           #
#   Description: Performs real-time volume rendering with a single          #
#                scattering integrator                                      #
#                                                                           #
#       Copyright (C) 2012-2014 Lucas Sherman                               #
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

MESSAGE(STATUS "Configuring VolumeScatterRenderer")

INCLUDE_DIRECTORIES(.)

#===============================================#
#                  HEADER FILES                 #
#===============================================#

Set(HEADERS_CORE
    Core/Common.h
    Core/VolumeScatterRenderer.h
    Core/Version.h
)

Set(HEADERS_CORE_IMPL
    Core/CBuffer.h
    Core/CRandomGenerator.h
    Core/CSampleBuffer.h
    Core/Intersect.h
)

Set(HEADERS_SCENE_IMPL
    Scene/CCamera.h
    Scene/CClipGeometry.h
    Scene/CLight.h
    Scene/CTransferBuffer.h
    Scene/CVolumeBuffer.h
    Scene/CRenderParams.h
)

Set(HEADERS_CLIP_IMPL
    Clip/CClipGroup.cuh
    Clip/CClipGroup.h
    Clip/CClipPlane.cuh
    Clip/CClipPlane.h
)

Set(HEADERS_KERNELS
    Kernels/VolumeHistogramKernel.h
    Kernels/RandKernel.h
    Kernels/RenderKernel.h
    Kernels/TonemapKernel.h
)

set(HEADER_FILES
    ${HEADERS_CORE}
    ${HEADERS_CORE_IMPL}
    ${HEADERS_CLIP_IMPL}
    ${HEADERS_SCENE_IMPL}
    ${HEADERS_KERNELS}
)

SOURCE_GROUP("Header Files\\Core"        FILES ${HEADERS_CORE})
SOURCE_GROUP("Header Files\\Core\\Impl"  FILES ${HEADERS_CORE_IMPL})
SOURCE_GROUP("Header Files\\Scene\\Impl" FILES ${HEADERS_SCENE_IMPL})
SOURCE_GROUP("Header Files\\Clip\\Impl"  FILES ${HEADERS_CLIP_IMPL})
SOURCE_GROUP("Header Files\\Kernels"     FILES ${HEADERS_KERNELS})

#===============================================#
#                  SOURCE FILES                 #
#===============================================#

Set(SOURCES_CORE
    Core/VolumeScatterRenderer.cpp
    Core/FastRand.cpp
    Core/CBuffer.cu
)

Set(SOURCES_CLIP
    Clip/CClipGroup.cpp
    Clip/CClipPlane.cpp
)

Set(SOURCES_SCENE
    Scene/CCamera.cpp
    Scene/CClipGeometry.cpp
    Scene/CLight.cpp
    Scene/CRenderParams.cpp
    Scene/CVolumeBuffer.cpp
    Scene/CTransferBuffer.cpp
)

Set(SOURCES_KERNELS
    Kernels/VolumeHistogramKernel.cu
    Kernels/RandKernel.cu
    Kernels/RenderKernel.cu
    Kernels/TonemapKernel.cu
)

set(SOURCE_FILES
    ${SOURCES_CORE}
    ${SOURCES_SCENE}
    ${SOURCES_KERNELS}
    ${SOURCES_CLIP}
)

SOURCE_GROUP("Source Files\\Core"       FILES ${SOURCES_CORE})
SOURCE_GROUP("Source Files\\Scene"      FILES ${SOURCES_SCENE})
SOURCE_GROUP("Source Files\\Clip"       FILES ${SOURCES_CLIP})
SOURCE_GROUP("Source Files\\Kernels"    FILES ${SOURCES_KERNELS})

#===============================================#
#            LIBRARY CONFIGURATION              #
#===============================================#

CUDA_ADD_LIBRARY(VolumeScatterRenderer SHARED
                 ${SOURCE_FILES} 
                 ${HEADER_FILES})
				 
#===============================================#
#             Project Dependencies              #
#===============================================#

ADD_DEPENDENCIES(VolumeScatterRenderer VoxLib)
TARGET_LINK_LIBRARIES(VolumeScatterRenderer 
                      ${VOX_LIB} 
                      ${Boost_LIBRARIES} 
                      ${OPENGL_LIBRARY}
                      ${CUDA_LIBRARIES})







