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

MESSAGE(STATUS "Configuring CudaRenderer")

INCLUDE_DIRECTORIES(.)

#===============================================#
#                  HEADER FILES                 #
#===============================================#

Set(CUDARenderer_HEADERS_CORE
    Core/Common.h
    Core/CudaRenderer.h
    Core/Version.h
)

Set(CUDARenderer_HEADERS_CORE_IMPL
    Core/CBuffer.h
    Core/CRandomBuffer.h
    Core/CRandomGenerator.h
    Core/CSampleBuffer.h
)

Set(CUDARenderer_HEADERS_SCENE_IMPL
    Scene/CCamera.h
    Scene/CFilm.h
    Scene/CLight.h
    Scene/CVolume.h
)

set(CUDARenderer_HEADERS_OPTIX_IMPL
    Optix/RayTypes.h
)

set(CUDARenderer_HEADERS
    ${CUDARenderer_HEADERS_CORE}
    ${CUDARenderer_HEADERS_CORE_IMPL}
    ${CUDARenderer_HEADERS_OPTIX_IMPL}
    ${CUDARenderer_HEADERS_SCENE_IMPL}
)

SOURCE_GROUP("Header Files\\Core"        FILES ${CUDARenderer_HEADERS_CORE})
SOURCE_GROUP("Header Files\\Core\\Impl"  FILES ${CUDARenderer_HEADERS_CORE_IMPL})
SOURCE_GROUP("Header Files\\Optix\\Impl" FILES ${CUDARenderer_HEADERS_OPTIX_IMPL})
SOURCE_GROUP("Header Files\\Scene\\Impl" FILES ${CUDARenderer_HEADERS_SCENE_IMPL})

#===============================================#
#                  SOURCE FILES                 #
#===============================================#

Set(CUDARenderer_SOURCES_CORE
    Core/CudaRenderer.cpp
)

Set(CUDARenderer_SOURCES_CORE_IMPL
    Core/CBuffer.cu
)

Set(CUDARenderer_SOURCES_SCENE_IMPL
    Scene/CCamera.cpp
)

set(CUDARenderer_SOURCES
    ${CUDARenderer_SOURCES_CORE}
    ${CUDARenderer_SOURCES_CORE_IMPL}
    ${CUDARenderer_SOURCES_SCENE_IMPL}
)

SOURCE_GROUP("Source Files\\Core"       FILES ${CUDARenderer_SOURCES_CORE})
SOURCE_GROUP("Source Files\\Core\\Impl" FILES ${CUDARenderer_SOURCES_CORE_IMPL})
SOURCE_GROUP("Source Files\\Scene"      FILES ${CUDARenderer_SOURCES_SCENE_IMPL})

#===============================================#
#           Optix PTX Source Files              #
#===============================================#

Set(CUDARenderer_PTXS
    
)
	
CUDA_WRAP_SRCS(CUDARenderer PTX generated_files "OptiX/Programs.cu" "Optix/Geometry/Parallelogram.cu" "Optix/Materials/Checker.cu" "Optix/Materials/Phong.cu" "Optix/Materials/Translucent.cu" "Optix/Geometry/TriangleMesh.cu" "Optix/Geometry/TriangleMeshSmall.cu" "Optix/Geometry/Sphere.cu")

SOURCE_GROUP("PTX Files"            FILES ${CUDARenderer_PTXS})
SOURCE_GROUP("PTX Files\\generated" FILES ${generated_files})
	
#===============================================#
#            LIBRARY CONFIGURATION              #
#===============================================#

IF(CUDARenderer_SHARED)
	SET(Linkage SHARED)
ELSE()
	SET(Linkage STATIC)
ENDIF()

CUDA_ADD_LIBRARY(CUDARenderer ${Linkage}
                 ${CUDARenderer_SOURCES} 
                 ${CUDARenderer_HEADERS}
                 ${CUDARenderer_PTXS}
                 ${generated_files})
				 
#===============================================#
#             Project Dependencies              #
#===============================================#

ADD_DEPENDENCIES(CUDARenderer VoxLib)
TARGET_LINK_LIBRARIES(CUDARenderer 
                      ${VOX_LIB} 
                      ${Boost_LIBRARIES} 
                      ${GLEW_LIBRARY}
                      ${OPENGL_LIBRARY}
                      optix ${optix_rpath})







