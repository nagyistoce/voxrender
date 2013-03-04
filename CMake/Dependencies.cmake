#===========================================================================#
#                                                                           #
#   Project: VoxRender - CMake Dependencies                                 #
#                                                                           #
#   Description: Checks the project dependencies and locates any includes.  #
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

#===============================================#
#                    INCLUDES                   #
#===============================================#

# Default include directories from build scripts

SET(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_SOURCE_DIR}/Includes/${PLATFORM}/boost_1_49_0)
SET(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_SOURCE_DIR}/Includes/${PLATFORM}/qt-everywhere-opensource-src-4.8.0)
SET(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_SOURCE_DIR}/Includes/${PLATFORM}/glew-1.5.5)
SET(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "C:/Program Files/NVIDIA Corporation/OptiX SDK 3.0.0")
SET(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_SOURCE_DIR}/Includes/${PLATFORM}/curl-7.29.0/builds/libcurl-vc10-x86-release-dll-ipv6-sspi-spnego-winssl)

#===============================================#
#                  NVIDIA CUDA                  #
#===============================================#

FIND_PACKAGE(CUDA REQUIRED)

message(STATUS "Cuda include directory: " "${CUDA_INCLUDE_DIRS}")
message(STATUS "Cuda library directory: " "${CUDA_LIBRARIES}")
INCLUDE_DIRECTORIES(SYSTEM ${CUDA_INCLUDE_DIRS})

#===============================================#
#                  NVIDIA OptiX                 #
#===============================================#

FIND_PACKAGE(OptiX REQUIRED)

INCLUDE_DIRECTORIES("${OptiX_INCLUDE}")
MESSAGE(STATUS "OptiX include directory: " "${OptiX_INCLUDE}")

#===============================================#
#                     OPENGL                    #
#===============================================#

FIND_PACKAGE(OpenGL)

IF(OPENGL_FOUND)
    MESSAGE(STATUS "OpenGL include directory: " "${OPENGL_INCLUDE_DIRS}")
    MESSAGE(STATUS "OpenGL library directory: " "${OPENGL_LIBRARY}")
    INCLUDE_DIRECTORIES(SYSTEM ${OPENGL_INCLUDE_DIRS})
ELSE(OPENGL_FOUND)
    MESSAGE(FATAL_ERROR "OpenGL not found.")
ENDIF(OPENGL_FOUND)

#===============================================#
#                      GLEW                     #
#===============================================#

IF(NOT APPLE)
    FIND_PACKAGE(GLEW)
ENDIF()

IF(GLEW_FOUND)
    ADD_DEFINITIONS(-DGLEW_STATIC)
    MESSAGE(STATUS "GLEW include directory: " "${GLEW_INCLUDE_PATH}")
    MESSAGE(STATUS "GLEW library directory: " "${GLEW_LIBRARY}")
    INCLUDE_DIRECTORIES(${GLEW_INCLUDE_PATH})
ELSE(GLEW_FOUND)
    MESSAGE(FATAL_ERROR "GLEW not found.")
ENDIF(GLEW_FOUND)

#===============================================#
#                       CURL                    #
#===============================================#

FIND_PACKAGE(libcurl)

IF(LIBCURL_LIB_FOUND)
    ADD_DEFINITIONS(-DCURL_STATICLIB) # Allow DLL binary option #
    MESSAGE(STATUS "CURL include directory: " "${LIBCURL_INCLUDE_DIR}")
    MESSAGE(STATUS "CURL library directory: " "${LIBCURL_LIBRARIES}")
    INCLUDE_DIRECTORIES(${LIBCURL_INCLUDE_DIR})
ELSE(LIBCURL_LIB_FOUND)
    MESSAGE(STATUS "warning: libcurl not found.")
ENDIF(LIBCURL_LIB_FOUND)

#===============================================#
#                      BOOST                    #
#===============================================#

IF(APPLE)
    SET(BOOST_ROOT ${OSX_DEPENDENCY_ROOT})
ENDIF(APPLE)

# Acceptable Version Info
SET(Boost_MINIMUM_VERSION "1.43")
SET(Boost_ADDITIONAL_VERSIONS "1.49.0" "1.49" "1.46.2" "1.46.1" "1.46.0" 
    "1.46" "1.45.0" "1.45" "1.44.0" "1.44" "1.43.0" "1.43")

# Required Component Libraries
SET(Boost_COMPONENTS thread program_options filesystem serialization 
    iostreams regex system date_time unit_test_framework)

# Set Library type
IF(WIN32)
    SET(Boost_COMPONENTS ${Boost_COMPONENTS} zlib bzip2)
    SET(Boost_USE_STATIC_LIBS ON)
    SET(Boost_USE_MULTITHREADED ON)
    SET(Boost_USE_STATIC_RUNTIME OFF)
ENDIF(WIN32)

FIND_PACKAGE(Boost ${Boost_MINIMUM_VERSION} COMPONENTS ${Boost_COMPONENTS} REQUIRED)

MESSAGE(STATUS "Boost library directory: " ${Boost_LIBRARY_DIRS})
MESSAGE(STATUS "Boost include directory: " ${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(SYSTEM ${Boost_INCLUDE_DIRS})