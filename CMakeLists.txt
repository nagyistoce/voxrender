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

#===============================================#
#                  INITIAL SETUP                #
#===============================================#

MESSAGE(STATUS "CMAKE VERSION DETECTED " ${CMAKE_VERSION})

cmake_minimum_required(VERSION 2.8)

PROJECT(VoxRender)

SET(CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/CMake")
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_DIR}") 
SET(BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Binaries")
SET(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Source")

#===============================================#
#              BUILD CONFIGURATION              #
#===============================================#

MESSAGE(STATUS "Loading Configuration Information")
INCLUDE(Configuration)

#===============================================#
#               OUTPUT DIRECTORIES              #
#===============================================#

SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR}/${PLATFORM})
SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${BINARY_DIR}/${PLATFORM})
SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BINARY_DIR}/${PLATFORM})

SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BINARY_DIR}/${PLATFORM}/Release)
SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${BINARY_DIR}/${PLATFORM}/Release)
SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${BINARY_DIR}/${PLATFORM}/Release)

SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BINARY_DIR}/${PLATFORM}/Debug)
SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${BINARY_DIR}/${PLATFORM}/Debug)
SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${BINARY_DIR}/${PLATFORM}/Debug)

#===============================================#
#             GENERATE DOCUMENTATION            #
#===============================================#

IF(GENERATE_DOCUMENTATION)
    ADD_SUBDIRECTORY(Documentation)
ENDIF(GENERATE_DOCUMENTATION)

#===============================================#
#                 DEPENDENCIES                  #
#===============================================#

INCLUDE(Dependencies)

#===============================================#
#               CUSTOM FUNCTIONS                #
#===============================================#

INCLUDE(VoxProject)

#===============================================#
#                 SOURCE FILES                  #
#===============================================#

ADD_SUBDIRECTORY(Source)





























