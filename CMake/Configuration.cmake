#===========================================================================#
#                                                                           #
#   Project: VoxRender - CMake Configuration                                #
#                                                                           #
#   Description: Sets the general build options for the project             #
#                                                                           #
#       Copyright (C) 2012 Lucas Sherman                                    #
#                                                                           #
#   Lucas Sherman, email: LucasASherman@gmail.com                           #
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
#              BUILD CONFIGURATIONS             #
#===============================================#

SET(CMAKE_CONFIGURATION_TYPES "Debug;Release" 
        CACHE STRING "limit configs" FORCE)

IF( "${CMAKE_SIZEOF_VOID_P}" EQUAL "4" )
    SET( PLATFORM "x86" )
ELSEIF( "${CMAKE_SIZEOF_VOID_P}" EQUAL "8" )
    SET( PLATFORM "x64" )
ELSE( )
    MESSAGE( ERROR "Unable to detect platform: try clearing cache" )
    SET( PLATFORM "Unknown" )
ENDIF( )

MESSAGE( STATUS "Target Platform: " "${PLATFORM}" )

#===============================================#
#             USER BUILD OPTIONS                #
#===============================================#

# Build dynamic or static link libraries
option(VoxLib_SHARED       "On to build shared libraries." ON)		 
option(CUDARenderer_SHARED "On to build shared libraries." ON)

# Enable DOXYGEN Generation of project documentation files
option(GENERATE_DOCUMENTATION "Generate project documentation" ON)



