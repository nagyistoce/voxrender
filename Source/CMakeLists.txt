#===========================================================================#
#                                                                           #
#   Project: VoxRender - GPU Based Real-Time Volume Rendering               #
#                                                                           #
#   Description: Performs real-time rendering of volume data using volume   #
#		ray casting techniques.                                     #
#                                                                           #
#       Copyright (C) 2012 Lucas Sherman                                    #
#                                                                           #
#	Lucas Sherman, email: LucasASherman@gmail.com                       #
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

INCLUDE_DIRECTORIES(.)

#===============================================#
#               VOXRENDER LIBRARY               #
#===============================================#

IF( VoxLib_SHARED )
    ADD_DEFINITIONS(-DVoxLib_SHARED)
ENDIF( )

ADD_SUBDIRECTORY(VoxLib)

SET(VOX_LIB optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/VoxLib.lib
            debug     ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/VoxLib.lib)

#===============================================#
#            VOLUME SCATTER RENDERER            #
#===============================================#

ADD_SUBDIRECTORY(VolumeScatterRenderer)

SET(VSR_LIB optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/VolumeScatterRenderer.lib
            debug     ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/VolumeScatterRenderer.lib)
	
#===============================================#
#                 VOLT LIBRARY                  #
#===============================================#

ADD_SUBDIRECTORY(VoxVolt)

SET(VOX_VOLT_LIB optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/Volt.lib
                 debug     ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/Volt.lib)
	
#===============================================#
#              VOXRENDER APPLICATION            #
#===============================================#
ADD_SUBDIRECTORY(VoxRender)

#===============================================#
#                   WEB CLIENTS                 #
#===============================================#
ADD_SUBDIRECTORY(WebClients)

#===============================================#
#                    PLUGINS                    #
#===============================================#
ADD_SUBDIRECTORY(Plugins)

#===============================================#
#                   UNIT TESTS                  #
#===============================================#
ADD_SUBDIRECTORY(UnitTests)





























