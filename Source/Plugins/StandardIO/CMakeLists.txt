#===========================================================================#
#                                                                           #
#   Project: StandardIO package for the VoxIO library                       #
#                                                                           #
#   Description: Wraps libcurl to provide an IO module with support for     #
#       most common protocols.                                              #
#                                                                           #
#   Copyright (C) 2013 Lucas Sherman                                        #
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

INCLUDE_DIRECTORIES(.)

IF(LIBCURL_LIB_FOUND)
    VOX_PROJECT(
        TARGET "StandardIO" SHARED
        FOLDER "Plugins"
		HEADERS
            "Common.h"
            "CurlStreamBuf.h"
            "StandardIO.h"
            "Version.h"
			"Plugin.h"
        SOURCES
            "StandardIO.cpp"
			"Plugin.cpp"
        DEPENDENCIES
            VoxLib
        LIBRARIES
            ${Boost_LIBRARIES}
            ${LIBCURL_LIBRARIES}
        )
ELSE(LIBCURL_LIB_FOUND)
    MESSAGE(STATUS "warning: not building StandardIO (libcurl not found)")
ENDIF(LIBCURL_LIB_FOUND)
