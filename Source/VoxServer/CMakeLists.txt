#===========================================================================#
#                                                                           #
#   Project: VoxServer                                                      #
#                                                                           #
#   Description: Exports a C interface for rendering scene files            #
#                                                                           #
#   Copyright (C) 2014 Lucas Sherman                                        #
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

VOX_PROJECT(
    TARGET "VoxServer" SHARED
    HEADERS
        "Interface.h"
        "Common.h"
        "Version.h"
        "Base64.h"
        "WebSocket.h"
    SOURCES
        "Interface.cpp"
        "Base64.cpp"
        "WebSocket.cpp"
    DEPENDENCIES
        VoxLib
        VoxScene
        VolumeScatterRenderer
    LIBRARIES
        ${Boost_LIBRARIES}
    )
