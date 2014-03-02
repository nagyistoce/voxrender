#===========================================================================#
#                                                                           #
#   Project: SocketIO                                                       #
#                                                                           #
#   Description: Provides an IO module for low level socket IO              #
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
    TARGET "SocketIO" SHARED
    FOLDER "Plugins"
    HEADERS
        "Common.h"
        "TcpSocketIO.h"
        "UdpSocketIO.h"
        "Plugin.h"
        "Version.h"
    SOURCES
        "TcpSocketIO.cpp"
        "UdpSocketIO.cpp"
        "Plugin.cpp"
    DEPENDENCIES
        VoxLib
    LIBRARIES
        ${Boost_LIBRARIES}
    )
