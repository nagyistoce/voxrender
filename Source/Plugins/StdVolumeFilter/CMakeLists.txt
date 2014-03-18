#===========================================================================#
#                                                                           #
#   Project: StdVolumeFilter                                                #
#                                                                           #
#   Description: Exposes some standard filters provided by the volt library #
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
    TARGET "StdVolumeFilter" SHARED
    FOLDER "Plugins"
    HEADERS
        "Common.h"
        "Plugin.h"
        "Version.h"
        "Filters.h"
        "Views.h"
    SOURCES
        "Plugin.cpp"
        "Filters.cpp"
        "Views.cpp"
    DEPENDENCIES
        VoxLib
        VoxVolt
        VoxScene
    LIBRARIES
        ${Boost_LIBRARIES}
    )
