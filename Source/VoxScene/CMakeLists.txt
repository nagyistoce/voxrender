#===========================================================================#
#                                                                           #
#   Project: VoxScene                                                       #
#                                                                           #
#   Description: Defines classes encapsulating scene graph elements         #
#                                                                           #
#       Copyright (C) 2014 Lucas Sherman                                    #
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

MESSAGE(STATUS "Configuring VoxScene")

INCLUDE_DIRECTORIES(.)

VOX_PROJECT(
    TARGET "VoxScene" SHARED
    HEADERS
        Version.h
        Common.h
        Animator.h
        Camera.h
        Light.h
        Material.h
        Primitive.h
        PrimGroup.h
        RenderParams.h
        Scene.h
        Transfer.h
        TransferMap.h
        Volume.h
        FrameBuffer.h
        Renderer.h
        RenderController.h
        RenderThread.h
    SOURCES
        Animator.cpp
        Camera.cpp
        Light.cpp
        Material.cpp
        Primitive.cpp
        PrimGroup.cpp
        Scene.cpp
        Transfer.cpp
        TransferMap.cpp
        Volume.cpp
        RenderController.cpp
        Renderer.cpp
        RenderThread.cpp
    DEPENDENCIES
        VoxLib
    LIBRARIES
        ${Boost_LIBRARIES}
    )






