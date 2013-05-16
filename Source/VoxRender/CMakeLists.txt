#===========================================================================#
#                                                                           #
#   Project: VoxRender - GPU Based Real-Time Volume Rendering               #
#                                                                           #
#   Description: Performs real-time rendering of volume data using volume   #
#       ray casting techniques.                                             #
#                                                                           #
#       Copyright (C) 2012 Lucas Sherman                                    #
#                                                                           #
#    Lucas Sherman, email: LucasASherman@gmail.com                          #
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

MESSAGE(STATUS "Configuring VoxRender")

INCLUDE_DIRECTORIES(.)

#===============================================#
#                 QT4 Dependency                #
#===============================================#

# Find includes in corresponding build directories
#set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
# set(CMAKE_AUTOMOC ON)
# Find the QtWidgets library
find_package(Qt5Widgets)

IF(Qt5Widgets_FOUND)
    MESSAGE( STATUS "Qt library directory: " ${Qt55Widgets_LIBRARY_DIR} )
    MESSAGE( STATUS "Qt include directory: " ${Qt55Widgets_INCLUDE_DIR} )
    MESSAGE( STATUS "Qt binary directory: " ${Qt55Widgets_BINARY_DIR} )
    INCLUDE_DIRECTORIES(${QT_USE_FILE})
	
    #===============================================#
    #                  Entry Point                  #
    #===============================================#
    SET(VOXQTGUI_MAIN Interface/main.cpp)
    SOURCE_GROUP("Source Files" FILES ${VOXQTGUI_MAIN})

    #===============================================#
    #                 Source Files                  #
    #===============================================#
    SET(VOXQTGUI_SRCS
        Interface/aboutdialogue.cpp
        Interface/arealightwidget.cpp
        Interface/camerawidget.cpp
        Interface/edgeitem.cpp
        Interface/griditem.cpp
        Interface/histogramitem.cpp
        Interface/histogramview.cpp
        Interface/histogramwidget.cpp
        Interface/infowidget.cpp
		Interface/lightdialogue.cpp
        Interface/mainwindow.cpp
        Interface/nodeitem.cpp
        Interface/panewidget.cpp
		Interface/ambientlightwidget.cpp
        Interface/pointlightwidget.cpp
        Interface/renderview.cpp
        Interface/samplingwidget.cpp
        Interface/transferitem.cpp
        Interface/transferwidget.cpp
        Interface/voxapp.cpp
        )
	SOURCE_GROUP("Source Files" FILES ${VOXQTGUI_SRCS})

    #===============================================#
    #                 HEADER FILES                  #
    #===============================================#
    SET(VOXQTGUI_MOC
        Interface/aboutdialogue.h
	    Interface/arealightwidget.h
        Interface/camerawidget.h
        Interface/edgeitem.h
        Interface/griditem.h
        Interface/histogramitem.h
        Interface/histogramview.h
        Interface/histogramwidget.h
        Interface/infowidget.h
		Interface/lightdialogue.h
        Interface/mainwindow.h
        Interface/nodeitem.h
        Interface/panewidget.h
		Interface/ambientlightwidget.h
        Interface/pointlightwidget.h
        Interface/renderview.h
        Interface/samplingwidget.h
        Interface/transferitem.h
        Interface/transferwidget.h
        Interface/voxapp.h
        )
        SOURCE_GROUP("Header Files" FILES ${VOXQTGUI_MOC})

    #===============================================#
    #               QT FORM FILES                   #
    #===============================================#
    SET(VOXQTGUI_UIS
        Interface/aboutdialogue.ui
	    Interface/arealightwidget.ui
        Interface/camerawidget.ui
        Interface/histogramwidget.ui
        Interface/infowidget.ui
        Interface/lightdialogue.ui
        Interface/mainwindow.ui
        Interface/panewidget.ui
		Interface/ambientlightwidget.ui
        Interface/pointlightwidget.ui
        Interface/samplingwidget.ui
        Interface/transferwidget.ui
        )
        SOURCE_GROUP("Form Files" FILES ${VOXQTGUI_UIS})

    #===============================================#
    #              QT RESOURCE FILES                #
    #===============================================#
    SET(VOXQTGUI_RCS
        Interface/icons.qrc
        Interface/splash.qrc
        Interface/images.qrc
        )
        SOURCE_GROUP("Resource Files" FILES ${VOXQTGUI_RCS})

    #===============================================#
    #             QT GENERATED FILES                #
    #===============================================#
    QT5_ADD_RESOURCES( VOXQTGUI_RC_SRCS ${VOXQTGUI_RCS} )
    QT5_WRAP_UI( VOXQTGUI_UI_HDRS ${VOXQTGUI_UIS} )
    QT5_WRAP_CPP( VOXQTGUI_MOC_SRCS ${VOXQTGUI_MOC} )
        
    SOURCE_GROUP("Resource Files\\Generated" FILES ${VOXQTGUI_RC_SRCS})
    SOURCE_GROUP("Header Files\\Generated"   FILES ${VOXQTGUI_UI_HDRS})
    SOURCE_GROUP("Source Files\\Generated"   FILES ${VOXQTGUI_MOC_SRCS})

    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}) 

    #===============================================#
    #                  EXECUTABLE                   #
    #===============================================#
    CUDA_ADD_EXECUTABLE( VoxRender ${VOXQTGUI_MAIN} ${VOXQTGUI_SRCS} 
        ${VOXQTGUI_MOC} ${VOXQTGUI_MOC_SRCS} ${VOXQTGUI_RC_SRCS} 
        ${VOXQTGUI_UI_HDRS} "interface/utilities.h" "interface/utilities.cpp")

    INCLUDE_DIRECTORIES(../VoxLib)

    #===============================================#
    #               Qt DLL Dependency               #
    #===============================================#
    ADD_CUSTOM_COMMAND( 
        TARGET VoxRender
        POST_BUILD
	
        COMMAND cmake -E echo "Copying Qt DLLs to build directory: " 
        #COMMAND cmake -E copy "${QT_BINARY_DIR}/Qt5Widgets.dll" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIGURATION>"
        #COMMAND cmake -E copy "${QT_BINARY_DIR}/Qt5Core.dll" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIGURATION>"
        #COMMAND cmake -E copy "${QT_BINARY_DIR}/Qt5Gui.dll" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIGURATION>"
    )

    #===============================================#
    #            QT5 Dependency Modules             #
    #===============================================#
	qt5_use_modules(VoxRender Widgets)
	
    #===============================================#
    #             Project Dependencies              #
    #===============================================#
    ADD_DEPENDENCIES(VoxRender VoxLib VolumeScatterRenderer)

    #===============================================#
    #            TARGET LINK LIBRARIES              #
    #===============================================#
    TARGET_LINK_LIBRARIES(VoxRender ${VOX_LIB} ${VSR_LIB} ${Boost_LIBRARIES})

ELSE(Qt5Widgets_FOUND)
	MESSAGE( STATUS "Warning : could not find Qt - not building Qt GUI")
ENDIF(Qt5Widgets_FOUND)












