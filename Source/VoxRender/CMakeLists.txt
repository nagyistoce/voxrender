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

FIND_PACKAGE(Qt4 4.8.0 COMPONENTS QtCore QtGui QtMain)

IF(QT4_FOUND)
    MESSAGE( STATUS "Qt library directory: " ${QT_LIBRARY_DIR} )
    MESSAGE( STATUS "Qt include directory: " ${QT_INCLUDE_DIR} )
    MESSAGE( STATUS "Qt binary directory: " ${QT_BINARY_DIR} )
    INCLUDE(${QT_USE_FILE})

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
        Interface/griditem.cpp
        Interface/histogramitem.cpp
        Interface/histogramview.cpp
        Interface/histogramwidget.cpp
        Interface/infowidget.cpp
		Interface/lightdialogue.cpp
        Interface/mainwindow.cpp
        Interface/nodeitem.cpp
        Interface/panewidget.cpp
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
        Interface/griditem.h
        Interface/histogramitem.h
        Interface/histogramview.h
        Interface/histogramwidget.h
        Interface/infowidget.h
		Interface/lightdialogue.h
        Interface/mainwindow.h
        Interface/nodeitem.h
        Interface/panewidget.h
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
    QT4_ADD_RESOURCES( VOXQTGUI_RC_SRCS ${VOXQTGUI_RCS} )
    QT4_WRAP_UI( VOXQTGUI_UI_HDRS ${VOXQTGUI_UIS} )
    QT4_WRAP_CPP( VOXQTGUI_MOC_SRCS ${VOXQTGUI_MOC} )
        
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
        COMMAND cmake -E copy "${QT_BINARY_DIR}/QtCore4.dll" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIGURATION>"
        COMMAND cmake -E copy "${QT_BINARY_DIR}/QtGui4.dll" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIGURATION>"
    )

    #===============================================#
    #             Project Dependencies              #
    #===============================================#
    ADD_DEPENDENCIES(VoxRender VoxLib CUDARenderer)

    #===============================================#
    #            TARGET LINK LIBRARIES              #
    #===============================================#

    TARGET_LINK_LIBRARIES(VoxRender ${VOX_LIB} ${VSR_LIB} ${CR_LIB} ${QT_LIBRARIES} 
                                    ${Boost_LIBRARIES} optix ${optix_rpath})

ELSE(QT4_FOUND)
	MESSAGE( STATUS "Warning : could not find Qt - not building Qt GUI")
ENDIF(QT4_FOUND)












