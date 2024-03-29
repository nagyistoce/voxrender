#===========================================================================#
#                                                                           #
#   Project: VoxRender - GPU Based Real-Time Volume Rendering               #
#                                                                           #
#   Description: Performs real-time rendering of volume data using volume   #
#       ray casting techniques.                                             #
#                                                                           #
#       Copyright (C) 2012-2014 Lucas Sherman                               #
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
#                 QT5 Dependency                #
#===============================================#

# Find the QtWidgets library
find_package(Qt5Widgets)

IF(Qt5Widgets_FOUND)
    MESSAGE( STATUS "Qt library directory: " ${Qt5Widgets_LIBRARY_DIR} )
    MESSAGE( STATUS "Qt include directory: " ${Qt5Widgets_INCLUDE_DIRS} )
    MESSAGE( STATUS "Qt binary directory: "  ${Qt5Widgets_BINARY_DIR} )
    INCLUDE_DIRECTORIES(${QT_USE_FILE})

    # ADD_DEFINITIONS(-DQT_NO_KEYWORDS) # The foreach macro breaks alot of 3rd party code

    #===============================================#
    #                  ENTRY POINT                 #
    #===============================================#
    SET(VOXQTGUI_MAIN Interface/main.cpp)
    SOURCE_GROUP("Source Files" FILES ${VOXQTGUI_MAIN})

    #===============================================#
    #                 SOURCE FILES                  #
    #===============================================#
    SET(VOXQTGUI_ACTION_SRCS
        Actions/CamEditAct.cpp
        Actions/MaterialEditAct.cpp
        Actions/AddRemLightAct.cpp
        Actions/AddRemClipAct.cpp
        Actions/AddRemKeyAct.cpp
    )

    #===============================================#
    #                 HEADER FILES                  #
    #===============================================#
    SET(VOXQTGUI_ACTION_HDRS
        Actions/CamEditAct.h
        Actions/MaterialEditAct.h
        Actions/AddRemLightAct.h
        Actions/AddRemClipAct.h
        Actions/AddRemKeyAct.h
    )

    SOURCE_GROUP("Header Files\\Actions"   FILES ${VOXQTGUI_ACTION_HDRS})
    SOURCE_GROUP("Source Files\\Actions"   FILES ${VOXQTGUI_ACTION_SRCS})

    #===============================================#
    #                 Source Files                  #
    #===============================================#
    SET(VOXQTGUI_SRCS
        Extensions/QColorPushButton.cpp
        Interface/aboutdialogue.cpp
        Interface/ambientlightwidget.cpp
        Interface/animatewidget.cpp
        Interface/animateview.cpp
        Interface/arealightwidget.cpp
        Interface/camerawidget.cpp
        Interface/clipdialogue.cpp
        Interface/clipplanewidget.cpp
        Interface/clipwidget.cpp
        Interface/edgeitem.cpp
        Interface/griditem.cpp
        Interface/genericdialogue.cpp
        Interface/animateitem.cpp
        Interface/histogramgenerator.cpp
        Interface/histogramview.cpp
        Interface/histogramwidget.cpp
        Interface/infowidget.cpp
        Interface/lightdialogue.cpp
        Interface/lightingwidget.cpp
        Interface/mainwindow.cpp
        Interface/nodeitem.cpp
        Interface/panewidget.cpp
        Interface/pluginwidget.cpp
        Interface/pointlightwidget.cpp
        Interface/renderview.cpp
        Interface/samplingwidget.cpp
        Interface/timingwidget.cpp
        Interface/transferitem.cpp
        Interface/transferwidget.cpp
        Interface/voxapp.cpp
        )
	SOURCE_GROUP("Source Files" FILES ${VOXQTGUI_SRCS})

    #===============================================#
    #                 HEADER FILES                  #
    #===============================================#
    SET(VOXQTGUI_MOC
        Extensions/QColorPushButton.h
        Interface/aboutdialogue.h
        Interface/ambientlightwidget.h
        Interface/animatewidget.h
        Interface/animateview.h
        Interface/arealightwidget.h
        Interface/camerawidget.h
        Interface/clipdialogue.h
        Interface/clipplanewidget.h
        Interface/clipwidget.h
        Interface/edgeitem.h
        Interface/animateitem.h
        Interface/griditem.h
        Interface/genericdialogue.h
        Interface/histogramgenerator.h
        Interface/histogramview.h
        Interface/histogramwidget.h
        Interface/infowidget.h
        Interface/lightdialogue.h
        Interface/lightingwidget.h
        Interface/mainwindow.h
        Interface/nodeitem.h
        Interface/panewidget.h
        Interface/pluginwidget.h
        Interface/pointlightwidget.h
        Interface/renderview.h
        Interface/samplingwidget.h
        Interface/timingwidget.h
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
        Interface/ambientlightwidget.ui
        Interface/animatewidget.ui
        Interface/arealightwidget.ui
        Interface/camerawidget.ui
        Interface/clipdialogue.ui
        Interface/clipplanewidget.ui
        Interface/histogramwidget.ui
        Interface/genericdialogue.ui
        Interface/infowidget.ui
        Interface/lightdialogue.ui
        Interface/mainwindow.ui
        Interface/panewidget.ui
        Interface/pluginwidget.ui
        Interface/pointlightwidget.ui
        Interface/samplingwidget.ui
        Interface/timingwidget.ui
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
        ${VOXQTGUI_UI_HDRS} "interface/utilities.h" "interface/utilities.cpp"
        ${VOXQTGUI_ACTION_HDRS} ${VOXQTGUI_ACTION_SRCS})

    #===============================================#
    #            QT5 Dependency Modules             #
    #===============================================#
	qt5_use_modules(VoxRender Widgets)
	
    #===============================================#
    #             Project Dependencies              #
    #===============================================#
    ADD_DEPENDENCIES(VoxRender VoxLib VoxScene VoxVolt VolumeScatterRenderer)

    #===============================================#
    #            TARGET LINK LIBRARIES              #
    #===============================================#
    TARGET_LINK_LIBRARIES(VoxRender ${VOX_LIB} ${VOX_SCENE} ${VOLT_LIB} ${VSR_LIB} ${Boost_LIBRARIES})

ELSE(Qt5Widgets_FOUND)
	MESSAGE( STATUS "Warning : could not find Qt - not building Qt GUI")
ENDIF(Qt5Widgets_FOUND)












