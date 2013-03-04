#===========================================================================#
#                                                                           #
#   Project: VoxRender - CMake Documentation                                #
#                                                                           #
#   Description: Generates Doxygen documentation files for the project      #
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

FIND_PACKAGE(Doxygen)

IF (DOXYGEN_FOUND)
    MESSAGE( STATUS "Found Doxygen - adding 'Documentation' target" )
	
    SET(DOXYGEN_TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
    SET(DOXYGEN_INPUT ${CMAKE_CURRENT_BINARY_DIR}/doxygen.conf)
    SET(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    SET(DOXYGEN_OUTPUT ${DOXYGEN_OUTPUT_DIR}/html/index.html)
	
    MESSAGE( STATUS "Doxygen output: " ${DOXYGEN_OUTPUT} )
	
    #
    # Check for GraphViz's dot
    #
    IF(DOXYGEN_DOT_FOUND)
        MESSAGE( STATUS "Found GraphViz")
        SET(DOXYGEN_DOT_CONF "HAVE_DOT = YES")
    ENDIF(DOXYGEN_DOT_FOUND)
		
    #
    # Generate Build File
    #
    ADD_CUSTOM_COMMAND( 
        OUTPUT ${DOXYGEN_OUTPUT}
	
        # Create custom configurations file
        # :TODO: OS independant config file generation
        #COMMAND cp ${DOXYGEN_TEMPLATE} ${DOXYGEN_INPUT}
        #COMMAND echo "INPUT = " ${CMAKE_CURRENT_SOURCE_DIR} >> ${DOXYGEN_INPUT}
        #COMMAND echo "OUTPUT_DIRECTORY = " ${DOXYGEN_OUTPUT_DIR} >> ${DOXYGEN_INPUT}
        #COMMAND echo ${DOXYGEN_DOT_CONF} >> ${DOXYGEN_INPUT}

        # Launch Doxygen
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_TEMPLATE}
        DEPENDS ${DOXYGEN_TEMPLATE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    #
    # Register Doxygen Target
    # 
    ADD_CUSTOM_TARGET(Documentation DEPENDS ${DOXYGEN_OUTPUT})

ENDIF (DOXYGEN_FOUND)