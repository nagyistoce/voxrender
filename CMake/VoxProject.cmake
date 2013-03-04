#===========================================================================#
#                                                                           #
#   Project: VoxRender - CMake Project Generator                            #
#                                                                           #
#   Description: Defines a project generation macro                         #
#                                                                           #
#       Copyright (C) 2012 Lucas Sherman                                    #
#                                                                           #
#	Lucas Sherman, email: LucasASherman@gmail.com                           #
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

INCLUDE (CMakeParseArguments)

FUNCTION(VOX_PROJECT)

    SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)
	
	# Parameters:
	# SHARED, STATIC, EXE -- Library type
	# TARGET              -- Name of the target library
	# FOLDER              -- Visual Studio filter folder
	# HEADERS, SOURCES    -- Source code files
	# QT4_...             -- QT Resource, Moc header, and UI form files
	# DEPENDENCIES        -- Internal library dependencies
	# LIBRARIES           -- External library dependencies
	
    # Parse the project generation arguments 
    SET(options        SHARED STATIC EXE)
    SET(oneValueArgs   TARGET FOLDER)
    SET(multiValueArgs HEADERS SOURCES QT4_UIS QT4_RCS QT4_MOC DEPENDENCIES LIBRARIES)
    CMAKE_PARSE_ARGUMENTS("VOX_PROJECT" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

	# Display begin configuration message
	MESSAGE(STATUS "Configuring " ${VOX_PROJECT_TARGET})
	
	# Configure to add the current directory to include pths
	INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}) 
	
    # Configure the QT file build rules if applicable
	IF(QT4_FOUND)
        QT4_ADD_RESOURCES( resource_files_generated ${VOX_PROJECT_QT4_RCS} )
        QT4_WRAP_UI( header_files_generated ${VOX_PROJECT_QT4_UIS} )
        QT4_WRAP_CPP( source_files_generated ${VOX_PROJECT_QT4_MOC} )
        
        SOURCE_GROUP("Resource Files" FILES ${VOX_PROJECT_QT4_RCS})
        SOURCE_GROUP("Form Files"     FILES ${VOX_PROJECT_QT4_UIS})
        SOURCE_GROUP("Header Files"   FILES ${VOX_PROJECT_QT4_MOC})
	
        SOURCE_GROUP("Resource Files\\Generated" FILES ${resource_files_generated})
        SOURCE_GROUP("Header Files\\Generated"   FILES ${header_files_generated})
        SOURCE_GROUP("Source Files\\Generated"   FILES ${source_files_generated})
    ENDIF(QT4_FOUND)
	
	# Visual Studio solution filter arrangement
	SOURCE_GROUP("Header Files" FILES ${VOX_PROJECT_HEADERS})
	SOURCE_GROUP("Source Files" FILES ${VOX_PROJECT_SOURCES})

	# Compile full listing of project source files
	SET(ALL_FILES ${VOX_PROJECT_HEADERS} ${VOX_PROJECT_SOURCES}
                  ${VOX_PROJECT_QT4_RCS} ${VOX_PROJECT_QT4_UIS}
                  ${VOX_PROJECT_QT4_MOC})
	
	# Generate the project 
    IF(${VOX_PROJECT_EXE})
        CUDA_ADD_EXECUTABLE(${VOX_PROJECT_TARGET} 
                            ${VOX_PROJECT_HEADERS} 
                            ${VOX_PROJECT_SOURCES} 
                            ${header_files_generated} 
							${source_files_generated})
	ELSEIF(${VOX_PROJECT_STATIC})
	    CUDA_ADD_LIBRARY(${VOX_PROJECT_TARGET} STATIC
                            ${VOX_PROJECT_HEADERS} 
                            ${VOX_PROJECT_SOURCES} 
                            ${header_files_generated} 
							${source_files_generated})
	ELSEIF(${VOX_PROJECT_SHARED})
	    CUDA_ADD_LIBRARY(${VOX_PROJECT_TARGET} SHARED
                            ${VOX_PROJECT_HEADERS} 
                            ${VOX_PROJECT_SOURCES} 
                            ${header_files_generated} 
							${source_files_generated})
    ENDIF()
	
    # Set the folder path for the visual studio solution in the solution browser
    SET_PROPERTY(TARGET ${VOX_PROJECT_TARGET} PROPERTY FOLDER ${VOX_PROJECT_FOLDER})
		#PATH
    # Register project dependencies
    FOREACH(DEP ${VOX_PROJECT_DEPENDENCIES})
        GET_FILENAME_COMPONENT(TYPE ${DEP} EXT)
		IF(${TYPE} MATCHES "")
            ADD_DEPENDENCIES(${VOX_PROJECT_TARGET} ${DEP})
            TARGET_LINK_LIBRARIES(${VOX_PROJECT_TARGET}
                optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/${DEP}.lib
				debug     ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/${DEP}.lib
				)
        ELSE()
            TARGET_LINK_LIBRARIES(${VOX_PROJECT_TARGET} ${DEP})
        ENDIF()
	ENDFOREACH()
	
	# Target additional libraries for linking
	TARGET_LINK_LIBRARIES(${VOX_PROJECT_TARGET} ${VOX_PROJECT_LIBRARIES})

	# Display end configuration message
	MESSAGE(STATUS "Configuration of " ${VOX_PROJECT_TARGET} " complete.")
	
ENDFUNCTION() 




















































