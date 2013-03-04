/* ===========================================================================

	Project: VoxRender - Error

	Defines the base exception class from which all other internal exception
    objects derive. This base class itself derives from std::exception.

    Copyright (C) 2012 Lucas Sherman

	Lucas Sherman, email: LucasASherman@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

=========================================================================== */

// Begin definition
#ifndef VOX_SYSTEM_ERROR
#define VOX_SYSTEM_ERROR

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Format.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/Error/ErrorCodes.h"

// API namespace
namespace vox
{

    /**
     * VoxRender's Filesystem Error Exception Class
     *
     * This exception is thrown by the library when a filesystem level 
     * runtime error has occured. The exception takes an additional
     * parameter specifying the filename which will be appended to the
     * message in a coherent fashion. [ msg = msg + " " + file ]
     */ 
    class VOX_EXPORT FileError : public Error
    {
    public:
        /** 
         * Constructor - Initializes the exception content
         *
         * @param _file     The file from which the error was thrown (use the __FILE__ macro)
         * @param _line     The line from which the error was thrown (use the __LINE__ macro)
         * @param _category The system associated with the exception (internally Vox)
         * @param _msg      The error message to be returned when what() is called.
         * @param _path     The path to the file, if relevant. <<Piped, etc otherwise
         * @param _code     The error code associated with the Error
         */
        FileError( const char* _file, int _line, const char* _category, 
            std::string const& _msg, std::string const& _path,
            int _code = Error_Unknown ) : 
          Error( _file, _line, _category, _msg, _code ), 
          path(_path)
        {
            message += format(" [file=%1%]", path);
        }

        std::string path;   ///< Path to the file of interest
    };
}

// End definition
#endif // VOX_SYSTEM_ERROR