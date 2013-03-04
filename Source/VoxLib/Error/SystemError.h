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
#include "Core/CudaCommon.h"
#include "Error/Error.h"
#include "Error/ErrorCodes.h"

// API namespace
namespace vox
{

    /**
     * VoxRender's System Error Exception Class
     *
     * This exception is thrown by the library when a system level 
     * runtime error has occured. The system error code and status
     * message will be generated automatically. As such, it is best
     * to throw the exception as soon as possible following the 
     * occurrence of the error. If not, the last error information
     * pertaining to the current context may be inadvertantly overwritten.
     */ 
    class VOX_EXPORT VOX_HOST SystemError : public Error
    {
    public:
        /** 
         * Constructor - Initializes the exception content
         *
         * @param _file     The file from which the error was thrown (use the __FILE__ macro)
         * @param _line     The line from which the error was thrown (use the __LINE__ macro)
         * @param _category The system associated with the exception (internally Vox)
         * @param _msg      The error message to be returned when what() is called.
         * @param _code     The error code associated with the Error
         */
        SystemError( const char* _file, int _line, const char* _category, 
            std::string const& _msg, int _code = Error_Unknown ) : 
          Error( _file, _line, _category, _msg, _code )
        {  
            #if   defined __WIN32
            #elif defined __APPLE__
            #elif defined __GNUC__
            #else
                systemMessage = "Host platform not detected";
                systemCode = 0;
            #endif

            message += ": " + systemMessage;
        }

        int systemCode;             ///< The system error code 
        std::string systemMessage;  ///< System error message
    };

}

// End definition
#endif // VOX_SYSTEM_ERROR