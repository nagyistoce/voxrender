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
#ifndef VOX_ERROR
#define VOX_ERROR

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Types.h"
#include "VoxLib/Error/ErrorCodes.h"

// API namespace
namespace vox
{
    /**
     * VoxRender's Base Exception Class
     *
     * All internal VoxRender exceptions derive from this base class.
     * The class contains essential information pertaining to the source 
     * from which the error originates.
     */ 
    class VOX_EXPORT Error : public virtual std::exception
    {
    public:
        /** 
         * Constructor - Initializes the exception content
         *
         * The exception message is intended primarily for logging purposes. Users are 
         * suggested to keep them as short as possible and to avoid expecting coherent or
         * parsable messages at the catch site. Derived classes will often append message
         * content in addition to the supplied message.
         *
         * @param _file     The file from which the error was thrown (use the __FILE__ macro)
         * @param _line     The line from which the error was thrown (use the __LINE__ macro)
         * @param _category The system associated with the exception (internally Vox)
         * @param _msg      The error message to be returned when what() is called.
         * @param _code     The error code associated with the Error
         */
        Error(const char* _file, int _line, const char* _category, 
              String const& _msg, int _code = Error_Unknown) : 
          file(_file), line(_line), message(_msg), code(_code), category(_category)
        {
        }

        /** Returns the message associated with the error */
        virtual char const* what() const throw () { return message.c_str(); }

        const char* category;   ///< Category name
        int         code;       ///< Error code
        String      message;    ///< Error message
        const char* file;       ///< Originating file
        int         line;       ///< Originating line
    };

}

// End definition
#endif // VOX_ERROR