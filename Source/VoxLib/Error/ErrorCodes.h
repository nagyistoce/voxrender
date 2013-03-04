/* ===========================================================================

	Project: VoxRender - Error Codes

	Description: Defines error codes used by the VoxRender library

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
#ifndef VOX_ERROR_CODES_H
#define VOX_ERROR_CODES_H

// API namespace
namespace vox
{

/** VoxRender Error codes */
enum ErrorCode 
{
	Error_None           = 0,		///< Operation successfull
    
	Error_System         = 1,		///< System : unknown system error 
	Error_NoMemory       = 2,		///< System : out of memory
	Error_Device         = 3,		///< System : device error
	Error_NoDeviceMemory = 4,		///< System : device out of memory

	Error_NoPermission   = 11,		///< File : permissions
    Error_BadFormat      = 12,      ///< File : format error
	Error_BadVersion     = 13,		///< File : version mismatch

	Error_NotImplemented = 21,		///< Misc : unimplemented feature 
	Error_ProgramLimit   = 22,		///< Misc : arbitrary program limitation
	Error_Bug            = 23,		///< Misc : probably a bug
	Error_Math           = 24,		///< Misc : zero-divide etc

	Error_BadToken       = 41,		///< IO : invalid token for request 
	Error_Range          = 42,		///< IO : parameter out of range 
	Error_Consistency    = 43,		///< IO : parameters inconsistent 
	Error_MissingData    = 44,		///< IO : required params not provided
	Error_Syntax         = 45,		///< IO : syntax error
    Error_BadStream      = 46,      ///< IO : a stream operation has failed
    Error_NotAllowed     = 47,      ///< IO : operation not allowed
    Error_NotFound       = 48,      ///< IO : requested resource not found

	Error_Unknown   	 = 101		///< Unknown error occured
};

}

// End definition
#endif // VOX_ERROR_CODES_H