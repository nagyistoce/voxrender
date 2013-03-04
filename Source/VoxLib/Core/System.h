/* ===========================================================================

	Project: VoxRender - Systems header

	Description: Provides access to OS or system specific information

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
#ifndef VOX_SYSTEM_H
#define VOX_SYSTEM_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Types.h"

// API namespace
namespace vox
{
	/** 
	 * @brief System Class
     *
     * This class provides static member functions for accessing system or OS information.
	 */
	class VOX_EXPORT System
	{
    public:
        /** Returns the computer name if available, "" otherwise */
        VOX_HOST static String computerName();

        /** Returns the current directory */
        VOX_HOST static String currentDirectory();

        /** Returns the user name if available, "" otherwise */
        VOX_HOST static String userName();
        
        /** Returns the getLastError value for this system */
        VOX_HOST static size_t getLastError();
        
        /** Formats the system error value as a message string */
        VOX_HOST static String formatError(size_t error);
        
        /** Returns the number of processors on the current system */
        VOX_HOST static size_t getNumProcessors();

        /** Returns the current process ID */
        VOX_HOST static size_t getCurrentPid();
        
        /** Returns the current thread ID */
        VOX_HOST static size_t getCurrentTid();
	};
}

// End definition
#endif // VOX_SYSTEM_H