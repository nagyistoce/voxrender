/* ===========================================================================

	Project: VoxRender - Profiling

	Description: Implements a class for performing light weight profiling.

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
#ifndef VOX_PROFILER_H
#define VOX_PROFILER_H

// Include Dependencies
#include "Core/Common.h"
//#include "Core/HighResTimer.h"

// API namespace
namespace vox
{

    class Profiler
    {
    public:
        static void ExecCount(const char* key);

        // static Profile resetAndProfile(const char* key);
        // static Profile profile(const char* key);
    }
}

// End definition
#endif