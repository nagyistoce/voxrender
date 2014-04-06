/* ===========================================================================

	Project: VoxServer

	Description: Implements a WebSocket based server for interactive rendering

    Copyright (C) 2014 Lucas Sherman

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
#ifndef INTERFACE_H
#define INTERFACE_H

// Include Header
#include "VoxServer/Common.h"
#include "VoxLib/Core/Types.h"

// Plugin interface 
extern "C"
{
    /** Returns the library version number */
    VOX_SERVER_EXPORTS char const* voxServerVersion();

    /** Must be called before any functions in the library (besides version) */
    VOX_SERVER_EXPORTS int voxServerStart(char const* directory, bool logToFile = true);

    /** Opens a websocket for streaming the specified scenefile */
    VOX_SERVER_EXPORTS int voxServerBeginStream(uint16_t * portOut, uint64_t * keyOut);

    /** Called to terminate the library and any active render streams */
    VOX_SERVER_EXPORTS void voxServerEnd();
}

// End definition
#endif // INTERFACE_H