/* ===========================================================================

    Project: VoxServer
    
	Description: Rendering library for VoxRenderWeb

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

// Plugin interface 
extern "C"
{
    /** Returns the library version number */
    __declspec(dllexport) char const* voxServerVersion();

    /** Must be called before any functions in the library (besides version) */
    __declspec(dllexport) int voxServerStart(char const* directory);

    /** Opens a websocket for streaming the specified scenefile */
    __declspec(dllexport) int voxServerStream(char const* filename);

    /** Called to terminate the library and any active render streams */
    __declspec(dllexport) void voxServerEnd();
}

// End definition
#endif // INTERFACE_H