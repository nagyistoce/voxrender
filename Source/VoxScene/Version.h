/* ===========================================================================

    Project: VoxScene
    
	Description: Defines scene graph elements for rendering

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
#ifndef VOXS_VERSION_H
#define VOXS_VERSION_H

// Stringify macro
#define VOXS_XSTR(v) #v
#define VOXS_STR(v) VOXS_XSTR(v)

// Plugin version info
#define VOXS_VERSION_MAJOR 1
#define VOXS_VERSION_MINOR 0
#define VOXS_VERSION_PATCH 0

// API support version info
#define VOXS_API_VERSION_MIN_STR "0.0.0"
#define VOXS_API_VERSION_MAX_STR "999.999.999"

// Plugin version string
#define VOXS_VERSION_STRING VOXS_STR(VOXS_VERSION_MAJOR) \
	"." VOXS_STR(VOXS_VERSION_MINOR) "." VOXS_STR(VOXS_VERSION_PATCH)

// End definition
#endif // VOXS_VERSION_H