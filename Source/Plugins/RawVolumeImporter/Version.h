/* ===========================================================================

    Project: Raw Volume Import Module
    
	Description: Defines a VoxScene import module for .raw format volumes

    Copyright (C) 2013 Lucas Sherman

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
#ifndef RVI_VERSION_H
#define RVI_VERSION_H

// Stringify macro
#define RVI_XSTR(v) #v
#define RVI_STR(v) RVI_XSTR(v)

// Plugin version info
#define RVI_VERSION_MAJOR 1
#define RVI_VERSION_MINOR 0
#define RVI_VERSION_PATCH 0

// API support version info
#define RVI_API_VERSION_MIN_STR "0.0.0"
#define RVI_API_VERSION_MAX_STR "999.999.999"

// Plugin version string
#define RVI_VERSION_STRING RVI_STR(RVI_VERSION_MAJOR) \
	"." RVI_STR(RVI_VERSION_MINOR) "." RVI_STR(RVI_VERSION_PATCH)

// End definition
#endif // RVI_VERSION_H