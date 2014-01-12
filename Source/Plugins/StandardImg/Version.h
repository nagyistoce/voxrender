/* ===========================================================================

    Project: Standard Image ExIm Module
    
	Description: Defines an image import module for common LDR image formats

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
#ifndef VOX_SIMG_VERSION_H
#define VOX_SIMG_VERSION_H

// Stringify macro
#define VOX_SIMG_XSTR(v) #v
#define VOX_SIMG_STR(v) VOX_SIMG_XSTR(v)

// Plugin version info
#define VOX_SIMG_VERSION_MAJOR 1
#define VOX_SIMG_VERSION_MINOR 0
#define VOX_SIMG_VERSION_PATCH 0

// API support version info
#define VOX_SIMG_API_VERSION_MIN_STR "0.0.0"
#define VOX_SIMG_API_VERSION_MAX_STR "999.999.999"

// Plugin version string
#define VOX_SIMG_VERSION_STRING VOX_SIMG_STR(VOX_SIMG_VERSION_MAJOR) \
	"." VOX_SIMG_STR(VOX_SIMG_VERSION_MINOR) "." VOX_SIMG_STR(VOX_SIMG_VERSION_PATCH)

// End definition
#endif // VOX_SIMG_VERSION_H