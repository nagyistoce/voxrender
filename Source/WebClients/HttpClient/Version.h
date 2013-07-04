/* ===========================================================================

    Project: HttpClient
    
	Description: Provides an http client library for VoxLib

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
#ifndef VOX_HTTP_VERSION_H
#define VOX_HTTP_VERSION_H

// Stringify macro
#define VOX_HTTP_XSTR(v) #v
#define VOX_HTTP_STR(v) VOX_HTTP_XSTR(v)

// Plugin version info
#define VOX_HTTP_VERSION_MAJOR 1
#define VOX_HTTP_VERSION_MINOR 0
#define VOX_HTTP_VERSION_PATCH 0

// API support version info
#define VOX_HTTP_API_VERSION_MIN_STR "0.0.0"
#define VOX_HTTP_API_VERSION_MAX_STR "999.999.999"

// Plugin version string
#define VOX_HTTP_VERSION_STRING VOX_HTTP_STR(VOX_HTTP_VERSION_MAJOR) \
	"." VOX_HTTP_STR(VOX_HTTP_VERSION_MINOR) "." VOX_HTTP_STR(VOX_HTTP_VERSION_PATCH)

// End definition
#endif // VOX_HTTP_VERSION_H