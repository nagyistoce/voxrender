/* ===========================================================================

    Project: StandardIO - Module definition for exported interface

    Description: A libcurl wrapper compatible with the VoxIO library

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
#ifndef SIO_VERSION_H
#define SIO_VERSION_H

// Stringify macro
#define SIO_XSTR(v) #v
#define SIO_STR(v) SIO_XSTR(v)

// Plugin version info
#define SIO_VERSION_MAJOR 1
#define SIO_VERSION_MINOR 0
#define SIO_VERSION_PATCH 0

// API support version info
#define SIO_API_VERSION_MIN_STR "0.0.0"
#define SIO_API_VERSION_MAX_STR "999.999.999"

// Plugin version string
#define SIO_VERSION_STRING SIO_STR(SIO_VERSION_MAJOR) \
	"." SIO_STR(SIO_VERSION_MINOR) "." SIO_STR(SIO_VERSION_PATCH)

// End definition
#endif // SIO_VERSION_H