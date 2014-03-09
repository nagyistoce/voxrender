/* ===========================================================================

	Project: Standard Volume Filters
    
	Description: Exposes some standard filters provided by the volt library

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
#ifndef SVF_VERSION_H
#define SVF_VERSION_H

// Stringify macro
#define SVF_XSTR(v) #v
#define SVF_STR(v) SVF_XSTR(v)

// Plugin version info
#define SVF_VERSION_MAJOR 1
#define SVF_VERSION_MINOR 0
#define SVF_VERSION_PATCH 0

// API support version info
#define SVF_API_VERSION_MIN_STR "0.0.0"
#define SVF_API_VERSION_MAX_STR "999.999.999"

// Plugin version string
#define SVF_VERSION_STRING SVF_STR(SVF_VERSION_MAJOR) \
	"." SVF_STR(SVF_VERSION_MINOR) "." SVF_STR(SVF_VERSION_PATCH)

// End definition
#endif // SVF_VERSION_H