/* ===========================================================================

    Project: Vox Scene Importer - Module definition for scene importer

    Description: A vox scene file importer module

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
#ifndef VSI_VERSION_H
#define VSI_VERSION_H

// Stringify macro
#define VSI_XSTR(v) #v
#define VSI_STR(v) VSI_XSTR(v)

// Plugin version info
#define VSI_VERSION_MAJOR 1
#define VSI_VERSION_MINOR 0
#define VSI_VERSION_PATCH 0

// API support version info
#define VSI_API_VERSION_MIN_STR "0.0.0"
#define VSI_API_VERSION_MAX_STR "999.999.999"

// Plugin version string
#define VSI_VERSION_STRING VSI_STR(VSI_VERSION_MAJOR) \
	"." VSI_STR(VSI_VERSION_MINOR) "." VSI_STR(VSI_VERSION_PATCH)

// End definition
#endif // VSI_VERSION_H