/* ===========================================================================

    Project: PVM Volume Import Module
    
	Description: Defines a VoxScene import module for .pvm format volumes

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
#ifndef PVMI_VERSION_H
#define PVMI_VERSION_H

// Stringify macro
#define PVMI_XSTR(v) #v
#define PVMI_STR(v) PVMI_XSTR(v)

// Plugin version info
#define PVMI_VERSION_MAJOR 1
#define PVMI_VERSION_MINOR 0
#define PVMI_VERSION_PATCH 0

// API support version info
#define PVMI_API_VERSION_MIN_STR "0.0.0"
#define PVMI_API_VERSION_MAX_STR "999.999.999"

// Plugin version string
#define PVMI_VERSION_STRING PVMI_STR(PVMI_VERSION_MAJOR) \
	"." PVMI_STR(PVMI_VERSION_MINOR) "." PVMI_STR(PVMI_VERSION_PATCH)

// End definition
#endif // PVMI_VERSION_H