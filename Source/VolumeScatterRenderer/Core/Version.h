/* ===========================================================================

	Project: Volume Scatter Renderer

	Description: Defines project version macros

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
#ifndef VSR_VERSION_H
#define VSR_VERSION_H

// Stringify macro
#define VSR_XSTR(v) #v
#define VSR_STR(v) VSR_XSTR(v)

// VoxRender version
#define VSR_VERSION_POSTFIX " (dev)"
#define VSR_VERSION_MAJOR 1
#define VSR_VERSION_MINOR 0

// VoxRender version string
#define VSR_VERSION_STRING VSR_STR(VSR_VERSION_MAJOR) \
	"." VSR_STR(CR_VERSION_MINOR) VSR_VERSION_POSTFIX

// End definition
#endif // VSR_VERSION_H