/* ===========================================================================

	Project: VoxRender - Version Info

	Description: Defines VoxRender version info macros

    Copyright (C) 2012 Lucas Sherman

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
#ifndef VOX_VERSION_H
#define VOX_VERSION_H

// Stringify macro
#define VOX_XSTR(v) #v
#define VOX_STR(v) VOX_XSTR(v)

// VoxRender version
#define VOX_VERSION_POSTFIX " (dev)"
#define VOX_VERSION_MAJOR 1
#define VOX_VERSION_MINOR 0
#define VOX_VERSION_PATCH 0

// VoxRender version string
#define VOX_VERSION_STRING VOX_STR(VOX_VERSION_MAJOR) \
	"." VOX_STR(VOX_VERSION_MINOR) "."                \
    VOX_STR(VOX_VERSION_PATCH) VOX_VERSION_POSTFIX

// End definition
#endif // VOX_VERSION_H