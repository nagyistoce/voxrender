/* ===========================================================================

	Project: CUDA Renderer - Version Info

	Description: Defines CUDA Renderer version info macros

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
#ifndef CR_VERSION_H
#define CR_VERSION_H

// Stringify macro
#define CR_XSTR(v) #v
#define CR_STR(v) CR_XSTR(v)

// VoxRender version
#define CR_VERSION_POSTFIX " (dev)"
#define CR_VERSION_MAJOR 1
#define CR_VERSION_MINOR 0

// VoxRender version string
#define CR_VERSION_STRING CR_STR(CR_VERSION_MAJOR) \
	"." CR_STR(CR_VERSION_MINOR) CR_VERSION_POSTFIX

// End definition
#endif // CR_VERSION_H