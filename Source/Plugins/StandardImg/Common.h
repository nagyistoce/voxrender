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
#ifndef VOX_SIMG_COMMON_H
#define VOX_SIMG_COMMON_H

// VoxRender log category
static char const* VOX_SIMG_LOG_CATEGORY = "SIMG";

// Export macro
#ifdef StandardImg_EXPORTS
#   define VOX_SIMG_EXPORT __declspec(dllexport)
#else
#   define VOX_SIMG_EXPORT __declspec(dllimport)
#endif

// Version info
#include "Version.h"

// End definition
#endif // VOX_SIMG_COMMON_H