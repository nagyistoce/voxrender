/* ===========================================================================

	Project: Volume Scatter Renderer
    
	Description: Core Includes for internal usage

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
#ifndef CR_COMMON_H
#define CR_COMMON_H

// Export configurations
#ifdef VolumeScatterRenderer_EXPORTS
#   define VSR_EXPORT __declspec(dllexport)
#else
#   define VSR_EXPORT __declspec(dllimport)
#endif

// VoxRender log category
static char const* VSR_LOG_CATEGORY = "VSR";

// Version info
#include "VolumeScatterRenderer/Core/Version.h"

// VoxRender Common Headers
#include "VoxLib/Core/CudaCommon.h"

// End definition
#endif // CR_COMMON_H