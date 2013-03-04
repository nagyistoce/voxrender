/* ===========================================================================

	Project: CUDA Renderer - CUDA Renderer for VoxRender

	Includes the VoxRender common header as well as defining some additional 
    macros specific to the CUDA Renderer library.

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
#ifndef CR_COMMON_H
#define CR_COMMON_H

// Export configurations
#if defined(CUDARenderer_SHARED) 
#   ifdef CUDARenderer_EXPORTS
#       define CR_EXPORT __declspec(dllexport)
#   else
#       define CR_EXPORT __declspec(dllimport)
#   endif
#else
#	define CR_EXPORT
#endif

// VoxRender log category
static char const* CR_LOG_CATEGORY = "CRR";

// Version info
#include "CudaRenderer/Core/Version.h"

// VoxRender Common Headers
#include "VoxLib/Core/CudaCommon.h"

// End definition
#endif // CR_COMMON_H