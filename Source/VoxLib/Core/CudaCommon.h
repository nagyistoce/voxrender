/* ===========================================================================

	Project: VoxRender - CudaCommon

	Includes common headers used by most classes. This header is used by NVCC
    compiled source files and only includes headers which NVCC can properly 
    handle.

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
#ifndef VOX_CUDA_COMMON_H
#define VOX_CUDA_COMMON_H

// Export configurations
#if defined(VoxLib_SHARED) 
#   ifdef VoxLib_EXPORTS
#       define VOX_EXPORT __declspec(dllexport)
#   else
#       define VOX_EXPORT __declspec(dllimport)
#   endif
#else
#	define VOX_EXPORT
#endif

// VoxRender log category
static char const* VOX_LOG_CATEGORY = "Vox";

// Version info
#include "VoxLib/Core/Version.h"

// CUDA runtime API header
#include <cuda_runtime_api.h>

// CUDA intellisense detection
#ifdef __INTELLISENSE__
#define __CUDACC__
#define __THROW
#endif // __INTELLISENSE__

// CUDA host/device code
#define VOX_HOST        __host__
#define VOX_DEVICE      __device__
#define VOX_HOST_DEVICE __host__ __device__

// Platform independent deprecated code macro
#ifdef __GNUC__
#   define VOX_DEPRECATED __attribute__ ((deprecated))
#elif defined(_MSC_VER)
#   define VOX_DEPRECATED _declspec(deprecated)
#else
#   define VOX_DEPRECATED
#   pragma message("warning: 'deprecated' tag not defined for this compiler, no warnings will be issued")
#endif

// Platform independent compiler warning macro
#ifdef _MSC_VER
#   define VOX_LOC __FILE__ "("VOX_STR(__LINE__)") : Warning Msg: "
#   define VOX_WARN(msg) _Pragma(message(VOX_LOC msg))
#else
#   define VOXWARN(msg)
#endif

// VS2012 limitations
#define _VARIADIC_MAX 10

// Not really a great idea to include a bunch of headers here
// but they are used so frequently that I deemed it worthwhile

// <math.h> Include Macro
#define _USE_MATH_DEFINES

// STD Standard Includes
#include <algorithm>
#include <deque>
#include <exception>
#include <functional>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <math.h>
#include <memory>
#include <sstream>
#include <stack>
#include <string>
#include <vector>

// Boost Standard Includes
#ifndef Q_MOC_RUN
#	include <boost/algorithm/string.hpp>
#	include <boost/foreach.hpp>
#	include <boost/format.hpp>
#	include <boost/utility.hpp>
#endif // Q_MOC_RUN

// End definition
#endif // VOX_CUDA_COMMON_H