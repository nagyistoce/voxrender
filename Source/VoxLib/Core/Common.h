/* ===========================================================================

	Project: VoxRender - Common

	Includes common headers used by most classes. Also defines the export 
    symbol for linking.

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
#ifndef VOX_COMMON_H
#define VOX_COMMON_H

// Early detection of invalid NVCC compiler pass-through.
// This ensures that the error reported when attempting
// to compile this header with NVCC is actually useful.
#ifdef __CUDACC__
#error Attempting to compile incompatible headers with NVCC
#endif

// Cuda Commons Header
#include "VoxLib/Core/CudaCommon.h"

#ifndef Q_MOC_RUN

// Boost Standard Includes
#	include <boost/algorithm/string.hpp>
#	include <boost/any.hpp>
#	include <boost/date_time.hpp>
#	include <boost/filesystem.hpp>
#	include <boost/lexical_cast.hpp>
#	include <boost/property_tree/ptree.hpp>
#	include <boost/regex.hpp>
#	include <boost/thread.hpp>

// Boost IO Streams with Compression Filters
#	include <boost/iostreams/copy.hpp>
#	include <boost/iostreams/filtering_streambuf.hpp>
#	include <boost/iostreams/filter/bzip2.hpp>
#	include <boost/iostreams/filter/gzip.hpp>
#	include <boost/iostreams/filter/zlib.hpp>
#	include <boost/iostreams/positioning.hpp>
#	include <boost/iostreams/stream.hpp>
#	include <boost/iostreams/stream_buffer.hpp>

// Boost condition_variable alias
namespace boost 
{ 
    typedef condition_variable cond_var; 
}

#endif // Q_MOC_RUN

// End definition
#endif // VOX_COMMON_H