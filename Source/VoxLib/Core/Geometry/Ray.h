/* ===========================================================================

	Project: VoxRender - Ray

	Defines a class for handling traditional mathematical ray operations.

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
#ifndef VOX_RAY_H
#define VOX_RAY_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Geometry/Vector.h"
#include "VoxLib/Core/Types.h"

// API namespace
namespace vox
{
	/** Ray management structure */
    template<typename T, size_t N> 
	struct Ray
	{
        VOX_HOST_DEVICE Ray() { } 

        VOX_HOST_DEVICE Ray(
            Vector<T,N> const& origin,
            Vector<T,N> const& direction
            ) :
            pos(origin),
            dir(direction),
            min(0),
            max(0)
        {
        }

		Vector<T,N> pos; ///< Ray beg point
        Vector<T,N> dir; ///< Ray direction
        T min;
        T max;
	};

	// Common ray types
	typedef Ray<int,2> Ray2;
	typedef Ray<int,3> Ray3;
	typedef Ray<int,4> Ray4;
	typedef Ray<unsigned int,2> Ray2u;
	typedef Ray<unsigned int,3> Ray3u;
	typedef Ray<unsigned int,4> Ray4u;
	typedef Ray<float,2> Ray2f;
	typedef Ray<float,3> Ray3f;
	typedef Ray<float,4> Ray4f;
	typedef Ray<double,2> Ray2d;
	typedef Ray<double,3> Ray3d;
	typedef Ray<double,4> Ray4d;
}

// End Definition
#endif // VOX_RAY_H