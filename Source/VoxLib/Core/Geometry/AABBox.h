/* ===========================================================================

	Project: VoxRender - Axis Aligned Bounding Box

	Defines a class for managing axis aligned bounding boxes.

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
#ifndef VOX_AABBOX_H
#define VOX_AABBOX_H

// VoxRender Includes
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Functors.h"
#include "VoxLib/Core/Geometry/Vector.h"

// API namespace
namespace vox
{
	/** Axis Aligned Bounding Box */
    template< typename T, int N > struct AABBox
    {
        /* Returns the volume of the bounding box */
        VOX_HOST_DEVICE T volume( ) const { return (max-min).template fold<T>(1,vox::mul<T>); }

        Vector<T,N> min;    ///< Minimum extent of the bounding box
        Vector<T,N> max;    ///< Minimum extent of the bounding box
	};

	/** AABBox stream operator */
	template< typename T, int N >
    VOX_HOST std::ostream &operator<<( std::ostream &os, const AABBox<T,N> &bbox ) 
    {
        os << "[" << bbox.min << "," << bbox.max << "]";
        return os;
    }
}

// End Definition
#endif // VOX_AABBOX_H