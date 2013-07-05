/* ===========================================================================

	Project: VoxRender - Functors

	Defines some basic functors for simplifying container operations such as
    folding and mapping.

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
#ifndef VOX_FUNCTORS_H
#define VOX_FUNCTORS_H

// Common headers
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Types.h"
#include "VoxLib/Core/Preprocessor.h"

// Repetition limit for functors
#ifndef VOX_FUNCTOR_LIMIT
#define VOX_FUNCTOR_LIMIT 10
#endif // VOX_FUNCTOR_LIMIT

#undef min
#undef max

// API namespace
namespace vox
{
    /** Addition operator */
    template<typename T> inline
    VOX_HOST_DEVICE T sum(T const& x, T const& y) { return x+y; }
    
    /** Subtraction operator */
    template<typename T> inline
    VOX_HOST_DEVICE T sub(T const& x, T const& y) { return x-y; }
    
    /** Division operator */
    template<typename T> inline
    VOX_HOST_DEVICE T div(T const& x, T const& y) { return x/y; }
    
    /** Multiplication operator */
    template<typename T> inline
    VOX_HOST_DEVICE T mul(T const& x, T const& y) { return x*y; }
    
    /** Minimum operator */
    template<typename T> inline
    VOX_HOST_DEVICE T low(T const& x, T const& y) { return x < y ? x : y; }
    
    /** Maximum operator */
    template<typename T> inline 
    VOX_HOST_DEVICE T high(T const& x, T const& y) { return x > y ? x : y; }
    
    /** Is equal comparison operator */
    template<typename T> inline
    VOX_HOST_DEVICE bool equal(T const& x, T const& y) { return x == y; }

    /** Is not equal comparison operator */
    template<typename T> inline
    VOX_HOST_DEVICE bool notEqual(T const& x, T const& y) { return x != y; }

    /** Clamps the input value to the specified range */
    template<typename T> inline
    VOX_HOST_DEVICE T clamp(T const& value, T const& low, T const& high)
    {
        return (value<low) ? low : ( (value>high) ? high : value );
    }

    /** Creates a shared array of the specified size */
    VOX_EXPORT VOX_HOST std::shared_ptr<UInt8> makeSharedArray(size_t bytes);

    /** Functional array deleter method for shared_ptr */
    VOX_EXPORT VOX_HOST void arrayDeleter(void* data);
}

#undef VOX_FUNCTOR_LIMIT

// End definition
#endif // VOX_FUNCTORS_H