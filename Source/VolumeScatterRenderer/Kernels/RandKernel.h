/* ===========================================================================

	Project: GPU based Volume Scatter Renderer
    
	Description: Performs initialization of the CUDA RNG states

    Copyright (C) 2014 Lucas Sherman

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
#ifndef VSR_RANDKERNEL_H
#define VSR_RANDKERNEL_H

// Common Library Header
#include "VolumeScatterRenderer/Core/Common.h"

#include <curand_kernel.h>

// API namespace
namespace vox 
{

/** Defines the interface for initializing random generator states */
class RandKernel
{
public:
    /** Executes the tonemapping kernel on the device */
    static curandState * create(size_t width, size_t height);

    /** Destroys a collection of CRNG states */
    static void destroy(curandState * states);

    /** Returns the time for the last kernel execution */
    static float getTime() { return m_elapsedTime; }

private:
    static float m_elapsedTime; ///< Kernel execution time
};

} // namespace vox

// End Definition
#endif // VSR_RANDKERNEL_H