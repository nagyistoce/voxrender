/* ===========================================================================

	Project: Volume Scatter Renderer

	Description: Encapsulates various render parameters for a scene

    Copyright (C) 2012-2014 Lucas Sherman

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
#ifndef VSR_CRANDOM_GENERATOR_H
#define VSR_CRANDOM_GENERATOR_H

// Common Library Header
#include "VolumeScatterRenderer/Core/Common.h"

// VoxLib Dependencies
#include "VoxLib/Core/Geometry/Vector.h"
#include "VoxLib/Core/Functors.h"

// Standard Library Includes
#define _USE_MATH_DEFINES
#include <math.h>

#include <curand_kernel.h>

// API namespace
namespace vox
{
	/** CUDA Random Number Generator */
	class CRandomGenerator
	{
    public:
        /** Initializes the CRNG state */
        VOX_DEVICE CRandomGenerator(curandState state) :
            m_localState(state)
        {
        }

        /** Returns the CRNG state */
        VOX_DEVICE curandState state()
        {
            return m_localState;
        }

        /** Returns a single sample value */
        VOX_DEVICE float sample1D() 
        { 
            return curand_uniform(&m_localState);
        }

        /** Returns a Vector2 of sample values */
        VOX_DEVICE Vector2f sample2D() 
        {
            return Vector2f(sample1D(), sample1D());
        }

        /** Returns a Vector3 of sample values */
        VOX_DEVICE Vector3f sample3D()
        {
            return Vector3f(sample1D(), sample1D(), sample1D());
        }

        /** Returns a cartesian coordinate disk sample */
        VOX_DEVICE Vector2f sampleDisk()
        {
            float r = sqrtf(sample1D());
            float t = 2.0f * (float)M_PI * sample1D();

            return Vector2f(r*cosf(t), r*sinf(t));
        }

        /** Returns a uniform unit sphere sample */
        VOX_DEVICE Vector3f sampleSphere()
        {
            float z = 1.f - 2.f * sample1D();
            float r = sqrtf(high(0.f, 1.f - z*z));
            float phi = 2.f * (float)M_PI * sample1D();
            float x = r * sinf(phi);
            float y = r * cosf(phi);

            return Vector3f(x, y, z);
        }

    private:
        curandState m_localState;
	};
}

// End Definition
#endif // VSR_CRANDOM_GENERATOR_H