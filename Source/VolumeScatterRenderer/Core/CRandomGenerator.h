/* ===========================================================================

	Project: CUDA Renderer - CUDA Random Number Generator

	Defines a class for device side random number generation.

	Lucas Sherman, email: LucasASherman@gmail.com

    MODIFIED FROM EXPOSURE RENDER'S "RNG.cuh" SOURCE FILE:

    Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
    All rights reserved.

    Redistribution and use in source and binary forms, with or without 
    modification, are permitted provided that the following conditions are 
    met:

      - Redistributions of source code must retain the above copyright 
        notice, this list of conditions and the following disclaimer.
      - Redistributions in binary form must reproduce the above copyright 
        notice, this list of conditions and the following disclaimer in the 
        documentation and/or other materials provided with the distribution.
      - Neither the name of the <ORGANIZATION> nor the names of its 
        contributors may be used to endorse or promote products derived from 
        this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED 
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
    OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
    OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
    EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
            return rand(); 
        }

        /** Returns a Vector2 of sample values */
        VOX_DEVICE Vector2f sample2D() 
        {
            return Vector2f(rand(), rand());
        }

        /** Returns a Vector3 of sample values */
        VOX_DEVICE Vector3f sample3D()
        {
            return Vector3f(rand(), rand(), rand());
        }

        /** Returns a cartesian coordinate disk sample */
        VOX_DEVICE Vector2f sampleDisk()
        {
            float r     = sqrtf(sample1D());
            float theta = 2.0f * (float)M_PI * sample1D();

            return Vector2f(r*cosf(theta), r*sinf(theta));
        }

        /** Returns a uniform unit hemisphere sample */
        VOX_DEVICE Vector3f sampleHemisphere()
        {
            float z = 1.f - 2.f * sample1D();
            float r = sqrtf(high(0.f, 1.f - z*z));
            float phi = 2.f * (float)M_PI * sample1D();
            float x = r * cosf(phi);
            float y = r * sinf(phi);

            return Vector3f(x, y, z);
        }
        
        /** Returns a uniform unit sphere sample */
        VOX_DEVICE Vector3f sampleSphere()
        {
            float z = sample1D();
            float r = sqrtf(high(0.f, 1.f - z*z));
            float phi = 2.f * (float)M_PI * sample1D();
            float x = r * cosf(phi);
            float y = r * sinf(phi);

            return Vector3f(x, y, z);
        }

    private:
        curandState m_localState;

        /** CUDA side rand() function */
	    VOX_DEVICE float rand()
        {
            return curand_uniform(&m_localState);
        }
	};
}

// End Definition
#endif // VSR_CRANDOM_GENERATOR_H