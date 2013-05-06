/* ===========================================================================

	Project: VoxRender - CUDA random value buffer

	Defines a class for managing buffers of rand content on devices.

	Lucas Sherman, email: LucasASherman@gmail.com

    MODIFIED FROM EXPOSURE RENDER'S "Buffer.cuh" SOURCE FILE:

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

// Begin Definition
#ifndef VSR_CRANDOM_BUFFER_H
#define VSR_CRANDOM_BUFFER_H

// Common Library Header
#include "VolumeScatterRenderer/Core/Common.h"

// VoxLib Dependencies
#include "VoxLib/Core/Geometry/Vector.h"
#include "VoxLib/Error/CudaError.h"

// Device representations of scene components
#include "VolumeScatterRenderer/Core/CBuffer.h"

// API namespace
namespace vox 
{

/** CBuffer which automatically generates random content */
class CRandomBuffer2D : public CBuffer2D<unsigned int, false>
{
public:
    VOX_HOST_DEVICE CRandomBuffer2D() {}

    /** Initializes the buffer on construction */
    VOX_HOST CRandomBuffer2D(size_t width, size_t height) :
        CBuffer2D(width, height)
    { 
        randomize();
    }

    /** Resets the buffer with random content */
    VOX_HOST void randomize()
    {
        size_t bufSize = size();

        std::unique_ptr<unsigned int> seeds(new unsigned int[bufSize]);

        for (size_t i = 0; i < bufSize; i++) seeds.get()[i] = rand(); // :TODO: replace with thread safe rand alternative

        VOX_CUDA_CHECK(cudaMemcpy(m_pData, seeds.get(), bufSize, cudaMemcpyHostToDevice));
    }
};

}

// End Definition
#endif // VSR_CRANDOM_BUFFER_H