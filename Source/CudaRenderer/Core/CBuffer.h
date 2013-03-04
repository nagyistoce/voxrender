/* ===========================================================================

	Project: VoxRender - CUDA device buffers

	Defines a class for managing buffers on devices using CUDA.

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
#ifndef CR_CBUFFER_H
#define CR_CBUFFER_H

// Common Library Header
#include "CudaRenderer/Core/Common.h"

// VoxLib Dependencies
#include "VoxLib/Core/Geometry/Color.h"
#include "VoxLib/Core/Geometry/Image.h"
#include "VoxLib/Core/Geometry/Vector.h"
#include "VoxLib/Error/CudaError.h"
#include "VoxLib/Rendering/FrameBuffer.h"

// API namespace
namespace vox 
{

/** CUDA global memory buffer object */
template<typename T> class CBuffer1D
{
public:    
    VOX_HOST_DEVICE CBuffer1D() { }

    /** Initializes and allocates the buffer */
    VOX_HOST CBuffer1D(size_t size) 
    {
        init(); resize(size);
    }

    /** Initializes the buffer parameters */
    VOX_HOST void init()
    {
        m_pData = nullptr;
        m_size  = 0;
    }

    /** Clears the buffer memory */
    VOX_HOST void clear() { VOX_CUDA_CHECK(cudaMemset(m_pData, 0, m_size)); }

    /** Resizes the internal memory buffer */
    VOX_HOST void resize(size_t size)
    {
        VOX_CUDA_CHECK(cudaFree(m_pData));

        m_size = size;

        VOX_CUDA_CHECK(cudaMalloc((void**)&m_pData, m_size*sizeof(T)));
    }

    /** Deallocates the memory buffer */
    VOX_HOST void reset()
    {
        if (m_pData)
        {
            VOX_CUDA_CHECK(cudaFree(m_pData));

            m_pData = nullptr;
            m_size  = 0;
        }
    }

    /** Writes data to the internal memory buffer */
    VOX_HOST void write(T const* buffer, cudaMemcpyKind kind = cudaMemcpyHostToDevice)
    {
        VOX_CUDA_CHECK(cudaMemcpy(m_pData, buffer, m_size*sizeof(T), kind));
    }

    /** Writes data to the internal buffer from a standard vector, resizing as necessary */
    VOX_HOST void write(std::vector<T> const& data)
    {
        if (m_size != data.size()) resize(m_size);

        write(&data[0], cudaMemcpyHostToDevice);
    }

    /** Reads data from the internal memory buffer */
    VOX_HOST void read(T * buffer, cudaMemcpyKind kind = cudaMemcpyDeviceToHost)
    {
        VOX_CUDA_CHECK(cudaMemcpy(buffer, m_pData, m_size*sizeof(T), kind));
    }

    /** Reads data from the internal memory buffer into a standard vector */
    VOX_HOST std::vector<T> read()
    {
        std::vector<T> data(m_size);

        read(&data[0], cudaMemcpyDeviceToHost);

        return data;
    }

    /** Returns the size in bytes of the buffer */
    VOX_HOST_DEVICE size_t size() const { return m_size; }
    
    /** Returns a reference to the element at the specified index */ 
    VOX_HOST_DEVICE T & at(size_t index) { return m_pData[index]; }

    /** Returns a const reference to the element at the specified index */ 
    VOX_HOST_DEVICE T const& at(size_t index) const { return m_pData[index]; }

    /** Array style access operator */
    VOX_HOST_DEVICE T & operator [](size_t index) { return m_pData[index]; }

    /** Array style access operator (const) */
    VOX_HOST_DEVICE T const& operator [](size_t index) const { return m_pData[index]; }

    /** Returns the raw pointer to the beginning of the buffer */
    VOX_HOST_DEVICE T* get() { return m_pData; }

    /** Returns the const raw pointer to the beginning of the buffer */
    VOX_HOST_DEVICE T const* get() const { return m_pData; }

protected:
    size_t m_size;
    T *    m_pData;
};

/** CUDA global memory buffer object */
template<typename T, bool PITCHED = true> class CBuffer2D
{
public:    
    VOX_HOST_DEVICE CBuffer2D() { }

    /** Initializes and allocates the buffer */
    VOX_HOST CBuffer2D(size_t x, size_t y) 
    {
        init(); resize(x, y);
    }

    /** Initializes the buffer parameters */
    VOX_HOST void init()
    {
        m_pData  = nullptr;
        m_pitch  = 0;
        m_height = 0;
        m_width  = 0;
    }
    
    /** Writes data to the internal memory buffer */
    // VOX_HOST void write(T const* buffer, size_t stride, cudaMemcpyKind kind = cudaMemcpyHostToDevice) { }

    /** Reads data from internal memory buffer */
    // VOX_HOST void read(T * buffer, size_t stride, cudaMemcpyKind kind = cudaMemcpyDeviceToHost) { }

    /** Performs a buffer transfer copy */
    VOX_HOST void copy(T* buffer, size_t stride, cudaMemcpyKind kind)
    {
            // :TODO: NOT CORRECT :BUG: //
        cudaMemcpy2D(buffer, stride, m_pData, m_pitch, 
                     m_width*sizeof(T), m_height, kind); 
    }

    /** Resizes the internal memory buffer */
    VOX_HOST void resize(size_t width, size_t height)
    {
        VOX_CUDA_CHECK(cudaFree(m_pData));

        m_width  = width;
        m_height = height;

        if (PITCHED) 
        {
            VOX_CUDA_CHECK(cudaMallocPitch((void**)&m_pData, &m_pitch, m_width*sizeof(T), m_height));
        }
        else         
        {
            m_pitch = m_width*sizeof(T);

            VOX_CUDA_CHECK(cudaMalloc((void**)&m_pData, m_pitch*m_height));
        }
    }

    /** Clears the buffer memory */
    VOX_HOST void clear() { VOX_CUDA_CHECK(cudaMemset(m_pData, 0, size())); }

    /** Deallocates the memory buffer */
    VOX_HOST void reset()
    {
        if (m_pData)
        {
            VOX_CUDA_CHECK(cudaFree(m_pData));

            m_pData  = nullptr;
            m_width  = 0;
            m_height = 0;
            m_pitch  = 0;
        }
    }

    /** Returns the number of elements in the CUDA buffer */
    VOX_HOST_DEVICE size_t numElements() const { return m_width*m_height; }

    /** Returns the size in bytes of the buffer */
    VOX_HOST_DEVICE size_t size() const { return m_height*m_pitch; }
    
    /** Returns a reference to the element at the specified index */ 
    VOX_HOST_DEVICE T & at(size_t x, size_t y)
    {
        if (PITCHED) return *(T*)((char*)(m_pData)+y*m_pitch+x*sizeof(T));
        else         return m_pData[y*m_width+x];
    }

    /** Returns a const reference to the element at the specified index */ 
    VOX_HOST_DEVICE T const& at(size_t x, size_t y) const
    {
        if (PITCHED) return *(T*)((char*)(m_pData)+y*m_pitch+x*sizeof(T));
        else         return m_pData[y*m_width+x];
    }

    /** Returns the raw pointer to the beginning of the buffer */
    VOX_HOST_DEVICE T* get() { return m_pData; }

    /** Returns the const raw pointer to the beginning of the buffer */
    VOX_HOST_DEVICE T const* get() const { return m_pData; }

    /** Returns the width of the internal memory buffer */
    VOX_HOST_DEVICE size_t width() const { return m_width; }

    /** Returns the height of the internal memory buffer */
    VOX_HOST_DEVICE size_t height() const { return m_height; }

    /** Returns the pitch of the internal memory buffer */
    VOX_HOST_DEVICE size_t pitch() const { return m_pitch; }

protected:
    size_t m_width;
    size_t m_height;
    size_t m_pitch;
    T *    m_pData;
};

// Specialization for device side frame buffers
template <typename T>
class CImgBuffer2D : public CBuffer2D<T>
{
public:
    VOX_HOST CImgBuffer2D() : CBuffer2D<T>() { }

    /** Initializes and allocates the image buffer */
    VOX_HOST CImgBuffer2D(size_t x, size_t y) : 
        CBuffer2D<T>(x, y) { }

    /** Reads the CBuffer into the specified image */
    VOX_HOST void read(Image<T> & img)
    {
        copy(img.data(), img.stride(), cudaMemcpyDeviceToHost);
    }

    /** Writes the specified image into the CBuffer */
    VOX_HOST void write(Image<T> const& img)
    {
        copy(const_cast<T*>(img.data()), img.stride(), cudaMemcpyHostToDevice);
    }
};

}

// End Definition
#endif // CR_CBUFFER_H