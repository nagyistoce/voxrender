/* ===========================================================================
                                                                           
   Project: Volume Transform Library                                       
                                                                           
   Description: Performs volume transform operations                       
                                                                           
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

// Include Header
#include "Conv.h"

// Include Dependencies
#include "VoxVolt/Impl/VolumeBlocker.h"
#include "VoxLib/Error/CudaError.h"

// CUDA runtime API header
#include <cuda_runtime_api.h>

#define KERNEL_BLOCK_W		16
#define KERNEL_BLOCK_H		16
#define KERNEL_BLOCK_SIZE   (KERNEL_BLOCK_W * KERNEL_BLOCK_H)

namespace vox {
namespace volt {

namespace {
namespace filescope {

    static float elapsedTime = 0.0f;

    // Textures for the input volume data sampling
    #define VOX_TEXTURE(T) texture<##T,3,cudaReadModeNormalizedFloat>  gd_volumeTexIn_##T 
        VOX_TEXTURE(Int8);
        VOX_TEXTURE(UInt8);
        VOX_TEXTURE(Int16);
        VOX_TEXTURE(UInt16);
    #undef VOX_TEXTURE

    surface<void, 3> gd_volumeTexOut; ///< Surface for convolved volume data

    // Performs 3D convolution on a volume data set
    template<typename T>
    __global__ void convKernel(Vector3u apron, cudaExtent blockSize)
    {
        for (int z = 0; z < blockSize.depth; z++)
        {
            surf3Dwrite(0, gd_volumeTexOut, threadIdx.x, threadIdx.y, z);
        }
    }

#define VOX_SETUP_TEX(T)                                                     \
    case Volume::Type_##T: {                                                 \
	    filescope::gd_volumeTexIn_##T.normalized     = false;                \
        filescope::gd_volumeTexIn_##T.filterMode     = cudaFilterModeLinear; \
        filescope::gd_volumeTexIn_##T.addressMode[0] = cudaAddressModeClamp; \
        filescope::gd_volumeTexIn_##T.addressMode[1] = cudaAddressModeClamp; \
        filescope::gd_volumeTexIn_##T.addressMode[2] = cudaAddressModeClamp; \
        VOX_CUDA_CHECK(cudaBindTextureToArray(filescope::gd_volumeTexIn_##T, \
            blocker.arrayIn(), blocker.formatIn()));                         \
        break; }

    // Binds the buffers in a volume blocker to
    void bindBuffers(VolumeBlocker & blocker)
    {
        auto volume = blocker.volume();

        // Bind the volume data buffers
        auto type = volume.type();
        switch (type)
        {
        VOX_SETUP_TEX(UInt8);
        VOX_SETUP_TEX(UInt16);
        VOX_SETUP_TEX(Int8);
        VOX_SETUP_TEX(Int16);

        default: throw Error(__FILE__, __LINE__, VOX_VOLT_LOG_CATEGORY, 
            format("Unsupported volume data type (%1%)", Volume::typeToString(type)), 
            Error_NotImplemented);
        }

        VOX_CUDA_CHECK(cudaBindSurfaceToArray(filescope::gd_volumeTexOut, blocker.arrayOut()));
    }

} // namespace filescope
} // namespace anonymous

// ----------------------------------------------------------------------------
//  Executes the convolution kernel on the input volume data
// ----------------------------------------------------------------------------
std::shared_ptr<Volume> Conv::execute(Volume & volume, Image3D<float> kernel, Volume::Type type)
{    
    // Create the start event for performance timing
    cudaEvent_t start, stop;
    VOX_CUDA_CHECK(cudaEventCreate(&start));
    VOX_CUDA_CHECK(cudaEventRecord(start,0));

    // Initialize the volume block loader/scheduler
    Vector3u apron = (kernel.dims() - Vector3u(1)) / 2;
    VolumeBlocker blocker(volume, apron, type);

	// Setup the execution configuration
    auto blockSize = blocker.blockSize();
    dim3 threads(KERNEL_BLOCK_W, KERNEL_BLOCK_H);
    dim3 blocks( 
        (blockSize.width  + threads.x - 1) / threads.x,
        (blockSize.height + threads.y - 1) / threads.y 
        );

    // Execute the blocked volume convolution 
    blocker.begin();
    
    filescope::bindBuffers(blocker); // Bind the data buffers

    while (!blocker.atEnd());
    {
        auto blockIndex = blocker.loadNext();

        // Execute the convolution kernel call
        filescope::convKernel<UInt8> <<<blocks,threads>>> (apron, blockSize);
        VOX_CUDA_CHECK(cudaDeviceSynchronize());

        blocker.readNext();
    }
    
    auto result = blocker.finish();
    
    //Create the stop event for performance timing
    VOX_CUDA_CHECK(cudaEventCreate(&stop));
    VOX_CUDA_CHECK(cudaEventRecord(stop,0));
    VOX_CUDA_CHECK(cudaEventSynchronize(stop));

    // Compute the time elapsed during GPU execution
    cudaEventElapsedTime(&filescope::elapsedTime, start, stop);

    return result;
}

// ----------------------------------------------------------------------------
//  Executes the convolution kernel on the input volume data
// ----------------------------------------------------------------------------
float Conv::getElapsedTime()
{
    return filescope::elapsedTime;
}

} // namespace volt
} // namespace vox