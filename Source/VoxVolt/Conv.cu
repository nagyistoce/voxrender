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

#define MAX_KERNEL_SIZE 5

namespace vox {
namespace volt {

namespace {
namespace filescope {

    __constant__ float gd_kernel[MAX_KERNEL_SIZE*MAX_KERNEL_SIZE*MAX_KERNEL_SIZE];

    static float elapsedTime = 0.0f;

    // Textures for the input volume data sampling
    #define VOX_TEXTURE(T) texture<##T,3,cudaReadModeNormalizedFloat>  gd_volumeTexIn_##T 
        VOX_TEXTURE(Int8);
        VOX_TEXTURE(UInt8);
        VOX_TEXTURE(Int16);
        VOX_TEXTURE(UInt16);
    #undef VOX_TEXTURE

    surface<void, 3> gd_volumeTexOut; ///< Surface for convolved volume data output

    // Performs 3D convolution on a volume data set
    template<typename T>
    __global__ void convKernel(Vector3u apron, cudaExtent blockSize, Vector2f range)
    {
	    int x = blockIdx.x * blockDim.x + threadIdx.x;
	    int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= blockSize.width || y >= blockSize.height) return;

        for (int z = 0; z < blockSize.depth; z++)
        {
            float sum = 0.0f;
            float * filter = gd_kernel;
            // :TODO: Use shared memory
	        // Compute point convolution
	        int mbegin = x-apron[0]; if (mbegin < 0)               mbegin = 0;
	        int mend   = x+apron[0]; if (mend >= blockSize.width)  mend = blockSize.width-1;
	        int nbegin = y-apron[1]; if (nbegin < 0)               nbegin = 0;
	        int nend   = y+apron[1]; if (nend >= blockSize.height) nend = blockSize.height-1;
	        int obegin = z-apron[2]; if (obegin < 0)               obegin = 0;
	        int oend   = z+apron[2]; if (oend >= blockSize.depth)  oend = blockSize.depth-1;
	        for (int o = obegin; o <= oend; o++)
	        for (int n = nbegin; n <= nend; n++) 
	        for (int m = mbegin; m <= mend; m++)
	        {//:TODO: Implement this correctly
		        sum += *filter * tex3D(gd_volumeTexIn_UInt8, m, n, o); filter++;
	        }

            surf3Dwrite<T>((T)(range[0] + range[1]*sum), gd_volumeTexOut, x, y, z);
        }
    }

#define VOX_SETUP_TEX(T)                                                     \
    case Volume::Type_##T: {                                                 \
	    filescope::gd_volumeTexIn_##T.normalized     = false;                \
        filescope::gd_volumeTexIn_##T.filterMode     = cudaFilterModePoint; \
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

        VOX_CUDA_CHECK(cudaBindSurfaceToArray(gd_volumeTexOut, blocker.arrayOut()));
    }

} // namespace filescope
} // namespace anonymous

// ----------------------------------------------------------------------------
//  Executes the convolution kernel on the input volume data
// ----------------------------------------------------------------------------
std::shared_ptr<Volume> Conv::execute(Volume & volume, Image3D<float> kernel, Volume::Type type)
{   
    // Check the kernel dimensions
    if (kernel.width()  > MAX_KERNEL_SIZE ||
        kernel.height() > MAX_KERNEL_SIZE ||
        kernel.depth()  > MAX_KERNEL_SIZE)
        throw Error(__FILE__, __LINE__, VOX_VOLT_LOG_CATEGORY, "Kernel size exceeds library limit", Error_Range);
    Vector3u apron = (kernel.dims() - Vector3u(1)) / 2;

    // Copy the kernel into device memory
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(filescope::gd_kernel, 
        kernel.data(), sizeof(float)*kernel.dims().fold(mul)));

    // Create the start event for performance timing
    cudaEvent_t start, stop;
    VOX_CUDA_CHECK(cudaEventCreate(&start));
    VOX_CUDA_CHECK(cudaEventRecord(start,0));

    // Initialize the volume block loader/scheduler
    auto outType = (type == Volume::Type_End) ? volume.type() : type;
    VolumeBlocker blocker(volume, apron, outType);

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
        switch (volume.type())
        {
        case Volume::Type_UInt8: filescope::convKernel<UInt8> <<<blocks,threads>>> (apron, blockSize, Vector2f(0.0f, std::numeric_limits<UInt8>::max())); break;
        case Volume::Type_UInt16: filescope::convKernel<UInt16> <<<blocks,threads>>> (apron, blockSize, Vector2f(0.0f, std::numeric_limits<UInt16>::max())); break;
        default: throw Error(__FILE__, __LINE__, VOX_VOLT_LOG_CATEGORY, "Unsupported volume data type", Error_NotImplemented);
        }
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