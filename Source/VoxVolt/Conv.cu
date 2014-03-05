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
#include "VoxLib/Core/Logging.h"

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

    // Textures for the input volume data sampling
    #define VOX_TEXTURE(T) texture<##T,3,cudaReadModeNormalizedFloat>  gd_volumeTexIn_##T 
        VOX_TEXTURE(Int8);
        VOX_TEXTURE(UInt8);
        VOX_TEXTURE(Int16);
        VOX_TEXTURE(UInt16);
    #undef VOX_TEXTURE

    surface<void, 3> gd_volumeTexOut; ///< Surface for convolved volume data output

    template<typename T> VOX_DEVICE float fetchSample(int x, int y, int z) { return 0.f; }
    #define TEMPLATE(T) template<> VOX_DEVICE float fetchSample<##T>(int x, int y, int z) { return tex3D(gd_volumeTexIn_##T, x, y, z); }
    TEMPLATE(Int8)
    TEMPLATE(UInt8)
    TEMPLATE(UInt16)
    TEMPLATE(Int16)

    // Performs 3D convolution on a volume data set
    template<typename T>
    __global__ void convKernel(Vector3u apron, cudaExtent blockSize, Vector2f range)
    {
        extern __shared__ float cache[];

	    int x = blockIdx.x * blockDim.x + threadIdx.x;
	    int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= blockSize.width || y >= blockSize.height) return;

        for (int z = 0; z < blockSize.depth; z++)
        {
            float sum = 0.0f;
            float * filter = gd_kernel;

	        // Compute point convolution
	        int mbegin = x-apron[0]; int mend = x+apron[0];
	        int nbegin = y-apron[1]; int nend = y+apron[1];
	        int obegin = z-apron[2]; int oend = z+apron[2];
	        for (int o = obegin; o <= oend; o++)
	        for (int n = nbegin; n <= nend; n++) 
	        for (int m = mbegin; m <= mend; m++)
	        {
		        sum += *filter * fetchSample<T>(m, n, o); filter++;
	        }

            surf3Dwrite<T>((T)(range[0] + range[1]*sum), gd_volumeTexOut, x*sizeof(T), y, z);
        }
    }

    template<typename T>
    __global__ void convSepKernel()
    {
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
        auto type = volume->type();
        switch (type)
        {
        VOX_SETUP_TEX(UInt8);
        VOX_SETUP_TEX(UInt16);
        VOX_SETUP_TEX(Int8);
        VOX_SETUP_TEX(Int16);

        default: throw Error(__FILE__, __LINE__, VOLT_LOG_CAT, 
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
std::shared_ptr<Volume> Conv::execute(std::shared_ptr<Volume> volume, Image3D<float> kernel, Volume::Type type)
{   
    // Verify the kernel is of odd dimensions
    if (!(kernel.width() % 2 && kernel.height() % 2 && kernel.depth() % 2)) 
        throw Error(__FILE__, __LINE__, VOLT_LOG_CAT, 
                    "Kernel size must be odd", Error_Range);

    // Check the kernel dimensions against the library limit
    if (kernel.width()  > MAX_KERNEL_SIZE ||
        kernel.height() > MAX_KERNEL_SIZE ||
        kernel.depth()  > MAX_KERNEL_SIZE)
        throw Error(__FILE__, __LINE__, VOLT_LOG_CAT, "Kernel size exceeds library limit", Error_Range);
    Vector3u apron = (kernel.dims() - Vector3u(1)) / 2;

    // Copy the kernel into device memory
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(filescope::gd_kernel, 
        kernel.data(), sizeof(float)*kernel.dims().fold(mul)));

    // Create the start event for performance timing
    cudaEvent_t start, stop;
    VOX_CUDA_CHECK(cudaEventCreate(&start));
    VOX_CUDA_CHECK(cudaEventRecord(start,0));

    // Initialize the volume block loader/scheduler
    auto outType = (type == Volume::Type_End) ? volume->type() : type;
    VolumeBlocker blocker(volume, apron, outType);

	// Setup the execution configuration
    auto blockSize = blocker.blockSize();
    dim3 threads(KERNEL_BLOCK_W, KERNEL_BLOCK_H);
    dim3 blocks( 
        (blockSize.width  + threads.x - 1) / threads.x,
        (blockSize.height + threads.y - 1) / threads.y 
        );
    unsigned int shared = 1024;

    // Execute the blocked volume convolution 
    blocker.begin();
    
    filescope::bindBuffers(blocker); // Bind the data buffers

    while (!blocker.atEnd());
    {
        auto blockIndex = blocker.loadNext();

        // Execute the convolution kernel call
        switch (volume->type())
        {
        case Volume::Type_Int8:   filescope::convKernel<Int8>   <<<blocks,threads,shared>>> (apron, blockSize, 
            Vector2f(std::numeric_limits<Int8>::min(), std::numeric_limits<Int8>::max()));   break;
        case Volume::Type_UInt8:  filescope::convKernel<UInt8>  <<<blocks,threads,shared>>> (apron, blockSize, 
            Vector2f(std::numeric_limits<UInt8>::min(), std::numeric_limits<UInt8>::max()));  break;
        case Volume::Type_UInt16: filescope::convKernel<UInt16> <<<blocks,threads,shared>>> (apron, blockSize, 
            Vector2f(std::numeric_limits<UInt16>::min(), std::numeric_limits<UInt16>::max())); break;
        case Volume::Type_Int16:  filescope::convKernel<Int16>  <<<blocks,threads,shared>>> (apron, blockSize, 
            Vector2f(0, std::numeric_limits<Int16>::max()));  break;
        default: throw Error(__FILE__, __LINE__, VOLT_LOG_CAT, "Unsupported volume data type", Error_NotImplemented);
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
    float elapsedTime;
    VOX_CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    VOX_LOG_INFO(VOLT_LOG_CAT, format("Convolution completed in %1% ms", elapsedTime));

    return result;
}

// ----------------------------------------------------------------------------
//  Constructs a laplace kernel
// ----------------------------------------------------------------------------
void Conv::makeLaplaceKernel(Image3D<float> & kernel)
{
    kernel.resize(3, 3, 3);

    auto data = kernel.buffer().get();
    float vals[] = { 
                     0,   0.5, 0,
                     0.5, 1,   0.5,
                     0,   0.5, 0,
                     
                     0.5, 1,   0.5,
                     1,   -12,  1,
                     0.5, 1,   0.5,

                     0,   0.5, 0,
                     0.5, 1,   0.5,
                     0,   0.5, 0,
                   };
    memcpy(data, vals, kernel.size()*sizeof(float));
}

// ----------------------------------------------------------------------------
//  Constructs a gaussian kernel of the given size
// ----------------------------------------------------------------------------
void Conv::makeGaussianKernel(std::vector<float> & out, float variance, unsigned int size)
{
    unsigned int width = size ? size : ceil(6.f*variance);

    out.resize(size);

    float var2 = variance * variance;
    float K    = 1 / (sqrt(2 * M_PI * var2));
    unsigned int o = (width-1) / 2;
    if (width%2)
    {
        for (unsigned int i = 0; i <= o; i++)
        {
            float val = K * expf(-(float)(i*i) / (2.f * var2));
            out[o+i] = val;
            out[o-i] = val;
        }
    }
    else
    {
        for (unsigned int i = 0; i <= o; i++)
        {
            float x = i + 0.5f;
            float val = K * exp(-x*x / (2 * var2));
            out[o+i+1] = val;
            out[o-i]   = val;
        }
    }
}

// ----------------------------------------------------------------------------
//  Constructs a mean kernel of the given size
// ----------------------------------------------------------------------------
void Conv::makeMeanKernel(std::vector<float> & out, unsigned int size)
{
    out.resize(size, 1.0f / (float)size);
}

} // namespace volt
} // namespace vox