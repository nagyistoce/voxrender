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

// Include Header
#include "RandKernel.h"

// Include Dependencies
#include "VoxLib/Error/CudaError.h"

namespace {
namespace filescope {

    // --------------------------------------------------------------------
    //  Generates a set of CRNG states
    // --------------------------------------------------------------------
    __global__ void randKernel(size_t width, size_t height, curandState * state)
    { 	
	    // Establish the image coordinates of this pixel
	    int px = blockIdx.x * blockDim.x + threadIdx.x;
	    int py = blockIdx.y * blockDim.y + threadIdx.y;
        if (px >= width || py >= height) return;

        int id = px + py * height;
        curand_init(0, id, 0, &state[id]);
    }

} // namespace filescope
} // namespace anonymous

namespace vox {

float RandKernel::m_elapsedTime;

// --------------------------------------------------------------------
//  Executes the rand initialization kernel for the active device
// --------------------------------------------------------------------
curandState * RandKernel::create(size_t width, size_t height)
{
    curandState * state;
    VOX_CUDA_CHECK(cudaMalloc((void**)&state, width*height*sizeof(curandState)));

	// Setup the execution configuration
	static const unsigned int BLOCK_SIZE = 16;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks( 
        (width  + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y 
        );
    
	// Execute the kernel
    cudaEvent_t start, stop;
    VOX_CUDA_CHECK(cudaEventCreate(&start));
    VOX_CUDA_CHECK(cudaEventRecord(start,0));
    filescope::randKernel<<<blocks,threads>>>(width, height, state);
    VOX_CUDA_CHECK(cudaDeviceSynchronize());
    VOX_CUDA_CHECK(cudaEventCreate(&stop));
    VOX_CUDA_CHECK(cudaEventRecord(stop,0));
    VOX_CUDA_CHECK(cudaEventSynchronize(stop));

    // Acquire the time for this kernel execution
    VOX_CUDA_CHECK(cudaEventElapsedTime(&m_elapsedTime, start, stop));

    return state;
}

// --------------------------------------------------------------------
//  frees a block of global memory holding CRNG state info
// --------------------------------------------------------------------
void RandKernel::destroy(curandState * states)
{
    VOX_CUDA_CHECK(cudaFree(states));
}

} // namespace vox