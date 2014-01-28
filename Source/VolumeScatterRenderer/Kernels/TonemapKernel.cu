/* ===========================================================================

	Project: GPU based Volume Scatter Renderer
    
	Description: Performs tonemapping on a HDR input buffer

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

// Include Header
#include "TonemapKernel.h"

// Include Headers
#include "VolumeScatterRenderer/Core/CBuffer.h"
#include "VolumeScatterRenderer/Core/CSampleBuffer.h"

// Include Core Library Headers
#include "VoxLib/Core/Geometry/Color.h"
#include "VoxLib/Core/Geometry/Vector.h"

namespace vox {
    
namespace {
namespace filescope {

    // --------------------------------------------------------------------
    //  Performs reinhard based tonemapping of the input HDR image buffer
    // --------------------------------------------------------------------
    __global__ void tonemapKernel(CSampleBuffer2D sampleBuffer, CImgBuffer2D<ColorRgbaLdr> imageBuffer, float exposure)
    { 	
	    // Establish the image coordinates of this pixel
	    int px = blockIdx.x * blockDim.x + threadIdx.x;
	    int py = blockIdx.y * blockDim.y + threadIdx.y;
        if (px >= sampleBuffer.width() || py >= sampleBuffer.height()) return;

        // Compute the tonemapped RGB space data pixel
        auto sample = sampleBuffer.at(px, py);
    
        // Store the tonemapped result in the RGB output image buffer
        auto & pixel = imageBuffer.at(px, py);
        pixel.r = high( 0.0f, low(sample.l * 255.0f * expf(exposure), 255.0f) );
        pixel.g = high( 0.0f, low(sample.a * 255.0f * expf(exposure), 255.0f) );
        pixel.b = high( 0.0f, low(sample.b * 255.0f * expf(exposure), 255.0f) );
        pixel.a = 255;
    }

} // namespace anonymous
} // namespace filescope

float TonemapKernel::m_elapsedTime;

// --------------------------------------------------------------------
//  Executes the tonemapping kernel for the active device
// --------------------------------------------------------------------
void TonemapKernel::execute(CSampleBuffer2D sampleBuffer, CImgBuffer2D<ColorRgbaLdr> imageBuffer, float exposure)
{
	// Setup the execution configuration
	static const unsigned int BLOCK_SIZE = 16;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks( 
        (sampleBuffer.width()  + threads.x - 1) / threads.x,
        (sampleBuffer.height() + threads.y - 1) / threads.y 
        );
    
	// Execute the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventRecord(start,0);
    filescope::tonemapKernel<<<blocks,threads>>>(sampleBuffer, imageBuffer, exposure);
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    // Acquire the time for this kernel execution
    cudaEventElapsedTime(&m_elapsedTime, start, stop);
}

} // namespace vox