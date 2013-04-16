/* ===========================================================================

	Project: GPU based Volume Scatter Renderer
    
	Description: Abstracts the CUDA kernel operations from host code

    Copyright (C) 2013 Lucas Sherman

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
#include "RenderKernel.h"

// Include Headers
#include "VolumeScatterRenderer/Core/CBuffer.h"
#include "VolumeScatterRenderer/Core/CRandomBuffer.h"
#include "VolumeScatterRenderer/Core/CSampleBuffer.h"
#include "VolumeScatterRenderer/Core/CRandomGenerator.h"
#include "VolumeScatterRenderer/Scene/CCamera.h"
#include "VolumeScatterRenderer/Scene/CLight.h"
#include "VolumeScatterRenderer/Scene/CTransferBuffer.h"
#include "VolumeScatterRenderer/Scene/CVolumeBuffer.h"

// Include Core Library Headers
#include "VoxLib/Core/Geometry/Color.h"
#include "VoxLib/Core/Geometry/Vector.h"

namespace vox {

// --------------------------------------------------------------------
//                        RENDER PARAMETERS
// --------------------------------------------------------------------

__constant__ CCamera           gd_camera;           ///< Device camera model
__constant__ CBuffer1D<CLight> gd_lights;           ///< Device light buffer
__constant__ CRandomBuffer2D   gd_rndBuffer0;       ///< Device RNG seed buffer
__constant__ CRandomBuffer2D   gd_rndBuffer1;       ///< Device RNG seed buffer
__constant__ CSampleBuffer2D   gd_sampleBuffer;     ///< HDR sample data buffer
__constant__ CTransferBuffer   gd_transferBuffer;   ///< Transfer function info
__constant__ CVolumeBuffer     gd_volumeBuffer;     ///< Device volume buffer

// CRenderParms : wrap step sizes, clip values, etc in scene struct

__constant__ float        gd_rayStepSize; ///< Base step size of the volume trace ray
__constant__ ColorLabxHdr gd_backdropClr; ///< Color of the backdrop for the volume

// --------------------------------------------------------------------
//                        TEXTURE SAMPLERS
// --------------------------------------------------------------------

texture<unsigned char,3,cudaReadModeNormalizedFloat> gd_volumeTex;      ///< Volume data texture
texture<unsigned char,3,cudaReadModeNormalizedFloat> gd_volumeGradTex;  ///< Volume gradient texture

texture<float4,3,cudaReadModeElementType> gd_emissionTex; // Emission data texture
texture<float4,3,cudaReadModeElementType> gd_diffuseTex;  // Absorption data texture
texture<float4,3,cudaReadModeElementType> gd_specularTex; // Specular data texture

// ---------------------------------------------------------
//	Clips the input ray to the specified bounding box, the
//  return value is true if an intersection occurred
// :TODO: Move to header file
// ---------------------------------------------------------
VOX_HOST_DEVICE bool rayBoxIntersection( 
    const Vector3f &rayPos, 
    const Vector3f &rayDir, 
    const Vector3f &bmin, 
    const Vector3f &bmax, 
	float &rayMin, 
    float &rayMax 
    )
{
    Vector3f const invDir(1.0f / rayDir[0], 1.0f / rayDir[1], 1.0f / rayDir[2]);

	Vector3f const tBMin = ( bmin - rayPos ) * invDir;
	Vector3f const tBMax = ( bmax - rayPos ) * invDir;
    
	Vector3f const tNear( low(tBMin[0], tBMax[0]), 
                          low(tBMin[1], tBMax[1]), 
                          low(tBMin[2], tBMax[2]) );

	Vector3f const tFar ( high(tBMin[0], tBMax[0]), 
                          high(tBMin[1], tBMax[1]), 
                          high(tBMin[2], tBMax[2]) );
    
	rayMin = high(rayMin, high(tNear[0], high(tNear[1], tNear[2])));
	rayMax = low(rayMax, low(tFar[0], low(tFar[1], tFar[2])));
	
	return rayMin > rayMax;
}

// --------------------------------------------------------------------
//  Performs a single pass of the rendering algorithm over the given
//  region of the image buffer
// --------------------------------------------------------------------
__global__ void renderKernel()
{ 	
	// Establish the image coordinates of this pixel
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= gd_sampleBuffer.width() || py >= gd_sampleBuffer.height()) return;

    // Construct the thread's random number generator
    CRandomGenerator rng;
    rng.setSeeds(&gd_rndBuffer0.at(px, py), 
                 &gd_rndBuffer1.at(px, py));
    
    // Generate a sample ray from the camera for this iteration
    Ray3f ray = gd_camera.generateRay(
                    Vector2f(px, py) + rng.sample2D(), // Pixel position
                    rng.sampleDisk());                 // Aperture position

    // Clip the sample ray to the volume's bounding box
    Vector2f clipRange(0.0f, 5000.0f); //:TODO: float max
    bool hit = rayBoxIntersection(
        ray.pos, 
        ray.dir, 
        Vector3f(0.0f), 
        gd_volumeBuffer.size(), 
        clipRange[0], 
        clipRange[1]);

    if (hit) { gd_sampleBuffer.push(px, py, gd_backdropClr); return; }

    // Sample the volume for output radiance information
	while (clipRange[0] > clipRange[2])
	{
		clipRange[0] += gd_rayStepSize;
	}

    // :DEBUG: test output
    gd_sampleBuffer.push(px, py, ColorLabxHdr(rng.sample1D(), rng.sample1D(), rng.sample1D())); 
}

// --------------------------------------------------------------------
//  Sets the camera model for the active device
// --------------------------------------------------------------------
void RenderKernel::setCamera(CCamera const& camera)
{
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(gd_camera, &camera, sizeof(camera)));
}

// --------------------------------------------------------------------
//  Sets the lighting arrangement for the active device
// --------------------------------------------------------------------
void RenderKernel::setLights(CBuffer1D<CLight> const& lights)
{
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(gd_lights, &lights, sizeof(lights)));
}

// --------------------------------------------------------------------
//  Sets the volume data buffer for the active device
// --------------------------------------------------------------------
void RenderKernel::setVolume(CVolumeBuffer const& volume)
{
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(gd_volumeBuffer, &volume, sizeof(volume)));

	// Volume texture sampler settings
	gd_volumeTex.normalized     = true;
    gd_volumeTex.filterMode     = cudaFilterModeLinear; 
    gd_volumeTex.addressMode[0] = cudaAddressModeClamp;
    gd_volumeTex.addressMode[1] = cudaAddressModeClamp;
    gd_volumeTex.addressMode[2] = cudaAddressModeClamp;

    // Specify the format for volume data access
    auto texFormatDesc = cudaCreateChannelDesc(
        volume.voxelSize()*8, 0, 0, 0, 
        cudaChannelFormatKindUnsigned);

	// Bind the volume handle to a texture for sampling
    VOX_CUDA_CHECK(cudaBindTextureToArray(gd_volumeTex, volume.handle(), texFormatDesc));
}

// --------------------------------------------------------------------
//  Sets the transfer function for the active device
// --------------------------------------------------------------------
void RenderKernel::setTransfer(CTransferBuffer const& transfer)
{
    // :TODO: Set textures etc
}

// --------------------------------------------------------------------
//  Sets the device framebuffers used for rendering/post-processing
// --------------------------------------------------------------------
void RenderKernel::setFrameBuffers(
    CSampleBuffer2D const& sampleBuffer,
    CRandomBuffer2D const& rndSeeds0,
    CRandomBuffer2D const& rndSeeds1
    )
{
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(gd_sampleBuffer, &sampleBuffer, sizeof(sampleBuffer)));
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(gd_rndBuffer0,   &rndSeeds0,    sizeof(rndSeeds0)));
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(gd_rndBuffer1,   &rndSeeds1,    sizeof(rndSeeds1)));
}

// --------------------------------------------------------------------
//  Executes the rendering stage kernel on the active device
// --------------------------------------------------------------------
void RenderKernel::execute(size_t xstart, size_t ystart,
                           size_t width,  size_t height)
{
    float const step = 2.0f;
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(gd_rayStepSize, &step, sizeof(float)));
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(gd_backdropClr, &ColorLabxHdr(1.0f, 1.0f, 1.0f), sizeof(ColorLabxHdr)));

	// Setup the execution configuration
	static const unsigned int BLOCK_SIZE = 16;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks( 
        (width + threads.x - 1) / threads.x,
		(height + threads.y - 1) / threads.y 
        );

	// Execute the device rendering kernel
	renderKernel<<<blocks,threads>>>();
}

} // namespace vox