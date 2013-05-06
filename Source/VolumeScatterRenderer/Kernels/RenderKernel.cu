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
#include "VoxLib/Core/Types.h"

namespace vox {

namespace {
namespace filescope {

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

    texture<UInt8,3,cudaReadModeNormalizedFloat> gd_volumeTex_UInt8;      ///< Volume data texture
    texture<UInt8,3,cudaReadModeNormalizedFloat> gd_volumeGradTex_UInt8;  ///< Volume gradient texture

    texture<UInt16,3,cudaReadModeNormalizedFloat> gd_volumeTex_UInt16;      ///< Volume data texture
    texture<UInt16,3,cudaReadModeNormalizedFloat> gd_volumeGradTex_UInt16;  ///< Volume gradient texture

    texture<uchar4,3,cudaReadModeNormalizedFloat> gd_emissionTex; // Emission data texture
    texture<uchar4,3,cudaReadModeNormalizedFloat> gd_diffuseTex;  // Diffuse data texture
    texture<uchar4,3,cudaReadModeNormalizedFloat> gd_specularTex; // Specular data texture

    // ---------------------------------------------------------
    //	Clips the input ray to the specified bounding box, the
    //  return value is true if an intersection occurred
    // ---------------------------------------------------------
    VOX_HOST_DEVICE inline bool rayBoxIntersection( 
        const Vector3f &rayPos, 
        const Vector3f &rayDir, 
        const Vector3f &bmin, 
        const Vector3f &bmax, 
	    float &rayMin, 
        float &rayMax)
    {
        Vector3f const invDir(1.0f / rayDir[0], 1.0f / rayDir[1], 1.0f / rayDir[2]);

	    Vector3f const tBMax = ( bmax - rayPos ) * invDir;
	    Vector3f const tBMin = ( bmin - rayPos ) * invDir;
    
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
    //  Uses the appropriate texture sampler to acquire a density value
    // --------------------------------------------------------------------
    VOX_DEVICE float getSampleDensity(float x, float y, float z)
    {
        switch (gd_volumeBuffer.type())
        {
        case Volume::Type_UInt16: return tex3D(gd_volumeTex_UInt16, x, y, z);
        default:                  return tex3D(gd_volumeTex_UInt8, x, y, z);
        }
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
        CRandomGenerator rng(&gd_rndBuffer0.at(px, py), 
                             &gd_rndBuffer1.at(px, py));
    
        // Generate a sample ray from the camera for this iteration
        Ray3f ray = gd_camera.generateRay(
                        Vector2f(px, py) + rng.sample2D(), // Pixel position
                        rng.sampleDisk());                 // Aperture position

        // Clip the sample ray to the volume's bounding box
        Vector2f clipRange(0.0f, 2000000.0f); 
        bool miss = rayBoxIntersection(ray.pos, ray.dir, Vector3f(0.0f), 
            gd_volumeBuffer.size(), clipRange[0], clipRange[1]);

        // Offset the ray origin by a fraction of step size
        clipRange[0] += rng.sample1D() * gd_rayStepSize;

        ray.pos += ray.dir * clipRange[0];

        // Sample the volume for output radiance information
	    if (!miss) 
        {
            miss = true;
            while (clipRange[0] < clipRange[1])
	        {
                // Acquire an interpolated volume sample value at current position
                float density = getSampleDensity( 
                    ray.pos[0]*gd_volumeBuffer.invSpacing()[0], 
                    ray.pos[1]*gd_volumeBuffer.invSpacing()[1],
                    ray.pos[2]*gd_volumeBuffer.invSpacing()[2]);

                miss = (density < 0.1f); if (!miss) break;

                // Increment the current sample position
                ray.pos += ray.dir * gd_rayStepSize;
		        clipRange[0] += gd_rayStepSize;
	        }
        }

        __syncthreads();

        // :DEBUG: test output
        if (miss) gd_sampleBuffer.push(px, py, gd_backdropClr);
        else gd_sampleBuffer.push(px, py, ColorLabxHdr(rng.sample1D(), rng.sample1D(), rng.sample1D())); 
    }

} // namespace filescope
} // namespace anonymous

// --------------------------------------------------------------------
//  Sets the camera model for the active device
// --------------------------------------------------------------------
void RenderKernel::setCamera(CCamera const& camera)
{
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(filescope::gd_camera, &camera, sizeof(camera)));
}

// --------------------------------------------------------------------
//  Sets the lighting arrangement for the active device
// --------------------------------------------------------------------
void RenderKernel::setLights(CBuffer1D<CLight> const& lights)
{
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(filescope::gd_lights, &lights, sizeof(lights)));
}

// --------------------------------------------------------------------
//  Sets the volume data buffer for the active device
// --------------------------------------------------------------------
void RenderKernel::setVolume(CVolumeBuffer const& volume)
{
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(filescope::gd_volumeBuffer, &volume, sizeof(volume)));

    // Select the appropriate sampler for the data type
    switch (volume.type())
    {
    case Volume::Type_UInt8: 
    {
	    // Volume texture sampler settings
	    filescope::gd_volumeTex_UInt8.normalized     = false;
        filescope::gd_volumeTex_UInt8.filterMode     = cudaFilterModeLinear; 
        filescope::gd_volumeTex_UInt8.addressMode[0] = cudaAddressModeClamp;
        filescope::gd_volumeTex_UInt8.addressMode[1] = cudaAddressModeClamp;
        filescope::gd_volumeTex_UInt8.addressMode[2] = cudaAddressModeClamp;

	    // Bind the volume handle to a texture for sampling
        VOX_CUDA_CHECK(cudaBindTextureToArray(filescope::gd_volumeTex_UInt8, 
            volume.handle(), volume.formatDescriptor()));
        break;
    }

    case Volume::Type_UInt16: 
    {
	    // Volume texture sampler settings
	    filescope::gd_volumeTex_UInt16.normalized     = false;
        filescope::gd_volumeTex_UInt16.filterMode     = cudaFilterModeLinear; 
        filescope::gd_volumeTex_UInt16.addressMode[0] = cudaAddressModeClamp;
        filescope::gd_volumeTex_UInt16.addressMode[1] = cudaAddressModeClamp;
        filescope::gd_volumeTex_UInt16.addressMode[2] = cudaAddressModeClamp;

	    // Bind the volume handle to a texture for sampling
        VOX_CUDA_CHECK(cudaBindTextureToArray(filescope::gd_volumeTex_UInt16, 
            volume.handle(), volume.formatDescriptor()));
        break;
    }

    default:
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
            format("Unsupported volume data type (%1%)", 
                   Volume::typeToString(volume.type())),
            Error_NotImplemented);
    }
}

// --------------------------------------------------------------------
//  Sets the transfer function for the active device
// --------------------------------------------------------------------
void RenderKernel::setTransfer(CTransferBuffer const& transfer)
{
	// Diffuse texture sampler settings
	filescope::gd_diffuseTex.normalized     = true;
    filescope::gd_diffuseTex.filterMode     = cudaFilterModeLinear; 
    filescope::gd_diffuseTex.addressMode[0] = cudaAddressModeClamp;
    filescope::gd_diffuseTex.addressMode[1] = cudaAddressModeClamp;
    filescope::gd_diffuseTex.addressMode[2] = cudaAddressModeClamp;

    // Specify the format for diffuse data access
    auto texFormatDescDiffuse = cudaCreateChannelDesc(8, 8, 8, 8, 
        cudaChannelFormatKindUnsigned);

	// Bind the volume handle to a texture for sampling
    VOX_CUDA_CHECK(cudaBindTextureToArray(filescope::gd_diffuseTex, 
      transfer.diffuseHandle(), texFormatDescDiffuse));
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
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(filescope::gd_sampleBuffer, &sampleBuffer, sizeof(sampleBuffer)));
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(filescope::gd_rndBuffer0,   &rndSeeds0,    sizeof(rndSeeds0)));
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(filescope::gd_rndBuffer1,   &rndSeeds1,    sizeof(rndSeeds1)));
}

// --------------------------------------------------------------------
//  Executes the rendering stage kernel on the active device
// --------------------------------------------------------------------
void RenderKernel::execute(size_t xstart, size_t ystart,
                           size_t width,  size_t height)
{
    float const step = 2.0f;
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(filescope::gd_rayStepSize, &step, sizeof(float)));
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(filescope::gd_backdropClr, &ColorLabxHdr(1.0f, 1.0f, 1.0f), sizeof(ColorLabxHdr)));

	// Setup the execution configuration
	static const unsigned int BLOCK_SIZE = 32;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks( 
        (width + threads.x - 1) / threads.x,
		(height + threads.y - 1) / threads.y 
        );

	// Execute the device rendering kernel
	filescope::renderKernel<<<blocks,threads>>>();
}

} // namespace vox