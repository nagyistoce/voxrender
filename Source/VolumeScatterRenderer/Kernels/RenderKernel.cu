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
#include "VolumeScatterRenderer/Scene/CRenderParams.h"

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
    __constant__ CRenderParams     gd_renderParams;     ///< Rendering parameters

    __constant__ ColorLabxHdr gd_backdropClr;       ///< Color of the backdrop for the volume

    // --------------------------------------------------------------------
    //                        TEXTURE SAMPLERS
    // --------------------------------------------------------------------

    texture<UInt8,3,cudaReadModeNormalizedFloat>  gd_volumeTex_UInt8;     ///< Volume data texture
    texture<UInt16,3,cudaReadModeNormalizedFloat> gd_volumeTex_UInt16;    ///< Volume data texture

    texture<float,3,cudaReadModeNormalizedFloat>  gd_opacityTex;  // Opacity data texture
    texture<uchar4,3,cudaReadModeNormalizedFloat> gd_diffuseTex;  // Diffuse data texture

    // --------------------------------------------------------------------
    //	Clips the input ray to the specified bounding box
    // --------------------------------------------------------------------
    VOX_HOST_DEVICE inline void rayBoxIntersection( 
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
    }
    
    // --------------------------------------------------------------------
    //  Uses the appropriate texture sampler to acquire a density value
    //  The return result is the normalized density of the specified point
    //  :TODO: Implement in templatized kernel and check efficiency
    // --------------------------------------------------------------------
    VOX_DEVICE inline float sampleDensity(float x, float y, float z)
    {
        float density;

        switch (gd_volumeBuffer.type())
        {
            case Volume::Type_UInt16: 
                density = static_cast<float>(tex3D(gd_volumeTex_UInt16, 
                                                x*gd_volumeBuffer.invSpacing()[0], 
                                                y*gd_volumeBuffer.invSpacing()[1], 
                                                z*gd_volumeBuffer.invSpacing()[2])); 
                break;

            default: // :TODO: Some CUDA exception/error stuff rather than default to UInt8 
                density = static_cast<float>(tex3D(gd_volumeTex_UInt8, 
                                                x*gd_volumeBuffer.invSpacing()[0], 
                                                y*gd_volumeBuffer.invSpacing()[1], 
                                                z*gd_volumeBuffer.invSpacing()[2])); 
                break;
        }

        return gd_volumeBuffer.normalizeSample(static_cast<float>(density));
    }

    // --------------------------------------------------------------------
    //  Computes the gradient at a specified location using central diffs
    // --------------------------------------------------------------------
    VOX_DEVICE inline Vector3f sampleGradient(Vector3f const& location)
    {
        float const& x = location[0];
        float const& y = location[1];
        float const& z = location[2];

        return Vector3f(
            (sampleDensity(x+1, y  , z  ) - sampleDensity(x-1, y  , z  )),
            (sampleDensity(x  , y+1, z  ) - sampleDensity(x  , y-1, z  )),
            (sampleDensity(x  , y  , z+1) - sampleDensity(x  , y  , z-1))
            );
    }

    // --------------------------------------------------------------------
    //  Samples the opacity texture to provide a sigma absorption value 
    // --------------------------------------------------------------------
    VOX_DEVICE float sampleAbsorption(Vector3f const& location)
    {
        float density = sampleDensity(location[0], location[1], location[2]);

        return density < 0.3 ? 0.0f : -log(0.0f);
        //return tex3D(gd_opacityTex, sampleDensity(location), 0.0f, 0.0f);
    }

    // --------------------------------------------------------------------
    //  Computes the ambient occlusion for the specified sample point
    //  This is done by casting rays out in the hemisphere about the
    //  normal and evaluating the opacity of nearby voxels
    // --------------------------------------------------------------------
    VOX_DEVICE float computeAmbientOcclusion(
        Vector3f & sampleLocation, Vector3f normal)
    {
    }
    
    // --------------------------------------------------------------------
    //  Samples the scene lighting to compute the radiance contribution
    // --------------------------------------------------------------------
    VOX_DEVICE ColorLabxHdr estimateRadiance(
        CRandomGenerator & rng, Ray3f const& location)
    {
        // Sample the gradient at the point of interest
        Vector3f gradient = sampleGradient(location.pos);

        // Compute the ambient occlusion index

        // Sample the scene lighting
        if (gd_lights.size() != 0)
        {
            gradient.normalize();

            Vector3f lightEmission(1.0f, 1.0f, 1.0f);
            Vector3f lightDirection = (gd_lights[0].position() - location.pos).normalize();
            
            // :TODO: Compute attenuation

            Vector3f Lv = lightEmission * high(Vector3f::dot(gradient, lightDirection), 0.0f);

            return ColorLabxHdr(Lv[0], Lv[1], Lv[2]);
        }
        else return ColorLabxHdr(0.0f, 0.0f, 0.0f);
    }

    // --------------------------------------------------------------------
    //  Uses the appropriate texture sampler to acquire a density value
    // --------------------------------------------------------------------
    VOX_DEVICE bool selectVolumeSamplePoint(
        int px, int py,
        CRandomGenerator & rng,
        Ray3f & sampleLocation
        )
    {
        // Initialize the sample ray for marching
        sampleLocation = gd_camera.generateRay(
                            Vector2f(px, py) + rng.sample2D(), // Pixel position
                            rng.sampleDisk());                 // Aperture position

        // Clip the sample ray to the volume's bounding box
        float rayMin = 0.0f, rayMax = 100000.0f;
        rayBoxIntersection(sampleLocation.pos, sampleLocation.dir, 
            Vector3f(0.0f), gd_volumeBuffer.size(), rayMin, rayMax);

        // Offset the ray origin by a fraction of step size
        float rayStepSize = gd_renderParams.primaryStepSize();
        rayMin += rng.sample1D() * rayStepSize;  
        if (rayMin > rayMax) return false;                   // Non-intersection with volume
        sampleLocation.pos += sampleLocation.dir * rayMin;

        // Select a sample depth to evaluate for this iteration
        const float targetOpacity = -log(rng.sample1D());
        
        // Perform ray marching until the sample depth is reached
        float opacity = 0.0f;
        while (true)
        {
            float absorption = sampleAbsorption(sampleLocation.pos);

            opacity += absorption * rayStepSize;

            if (opacity >= targetOpacity) break;

            rayMin += rayStepSize;
            
            if (rayMin > rayMax) return false; // Outside volume => sample background

            sampleLocation.pos += sampleLocation.dir * rayStepSize;
        }

        return true;
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
        if (px >= gd_sampleBuffer.width() || 
            py >= gd_sampleBuffer.height()) return;

        // Construct the thread's random number generator
        CRandomGenerator rng(&gd_rndBuffer0.at(px, py), 
                             &gd_rndBuffer1.at(px, py));

        // Select a sampling point for the volume for this iteration
        Ray3f sampleLocation; bool hit = selectVolumeSamplePoint(px, py, rng, sampleLocation);

        // Evaluate the shading at the sample point
        if (hit)
        {
            gd_sampleBuffer.push(px, py, estimateRadiance(rng, sampleLocation));
        }
        else
        {
            gd_sampleBuffer.push(px, py, gd_backdropClr);
        }
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
//  Sets the camera model for the active device
// --------------------------------------------------------------------
void RenderKernel::setParameters(CRenderParams const& settings)
{
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(filescope::gd_renderParams, &settings, sizeof(settings)));
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
        throw Error(__FILE__, __LINE__, VSR_LOG_CATEGORY,
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
    
	// Opacity texture sampler settings
	filescope::gd_opacityTex.normalized     = true;
    filescope::gd_opacityTex.filterMode     = cudaFilterModeLinear; 
    filescope::gd_opacityTex.addressMode[0] = cudaAddressModeClamp;
    filescope::gd_opacityTex.addressMode[1] = cudaAddressModeClamp;
    filescope::gd_opacityTex.addressMode[2] = cudaAddressModeClamp;
    
    // Specify the format for volume data access
    auto texFormatDescOpacity = cudaCreateChannelDesc(
        32, 0, 0, 0, cudaChannelFormatKindFloat);

	// Bind the volume handle to a texture for sampling
    VOX_CUDA_CHECK(cudaBindTextureToArray(filescope::gd_opacityTex, 
      transfer.opacityHandle(), texFormatDescOpacity));
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