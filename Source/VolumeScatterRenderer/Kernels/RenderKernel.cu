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
#include "VolumeScatterRenderer/Core/Intersect.h"
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
    __constant__ Vector3f          gd_ambient;          ///< Maximum ambient light

    __constant__ ColorLabxHdr gd_backdropClr;       ///< Color of the backdrop for the volume

    // --------------------------------------------------------------------
    //                        TEXTURE SAMPLERS
    // --------------------------------------------------------------------

    texture<UInt8,3,cudaReadModeNormalizedFloat>  gd_volumeTex_UInt8;     ///< Volume data texture
    texture<UInt16,3,cudaReadModeNormalizedFloat> gd_volumeTex_UInt16;    ///< Volume data texture

    texture<float,3,cudaReadModeElementType>      gd_opacityTex;  // Opacity data texture
    texture<uchar4,3,cudaReadModeNormalizedFloat> gd_diffuseTex;  // Diffuse data texture
    texture<uchar4,3,cudaReadModeNormalizedFloat> gd_specularTex; // Specular data texture
    texture<float4,3,cudaReadModeElementType>     gd_emissiveTex; // Emission data texture

    // --------------------------------------------------------------------
    //  Uses the appropriate texture sampler to acquire a density value
    //  The return result is the normalized density of the specified point
    //  :TODO: Implement in templatized kernel and check efficiency
    //  :TODO: Pretransform by invSpacing during tracers
    // --------------------------------------------------------------------
    VOX_DEVICE float sampleDensity(float x, float y, float z)
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
    VOX_DEVICE Vector3f sampleGradient(Vector3f const& location)
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

        return tex3D(gd_opacityTex, density, 0.0f, 0.0f);
    }
    
    // --------------------------------------------------------------------
    //  Computes the ray intersection with the volume, taking into account
    //  any bounding volumes etc...
    // --------------------------------------------------------------------
    VOX_DEVICE void intersectVolume(
        const Vector3f &rayPos, 
        const Vector3f &rayDir, 
	    float &rayMin, 
        float &rayMax
        )
    {
        // Compute the intersection of the ray with the volume's outer bounding box
        Intersect::rayBoxIntersection(rayPos, rayDir, 
            Vector3f(0.0f), gd_volumeBuffer.size(), rayMin, rayMax);

        // Compute the intersection of the ray with clipping planes (:DEBUG:)
        //Intersect::rayPlaneIntersection(rayPos, rayDir, 
        //    Vector3f(0.0f, 1.0f, 0.0f), 150.0f, rayMin, rayMax);
    }

    // --------------------------------------------------------------------
    //  Computes attenuation of light along the specified ray
    // --------------------------------------------------------------------
    VOX_DEVICE inline float computeTransmission(
        CRandomGenerator & rng,
        Vector3f const& pos, 
        Vector3f const& dir,
        float rayStepSize,
        float maxDistance
        )
    {
        // Determine the maximum ray extent
        float rayMin = 0.0f, rayMax = maxDistance;
        intersectVolume(pos, dir, rayMin, rayMax);
        
        // Offset the ray origin by a fraction of step size
        rayMin += rng.sample1D() * rayStepSize;

        // Perform ray marching until the sample depth is reached
        float opacity = 0.0f;
        while (rayMin < rayMax)
        {
            float absorption = sampleAbsorption(pos + dir * rayMin);

            opacity += absorption * rayStepSize;

            rayMin += rayStepSize;
            
            if (rayMin > rayMax) break;
        }

        return max(1.0f - opacity, 0.0f);
    }

    // --------------------------------------------------------------------
    //  Computes the ambient occlusion for the specified sample point
    //  This is done by casting rays out around the sample point and 
    //  computing the average attenuation over a short distance.
    // --------------------------------------------------------------------
    VOX_DEVICE float computeAmbientOcclusion(CRandomGenerator & rng, Vector3f const& pos)
    {
        unsigned int samples = gd_renderParams.occludeSamples();

        // Compute ambient occlusion if specified
        if (samples != 0)
        {
            float transmission = 0.0f;
            for (unsigned int i = 0; i < samples; i++)
            {
                transmission += computeTransmission(rng, pos, rng.sampleSphere(), 
                    gd_renderParams.occludeStepSize(), 5.0f);/*:TODO: occlude distance*/
            }

            return transmission / (float)samples;
        }
        else // Otherwise return no attenuation
        {
            return 1.0f;
        }
    }

    // --------------------------------------------------------------------
    //  Samples the scene lighting to compute the radiance contribution
    // --------------------------------------------------------------------
    VOX_DEVICE ColorLabxHdr estimateRadiance(CRandomGenerator & rng, Ray3f const& location)
    {
        float density = sampleDensity(location.pos[0], location.pos[1], location.pos[2]); // :TODO: Carry density forward with location

        // Compute the gradient at the point of interest
        Vector3f gradient = sampleGradient(location.pos);
        float gradMag = gradient.length();
        gradient = gradient / gradMag;
        if (Vector3f::dot(gradient, location.dir) > 0)
        {
            gradient = -gradient;
        }

        // Determine the diffuse characteristic of the sample point 
        float4 sampleDiffuse = tex3D(gd_diffuseTex, density, 0.0f, 0.0f); 
        Vector3f diffuse(sampleDiffuse.x, sampleDiffuse.y, sampleDiffuse.z);

        // Compute the ambient component of the scene lighting
        Vector3f Lv = diffuse * gd_ambient * computeAmbientOcclusion(rng, location.pos);

        // Sample the scene lighting
        if (gd_lights.size() != 0)
        {
            // Sample the scene lighting for this iteration
            Vector3f lightEmission  = gd_lights[0].color();
            Vector3f lightDirection = (gd_lights[0].position() - location.pos).normalize();

            // Compute the attenuated light reaching the selected scattering point
            lightEmission *= computeTransmission(rng, location.pos, lightDirection,
                                    gd_renderParams.shadowStepSize(), 1000.0f);
            
            // Switch to surface based shading above the gradient threshold
            if (gradMag > gd_renderParams.gradientCutoff())
            {
                // Compute the diffuse component of the reflectance function
                Lv += lightEmission * diffuse * abs(Vector3f::dot(gradient, lightDirection)); 

                // Compute the specular component of the reflectance function

                // Calculate the half vector between the light and view
                Vector3f H = (lightDirection - location.dir).normalized();

                float4 specularData = tex3D(gd_specularTex, density, 0.0f, 0.0f);

                //Intensity of the specular light
                float NdotH = Vector3f::dot(gradient, H);
                float intensity = pow(saturate( NdotH ), specularData.w*100.0f + 2.0f);
 
                // Compute the resulting specular strength
                Lv += Vector3f(specularData.x, specularData.y, specularData.z) * lightEmission * intensity;
            }
            else
            {
                Lv += lightEmission * diffuse * 0.717; 
            }
        }

        return ColorLabxHdr(Lv[0], Lv[1], Lv[2]);
    }

    // --------------------------------------------------------------------
    //  Performs ray marching to locate a sample point at the selected opacity
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
        intersectVolume(sampleLocation.pos, sampleLocation.dir, rayMin, rayMax);

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
        
        Ray3f sampleLocation;

        // Evaluate the shading at the sample point
        if (selectVolumeSamplePoint(px, py, rng, sampleLocation))
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
void RenderKernel::setLights(CBuffer1D<CLight> const& lights, Vector3f const& ambient)
{
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(filescope::gd_lights, &lights, sizeof(lights)));
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(filescope::gd_ambient, &ambient, sizeof(ambient)));
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



	// Specular texture sampler settings
	filescope::gd_specularTex.normalized     = true;
    filescope::gd_specularTex.filterMode     = cudaFilterModeLinear; 
    filescope::gd_specularTex.addressMode[0] = cudaAddressModeClamp;
    filescope::gd_specularTex.addressMode[1] = cudaAddressModeClamp;
    filescope::gd_specularTex.addressMode[2] = cudaAddressModeClamp;
    
    // Specify the format for volume data access
    auto texFormatDescSpecular = cudaCreateChannelDesc(
        8, 8, 8, 8, cudaChannelFormatKindUnsigned);

	// Bind the volume handle to a texture for sampling
    VOX_CUDA_CHECK(cudaBindTextureToArray(filescope::gd_specularTex, 
      transfer.specularHandle(), texFormatDescSpecular));



	// Emissive texture sampler settings
	filescope::gd_emissiveTex.normalized     = true;
    filescope::gd_emissiveTex.filterMode     = cudaFilterModeLinear; 
    filescope::gd_emissiveTex.addressMode[0] = cudaAddressModeClamp;
    filescope::gd_emissiveTex.addressMode[1] = cudaAddressModeClamp;
    filescope::gd_emissiveTex.addressMode[2] = cudaAddressModeClamp;
    
    // Specify the format for volume data access
    auto texFormatDescEmissive = cudaCreateChannelDesc(
        32, 32, 32, 32, cudaChannelFormatKindFloat);

	// Bind the volume handle to a texture for sampling
    VOX_CUDA_CHECK(cudaBindTextureToArray(filescope::gd_emissiveTex, 
      transfer.emissiveHandle(), texFormatDescEmissive));
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