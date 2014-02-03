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

// Include Clip Geometry Functions
#include "VolumeScatterRenderer/Clip/CClipGroup.cuh"
#include "VolumeScatterRenderer/Clip/CClipPlane.cuh"

// Additional Includes
#define _USE_MATH_DEFINES
#include <math.h>
    
#define R3I                 0.57735026918962576450914878050196f;
#define KERNEL_BLOCK_W		16
#define KERNEL_BLOCK_H		16
#define KERNEL_BLOCK_SIZE   (KERNEL_BLOCK_W * KERNEL_BLOCK_H)

namespace vox {
    
float RenderKernel::m_elapsedTime;

namespace {
namespace filescope {

    // --------------------------------------------------------------------
    //                   HOST HANDLES FOR DEVICE DATA
    // --------------------------------------------------------------------

    std::shared_ptr<CClipGeometry> gh_clipRoot;

    // --------------------------------------------------------------------
    //                        RENDER PARAMETERS
    // --------------------------------------------------------------------

    __constant__ CCamera           gd_camera;           ///< Device camera model
    __constant__ CBuffer1D<CLight> gd_lights;           ///< Device light buffer
    __constant__ CSampleBuffer2D   gd_sampleBuffer;     ///< HDR sample data buffer
    __constant__ CVolumeBuffer     gd_volumeBuffer;     ///< Device volume buffer
    __constant__ CRenderParams     gd_renderParams;     ///< Rendering parameters
    __constant__ Vector3f          gd_ambient;          ///< Maximum ambient light
    __constant__ curandState *     gd_randStates;       ///< Random generator states

    __constant__ CClipGeometry::Clipper * gd_clipRoot;    ///< Clipping geometry root

    __constant__ ColorLabxHdr gd_backdropClr;       ///< Color of the backdrop for the volume

    // --------------------------------------------------------------------
    //                        TEXTURE SAMPLERS
    // --------------------------------------------------------------------

#define VOX_TEXTURE(T) texture<##T,3,cudaReadModeNormalizedFloat>  gd_volumeTex_##T
    VOX_TEXTURE(Int8);
    VOX_TEXTURE(UInt8);
    VOX_TEXTURE(Int16);
    VOX_TEXTURE(UInt16);
#undef VOX_TEXTURE

    texture<float,3,cudaReadModeElementType>      gd_opacityTex;  // Opacity data texture
    texture<uchar4,3,cudaReadModeNormalizedFloat> gd_diffuseTex;  // Diffuse data texture
    texture<uchar4,3,cudaReadModeNormalizedFloat> gd_specularTex; // Specular data texture
    texture<float4,3,cudaReadModeElementType>     gd_emissiveTex; // Emission data texture

    // --------------------------------------------------------------------
    //  Uses the appropriate texture sampler to acquire a density value
    // --------------------------------------------------------------------
    VOX_DEVICE float sampleDensity(float x, float y, float z)
    {
        float density;

        switch (gd_volumeBuffer.type())
        {
            case Volume::Type_Int16: 
                density = static_cast<float>(tex3D(gd_volumeTex_Int16, 
                                                x*gd_volumeBuffer.invSpacing()[0], 
                                                y*gd_volumeBuffer.invSpacing()[1], 
                                                z*gd_volumeBuffer.invSpacing()[2])); 
                break;

            case Volume::Type_UInt16: 
                density = static_cast<float>(tex3D(gd_volumeTex_UInt16, 
                                                x*gd_volumeBuffer.invSpacing()[0], 
                                                y*gd_volumeBuffer.invSpacing()[1], 
                                                z*gd_volumeBuffer.invSpacing()[2])); 
                break;

            case Volume::Type_Int8:
                density = static_cast<float>(tex3D(gd_volumeTex_Int8, 
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
        // :TODO: Factor in clip plane
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
        float gradMag = sampleGradient(location).length() * R3I;

        return tex3D(gd_opacityTex, density, gradMag, 0.0f);
    }
    
    // --------------------------------------------------------------------
    //  Computes the ray intersection with the volume, taking into account
    //  any bounding volumes etc...
    // --------------------------------------------------------------------
    VOX_DEVICE void intersectVolume(Ray3f & ray)
    {
        // Compute the intersection with the volume extent box
        Intersect::rayBoxIntersection(ray.pos, ray.dir, 
            Vector3f(0.0f), gd_volumeBuffer.size(), ray.min, ray.max);

        // Compute the intersection with the scene clip geometry
        if (filescope::gd_clipRoot) filescope::gd_clipRoot->clip(ray);
    }

    // --------------------------------------------------------------------
    //  Computes attenuation of light along the specified ray
    // --------------------------------------------------------------------
    VOX_DEVICE inline float computeTransmission(CRandomGenerator & rng, Ray3f & sampleRay)
    {
        // Clip the ray to the scene geometry
        intersectVolume(sampleRay);
        
        // Offset the ray origin by a fraction of step size
        sampleRay.min += rng.sample1D() * gd_renderParams.shadowStepSize();
        sampleRay.pos += sampleRay.dir * sampleRay.min;
        sampleRay.dir *= gd_renderParams.shadowStepSize();

        // Perform ray marching until the sample depth is reached
        float opacity = 0.0f;
        while (sampleRay.min < sampleRay.max)
        {
            sampleRay.pos += sampleRay.dir;

            opacity += sampleAbsorption(sampleRay.pos) * gd_renderParams.shadowStepSize();

            if (opacity > 1.0f) break;

            sampleRay.min += gd_renderParams.shadowStepSize();
        }

        sampleRay.dir /= gd_renderParams.shadowStepSize();
        sampleRay.min = 0;
        sampleRay.max = 0;

        return max(1.0f - opacity, 0.0f);
    }

    // --------------------------------------------------------------------
    //  Obscurance model for approximating ambient lighting more quickly
    // --------------------------------------------------------------------
    VOX_DEVICE float computeObscurance(CRandomGenerator & rng, Vector3f const& pos)
    {
            Ray3f ray(pos, rng.sampleSphere(), 0.0f, 10.0f);
            return computeTransmission(rng, ray);
    }

    // --------------------------------------------------------------------
    //  Samples the scene lighting to compute the radiance contribution
    // --------------------------------------------------------------------
    VOX_DEVICE ColorLabxHdr estimateRadiance(CRandomGenerator & rng, Vector3f const& pos, Vector3f const& dir)
    {
        // Compute the gradient at the point of interest
        Vector3f gradient = sampleGradient(pos);
        float gradMag = gradient.length();
        gradient = gradient / gradMag;
        gradMag *= R3I;
        // -- Adjust gradient towards camera for lighting computations
        float part = Vector3f::dot(gradient, dir);
        if (part > 0) gradient = -gradient; 
        
        // Perform edge enhancement (ie. Shade black around locations with gradients perpindicular to the eye direction)
        if (gradMag > gd_renderParams.gradientCutoff() && abs(part) < gd_renderParams.edgeEnhancement()) return ColorLabxHdr(0.0f, 0.0f, 0.0f);

        float density = sampleDensity(pos[0], pos[1], pos[2]); // :TODO: Carry density forward with location
        
        // Determine the diffuse characteristic of the sample point 
        float4 sampleDiffuse = tex3D(gd_diffuseTex, density, gradMag, 0.0f); 
        Vector3f diffuse(sampleDiffuse.x, sampleDiffuse.y, sampleDiffuse.z);
        Vector3f Lv = gd_ambient * diffuse * computeObscurance(rng, pos);

        if (gd_lights.size())
        {
            // Determine the sample light for this iteration
            auto const lightNum = (int)(ceilf(rng.sample1D() * ((float)(gd_lights.size()))) - 1.0f);

            // Compute the direction and distance to the light sample
            Vector3f lightDirection = gd_lights[lightNum].position() - pos;;
            float    lightDistance  = lightDirection.length();
            lightDirection *= (1.0f / lightDistance);

            // Compute the incident lighting at the scatter point
            float    lightTransmit  = computeTransmission(rng, Ray3f(pos-dir*gd_renderParams.primaryStepSize(), lightDirection, 0.0f, lightDistance));
            Vector3f lightIncident  = gd_lights[lightNum].color() * lightTransmit * gd_lights.size();

            // Switch to surface based shading above the gradient threshold
            if (gradMag > gd_renderParams.gradientCutoff())
            {
                // *** Compute the diffuse component of the reflectance function ***
                Lv += diffuse * lightIncident * abs(Vector3f::dot(gradient, lightDirection)); 

                // *** Compute the specular component of the reflectance function ***
                float4   specularData = tex3D(gd_specularTex, density, gradMag, 0.0f);
                Vector3f specular     = Vector3f(specularData.x, specularData.y, specularData.z) * lightIncident;

                //Intensity of the specular light
                Vector3f H = (lightDirection - dir).normalized(); // Half vector between light and view
                float NdotH = Vector3f::dot(gradient, H);
                float intensity = pow(saturate( NdotH ), specularData.w*100.0f + 2.0f);

                // Compute the resulting specular strength
                Lv += specular * intensity;
            }
            else
            {
                auto const g  = gd_renderParams.scatterCoefficient();
                auto cosTheta = - Vector3f::dot(dir, lightDirection);
                auto phaseHG  = (1 - g*g) / pow(1.0f + g*g + 2*g*cosTheta, 1.5f);

                Lv += lightIncident * phaseHG * diffuse;
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
        Vector3f & pos, Vector3f & dir
        )
    {
        // Initialize the sample ray for marching
        auto sampleRay = gd_camera.generateRay(
                            Vector2f(px, py) + rng.sample2D(), // Pixel position
                            rng.sampleDisk());                 // Aperture position

        // Clip the sample ray to the scene geometry
        intersectVolume(sampleRay);

        // Offset the ray origin by a fraction of step size
        float rayStepSize = gd_renderParams.primaryStepSize();
        sampleRay.min += rng.sample1D() * rayStepSize;  
        if (sampleRay.min > sampleRay.max) return false;        // Non-intersection with volume
        sampleRay.pos += sampleRay.dir * sampleRay.min;

        // Select a sample depth to evaluate for this iteration
        const float targetOpacity = -log(rng.sample1D());
        
        // Perform ray marching until the sample depth is reached
        float opacity = 0.0f;
        while (true)
        {
            float absorption = sampleAbsorption(sampleRay.pos);

            opacity += absorption * rayStepSize;

            if (opacity >= targetOpacity) break;

            sampleRay.min += rayStepSize;
            
            if (sampleRay.min > sampleRay.max) return false; // Outside volume => sample environment

            sampleRay.pos += sampleRay.dir * rayStepSize;
        }

        pos = sampleRay.pos;
        dir = sampleRay.dir;

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
        CRandomGenerator rng(gd_randStates[px + py * gd_sampleBuffer.height()]);
        
        // Compute the volume sample point
        Vector3f sampPos, sampDir;
        bool hit = selectVolumeSamplePoint(px, py, rng, sampPos, sampDir);

        __syncthreads();

        // Evaluate the shading at a single volume point ...
        if (hit)
        {
            gd_sampleBuffer.push(px, py, estimateRadiance(rng, sampPos, sampDir));
        }
        else // ... or sample environment
        {
            gd_sampleBuffer.push(px, py, gd_backdropClr);
        }

        // Store the CRNG state for subsequent launches 
        gd_randStates[px + py * gd_sampleBuffer.height()] = rng.state();
    }

} // namespace filescope
} // namespace anonymous

// --------------------------------------------------------------------
//  Sets the root primitive for clipping operations
// --------------------------------------------------------------------
void RenderKernel::setClipRoot(std::shared_ptr<CClipGeometry> root)
{
    CClipGeometry::Clipper * ptr = root ? root->clipper() : nullptr;
        
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(filescope::gd_clipRoot, &ptr, sizeof(CClipGeometry::Clipper*)));

    filescope::gh_clipRoot = root; // Store the pointer so we don't free the memory accidently
}

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

#define VOX_SETUP_TEX(T)                                                   \
    case Volume::Type_##T: {                                               \
	    filescope::gd_volumeTex_##T.normalized     = false;                \
        filescope::gd_volumeTex_##T.filterMode     = cudaFilterModeLinear; \
        filescope::gd_volumeTex_##T.addressMode[0] = cudaAddressModeClamp; \
        filescope::gd_volumeTex_##T.addressMode[1] = cudaAddressModeClamp; \
        filescope::gd_volumeTex_##T.addressMode[2] = cudaAddressModeClamp; \
        VOX_CUDA_CHECK(cudaBindTextureToArray(filescope::gd_volumeTex_##T, \
            volume.handle(), volume.formatDescriptor()));                  \
        break; }

    // Select the appropriate sampler for the data type
    switch (volume.type())
    {
    VOX_SETUP_TEX(UInt8);
    VOX_SETUP_TEX(UInt16);
    VOX_SETUP_TEX(Int8);
    VOX_SETUP_TEX(Int16);

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
void RenderKernel::setFrameBuffers(CSampleBuffer2D const& sampleBuffer, curandState * randStates)
{
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(filescope::gd_sampleBuffer, &sampleBuffer, sizeof(sampleBuffer)));
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(filescope::gd_randStates, &randStates, sizeof(randStates)));
}

// --------------------------------------------------------------------
//  Executes the rendering stage kernel on the active device
// --------------------------------------------------------------------
void RenderKernel::execute(size_t xstart, size_t ystart,
                           size_t width,  size_t height)
{
    VOX_CUDA_CHECK(cudaMemcpyToSymbol(filescope::gd_backdropClr, &ColorLabxHdr(0.0f, 0.0f, 0.0f), sizeof(ColorLabxHdr)));

	// Setup the execution configuration
    dim3 threads(KERNEL_BLOCK_W, KERNEL_BLOCK_H);
    dim3 blocks( 
        (width + threads.x - 1) / threads.x,
		(height + threads.y - 1) / threads.y 
        );

	// Execute the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventRecord(start,0);
	filescope::renderKernel<<<blocks,threads>>>();
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    // Acquire the time for this kernel execution
    cudaEventElapsedTime(&m_elapsedTime, start, stop);
}

} // namespace vox