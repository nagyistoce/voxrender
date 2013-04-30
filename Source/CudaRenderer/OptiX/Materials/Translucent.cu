/* ===========================================================================

	Project: CUDA Renderer - Translucent

	Defines a material for a single scatter volume trace through a photon map

    Copyright (C) 2012-2013 Lucas Sherman

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

// CudaRenderer Include Headers
#include "CudaRenderer/OptiX/Materials/Phong.h"
#include "CudaRenderer/OptiX/RayTypes.h"

// Optix SDK Headers
#include <optix.h> // Core library header
#include <optixu/optixu_math_namespace.h>

using namespace optix;
using namespace vox;

// =====================================
//         GEOMETRY ATTRIBUTES
// =====================================

rtDeclareVariable(float3, texcoord,         attribute texcoord,         ); ///< Geometry texture UV attribute
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); ///< Geometry normal attribute
rtDeclareVariable(float3, shading_normal,   attribute shading_normal,   ); ///< Geometry normal attribute

// =====================================
//          SHADER PARAMETERS
// =====================================

rtDeclareVariable(float,  rayStepSize, , );    ///< Step size of the volume trace ray
rtDeclareVariable(float,  specularFactor, , ); ///< Percent specular reflectivity of the surface 
rtDeclareVariable(float3, invSpacing, , );        ///< Inverse spacing between volume samples per dimension
rtDeclareVariable(float3, anchor, , );         ///< Anchor position of volume in object space
rtDeclareVariable(float3, transmission, , );   ///< Transmission coefficient of material
rtDeclareVariable(uint3,  extent, , );         ///< Extent of volume data 

rtTextureSampler<float4,3> volumeTexture;


//rtTextureSampler<uchar4, 1, cudaReadModeNormalizedFloat> transferTexture;

// ---------------------------------------------------------------------
//  Optix Program for computing light attenuation for shadow rays 
// ---------------------------------------------------------------------
RT_PROGRAM void any_hit_shadow()
{
    payloadShadow.attenuation = Vector3f(0.0f);

    rtTerminateRay();
}

// ---------------------------------------------------------------------
//  Optix Program for computing radiance along a ray through the volume
// ---------------------------------------------------------------------
RT_PROGRAM void closest_hit_radiance()
{
    if (payloadRadiance.isInVolume)
    {
        // --------------------------------------------------------------------
        //  Refract the ray at the volume exit point and compute the radiance
        // --------------------------------------------------------------------

        //float3 world_shading_normal   = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
        //float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
        //float3 ffnormal  = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

        //float3 R           = optix::reflect(ray.direction, p_normal);
        //optix::Ray reflRay = optix::make_Ray(hit_point, R, radianceRayType, sceneEpsilon, RT_DEFAULT_MAX);
        
        // Initialize the reflection ray
        payloadRadiance.isInVolume = false;
        payloadRadiance.color.x = t_hit; // :TODO: Use LAB color member and create float member 'closestHit'
                                         // :TODO: Ray refraction calculation on exiting the volume

        float3 pos = ray.origin + t_hit * ray.direction;
        optix::Ray selfRay = optix::make_Ray(pos, ray.direction, radianceRayType, sceneEpsilon, RT_DEFAULT_MAX);
        rtTrace(geometryRoot, selfRay, payloadRadiance);

        //payloadRadiance = selfPayload;
    }
    else
    {
        float3 pos = ray.origin + t_hit * ray.direction; // Current ray position

        // --------------------------------------------------------------------
        //  Compute the radiance component of the ray along the reflected ray
        // --------------------------------------------------------------------

        // Initialize the reflection ray
        RayPayloadRadiance reflPayload;             
        reflPayload.importance = payloadRadiance.importance;
        reflPayload.depth      = payloadRadiance.depth + 1;
        reflPayload.isInVolume = false;

        // Compute the object space normal for reflection calculations
        float3 world_shading_normal   = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
        float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
        float3 normal                 = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
    
        // Trace the reflectance ray and compute radiance contribution
        float3 R = optix::reflect(ray.direction, normal);
        optix::Ray reflRay = optix::make_Ray(pos, R, radianceRayType, sceneEpsilon, RT_DEFAULT_MAX);
        rtTrace(geometryRoot, reflRay, reflPayload);
            
        payloadRadiance.color.l = reflPayload.color.l * specularFactor;
        payloadRadiance.color.a = reflPayload.color.a * specularFactor;
        payloadRadiance.color.b = reflPayload.color.b * specularFactor;
        
        // --------------------------------------------------------------------
        //  Compute the radiance component of the ray along the incident ray 
        //  excluding the volume trace information
        // --------------------------------------------------------------------
        
        // Initialize the incident ray payload
        reflPayload.isInVolume = true;

        // Trace the reflectance ray and compute radiance contribution
        optix::Ray incidentRay = optix::make_Ray(pos, ray.direction, radianceRayType, sceneEpsilon, RT_DEFAULT_MAX);
        rtTrace(geometryRoot, incidentRay, reflPayload);

        // --------------------------------------------------------------------
        // Compute the input radiance contribution from the volume and compute 
        // the light attenuation for the portion of the ray through the volume
        // --------------------------------------------------------------------

        // :TODO: ray step size should be modified based on scale factor so that actual
        //        untransformed step is correct

        unsigned int steps = reflPayload.color.x / rayStepSize; // tmax / stepSize

        // Compute the volume trace parameters 
        pos = (pos - anchor) * invSpacing;      // Transform ray to object space
                                                         // :TODO: sampleOne() offset computation to prvent aliasing
        float3 radiance = make_float3(0.0f, 0.0f, 0.0f); // Color emitted by internal radiance 
        float3 alpha    = make_float3(1.0f, 1.0f, 1.0f); // Transmission along the ray

        float3 step     = ray.direction * rayStepSize * invSpacing;

        // Trace the volume body and compute the internal radiance reflected through the surface hit point
		for (unsigned int i = 0; i < steps; i++)
		{ 
			// Acquire interpolated sample intensity at current position
            float4 density = tex3D(volumeTexture, pos.x, pos.y, pos.z);
            
            // Compute modified transmission coefficient
            alpha *= transmission; // (transmission precomputed for rayStep)

			// Composite the sample radiance contribution
            radiance.x += density.x * alpha.x * 1.f; 
            radiance.y += density.y * alpha.y * 1.f; 
            radiance.z += density.z * alpha.z * 1.f; 
                // :TODO: Adjust light scaling/tonemapping
                // Also, actually    L{out} = density * transmission / sigma_absorption 
                // or if sigma = 0,  L{out} = density * rayStepSize 
                // (Verify these equations)

            // Early termination check
            if (alpha.x < 0.05 && 
                alpha.y < 0.05 &&
                alpha.z < 0.05
                    ) 
                { 
                    alpha.x = 0.0f;
                    alpha.y = 0.0f;
                    alpha.z = 0.0f;
                    break; 
                }

            // :TODO: Should possible recalc in case of small step size
            pos += step; // Advance position
		}

        // --------------------------------------------------------------------
        //  Compute the output radiance as a function of the volume trace info
        //  and the radiance incoming on the volumes opposite endpoint
        // --------------------------------------------------------------------
        payloadRadiance.color.l += radiance.x + alpha.x * reflPayload.color.l;
        payloadRadiance.color.a += radiance.y + alpha.y * reflPayload.color.a;
        payloadRadiance.color.b += radiance.z + alpha.z * reflPayload.color.b;
    }
}
