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

rtDeclareVariable(float,  rayStepSize, , );   // Step size of the volume trace ray
rtDeclareVariable(float,  reflectFactor, , ); // Percent specular reflectivity of the surface 
rtDeclareVariable(float,  diffuseFactor, , ); // Percent diffuse reflectivity of the surface
rtDeclareVariable(float3, volumeSpacing, , ); // Spacing between volume samples

rtDeclareVariable(float3, texcoord,         attribute texcoord,         ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal,   attribute shading_normal,   ); 

rtTextureSampler<UInt8,3> volumeTexture;
rtDeclareVariable(uint3, volumeExtent, , );

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
        payloadRadiance.color.x = t_hit; // :TODO: Use LAB and rename to closestHit

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
            
        payloadRadiance.color.l = reflPayload.color.l * 0.2f;
        payloadRadiance.color.a = reflPayload.color.a * 0.2f;
        payloadRadiance.color.b = reflPayload.color.b * 0.2f;
        
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
        
        unsigned int steps = reflPayload.color.x / 0.25f; // tmax / stepSize

        // Compute volume-space trace ray     
        pos -= make_float3(16.0f, 16.0f, 16.0f);  // Transform to object space :TODO:
        float3 volSize = make_float3(volumeExtent.x, volumeExtent.y, volumeExtent.z);
        float3 scale = volSize * 1 / 64.0f;             // Scale to volume space, Volume_Size / Mesh_AABox
        pos.x *= scale.x; 
        pos.y *= scale.y; 
        pos.z *= scale.z;

        bool hit = false;
        for (unsigned int i = 0; i < steps; i++)
        {
            pos += ray.direction * 0.25f;
            float sample = tex3D(volumeTexture, pos.x, pos.y, pos.z);
            if (sample > 100)
            {
                hit = true;
                break;
            }
        }

        // --------------------------------------------------------------------
        //  Compute the output radiance as a function of the volume trace info
        //  and the radiance incoming on the volumes opposite endpoint
        // --------------------------------------------------------------------
        if (hit)
        {
            payloadRadiance.color.l += 0.0f;
            payloadRadiance.color.a += 2.0f;
            payloadRadiance.color.b += 0.0f;
        }
        else
        {
            payloadRadiance.color.l += reflPayload.color.l * 1.0f;
            payloadRadiance.color.a += reflPayload.color.a * 1.0f;
            payloadRadiance.color.b += reflPayload.color.b * 1.0f;
        }
    }
}
