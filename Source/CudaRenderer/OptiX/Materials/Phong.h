
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include <optix_world.h>

// Optix SDK Headers
#include <optix.h> // Core library header
#include <optixu/optixu_math_namespace.h>

#include "CudaRenderer/Core/CBuffer.h"
#include "CudaRenderer/Scene/CLight.h"
#include "CudaRenderer/OptiX/RayTypes.h"

using namespace optix;
using namespace vox;

rtDeclareVariable(unsigned int, maxDepth, ,        ); ///< Max ray tree evaluation depth
rtDeclareVariable(float,        sceneEpsilon, ,    ); ///< Epsilon distance for intersect offset
rtDeclareVariable(unsigned int, radianceRayType, , ); ///< Type ID for radiance rays
rtDeclareVariable(unsigned int, shadowRayType, ,   ); ///< Type ID for shadow rays
rtDeclareVariable(rtObject,     geometryRoot, ,    ); ///< Root scene object

rtDeclareVariable(optix::Ray,         ray,             rtCurrentRay,           );
rtDeclareVariable(float,              t_hit,           rtIntersectionDistance, );
rtDeclareVariable(RayPayloadRadiance, payloadRadiance, rtPayload,              );
rtDeclareVariable(RayPayloadShadow,   payloadShadow,   rtPayload,              );
rtDeclareVariable(CBuffer1D<CLight>,  lightBuffer, ,                           );
rtDeclareVariable(ColorLabHdr,        ambientLight, ,                          );

// --------------------------------------------------------------------
//  Shadow program for phong shading: Fully attenuate all rays
// --------------------------------------------------------------------
__device__ void phongShadowed()
{
    payloadShadow.attenuation = Vector3f(0.0f);

    rtTerminateRay();
}

// --------------------------------------------------------------------
//  Radiance program for phong shading: 
//      Compute direct reflection + shadow
// --------------------------------------------------------------------
__device__ void phongShade(float3 p_Kd,
                           float3 p_Ka,
                           float3 p_Ks,
                           float3 p_normal,
                           float  p_phong_exp,
                           float3 p_reflectivity)
{
    // Compute the 3d intersection position for this pass
    float3 hit_point = ray.origin + t_hit * ray.direction;

    float3 result = p_Ka * make_float3(0.25f); // Ambient contribution :TODO:

    // Compute this points direct lighting component
    unsigned int nLights = lightBuffer.size();
    for (unsigned int i = 0; i < nLights; i++)
    {
        CLight const& light = lightBuffer[i];

        float3 pos = make_float3(light.position()[0],
                                 light.position()[1],
                                 light.position()[2]);
        float  len = optix::length(pos - hit_point);
        float3 dir = optix::normalize(pos - hit_point);
        float  nDl = optix::dot(p_normal, dir);

        // Compute the attenuation for this light due to shadowing
        float3 lightAttenuation = make_float3(static_cast<float>(nDl > 0.0f));
        if (nDl > 0.0f) 
        {
            // Construct the ray for shadow attenuation computation
            RayPayloadShadow shadowResult;
            shadowResult.attenuation = Vector3f(1.0f);
            optix::Ray shadowRay = optix::make_Ray(
                hit_point, dir, shadowRayType, sceneEpsilon, len);

            // Perform the shadow ray tace operation
            rtTrace(geometryRoot, shadowRay, shadowResult);
            lightAttenuation.x = shadowResult.attenuation[0];
            lightAttenuation.y = shadowResult.attenuation[1];
            lightAttenuation.z = shadowResult.attenuation[2];
        }

        // Compute the radiance transfer along the shadow ray
        if (fmaxf(lightAttenuation) > 0.0f) 
        {
            float3 Lc;
            Lc.x = light.color().l * lightAttenuation.x;
            Lc.y = light.color().a * lightAttenuation.y;
            Lc.z = light.color().b * lightAttenuation.z;

            result += p_Kd * nDl * Lc;

            float3 H  = optix::normalize(dir - ray.direction);
            float nDh = optix::dot( p_normal, H );
            if (nDh > 0) 
            {
                float power = pow(nDh, p_phong_exp);
                result += p_Ks * power * Lc;
            }
        }
    }

    // Compute the in-reflected radiance component
    if (fmaxf(p_reflectivity) > 0) 
    {
        // Initialize the reflection ray
        RayPayloadRadiance reflPayload;             
        reflPayload.importance = payloadRadiance.importance * optix::luminance(p_reflectivity);
        reflPayload.depth      = payloadRadiance.depth + 1;
        reflPayload.isInVolume = payloadRadiance.isInVolume;
        
        // Evaluate the reflectance ray if it is of significant importance
        if( reflPayload.importance >= 0.01f && reflPayload.depth <= maxDepth) 
        { 
            // Trace the reflectance ray and compute radiance contribution
            float3 R = optix::reflect(ray.direction, p_normal);
            optix::Ray reflRay = optix::make_Ray(hit_point, R, radianceRayType, sceneEpsilon, RT_DEFAULT_MAX);
            rtTrace(geometryRoot, reflRay, reflPayload);
            
            // Add the reflectance contribution
            result += p_reflectivity * 
                        make_float3(
                        reflPayload.color.l, 
                        reflPayload.color.a, 
                        reflPayload.color.b);
        }
    }

    // Provide the computed color value to the parent node 
    payloadRadiance.color.l = result.x;
    payloadRadiance.color.a = result.y;
    payloadRadiance.color.b = result.z;
}
