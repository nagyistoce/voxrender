/* ===========================================================================

	Project: VoxRender - CUDA Renderer OptiX Program

	Description: Implements the OptiX Programs for the CUDA renderer.

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

// Ray Type Definitions
#include "CudaRenderer/Optix/RayTypes.h"

// VoxRender Library Includes
#include "VoxLib/Core/Geometry/Color.h"
#include "VoxLib/Core/Geometry/Ray.h"

// CUDA/Device representations of scene components
#include "CudaRenderer/Core/CBuffer.h"
#include "CudaRenderer/Core/CRandomGenerator.h"
#include "CudaRenderer/Core/CSampleBuffer.h"
#include "CudaRenderer/Core/CRandomBuffer.h"
#include "CudaRenderer/Scene/CCamera.h"
#include "CudaRenderer/Scene/CVolume.h"

// Optix SDK Headers
#include <optix.h> // Core library header
#include <optixu/optixu_math_namespace.h>

using namespace optix; 
using namespace vox;
 
typedef CImgBuffer2D<ColorRgbaLdr> CLdrImgBuffer2D;

// Global variable declarations for OptiX rendering context
rtDeclareVariable(uint2,           launchIndex, rtLaunchIndex, ); ///< Thread launch indices
rtDeclareVariable(CCamera,         camera, ,                   ); ///< Device camera model
rtDeclareVariable(rtObject,        geometryRoot, ,             ); ///< Root scene object
rtDeclareVariable(CRandomBuffer2D, rndBuffer0, ,               ); ///< Device RNG seed buffer
rtDeclareVariable(CRandomBuffer2D, rndBuffer1, ,               ); ///< Device RNG seed buffer
rtDeclareVariable(CSampleBuffer2D, sampleBuffer, ,             ); ///< HDR sample data buffer
rtDeclareVariable(CLdrImgBuffer2D, imageBuffer, ,              ); ///< LDR processed image

// --------------------------------------------------------------------
//  Initializes the payload substructures and generates the optix::Ray
// --------------------------------------------------------------------
VOX_HOST_DEVICE optix::Ray initialize(RayPayloadRadiance & payload)
{
    // Initialize the payload
    payload.isInVolume = false;
    payload.importance = 1.0f;
    payload.color.l = 0.0f;
    payload.color.a = 0.0f;
    payload.color.b = 0.0f;
    payload.depth = 0;

    // Initialize the payload embedded RNG
    payload.rng.setSeeds(
        &rndBuffer0.at(launchIndex.x, launchIndex.y),
        &rndBuffer1.at(launchIndex.x, launchIndex.y));

    // Compute the sample ray parameters
    Vector2f uv = Vector2f(launchIndex.x, launchIndex.y) + payload.rng.sample2D();
    Ray3f ray = camera.generateRay(uv, payload.rng.sampleDisk());

    // Return the ray formatted for OptiX API usage
    return optix::Ray(
        make_float3(ray.pos[0], ray.pos[1], ray.pos[2]),
        make_float3(ray.dir[0], ray.dir[1], ray.dir[2]),
        0, 0.0f
        );
}

// --------------------------------------------------------------------
//  OptiX Program for handling rays which do not intersect geometry
// --------------------------------------------------------------------
RT_PROGRAM void missProgram()
{

}

// --------------------------------------------------------------------
//  OptiX Program for generating the trace rays for the scene
// --------------------------------------------------------------------
RT_PROGRAM void rayGenerationProgram()
{       
    RayPayloadRadiance payload;

    // Perform the optix ray-tracing subroutines
    optix::Ray ray = initialize(payload);
    rtTrace(geometryRoot, ray, payload);

    // Push the new sample value to the buffer
    sampleBuffer.push(launchIndex.x, launchIndex.y, payload.color);

    // :TEST:
    ColorLabxHdr const& mean   = sampleBuffer.at(launchIndex.x, launchIndex.y);
    ColorRgbaLdr &      target = imageBuffer.at(launchIndex.x, launchIndex.y); 
    target.r = vox::low(mean.l * 140.0f, 255.0f);
    target.g = vox::low(mean.a * 140.0f, 255.0f);
    target.b = vox::low(mean.b * 140.0f, 255.0f);
}

// --------------------------------------------------------------------
//  OptiX Program for handling exception during trace execution
// --------------------------------------------------------------------
RT_PROGRAM void exceptionProgram()
{
    const unsigned int code = rtGetExceptionCode();
    rtPrintf("Caught exception 0x%X at launch index (%d,%d)\n",
             code, launchIndex.x, launchIndex.y );
}