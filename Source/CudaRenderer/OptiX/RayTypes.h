/* ===========================================================================

	Project: VoxRender - CUDA Renderer OptiX Program

	Description: Defines the ray payload structures for the Whitted model.

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

// Begin definition
#ifndef CR_RAY_TYPES_H
#define CR_RAY_TYPES_H

// Random number generator for device generation
#include "CudaRenderer/Core/CRandomGenerator.h"

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Geometry/Color.h"

// Optix SDK Headers
#include <optix.h> // Core library header

// API namespace
namespace vox
{

// Ray payload structure for shadow rays
struct RayPayloadShadow
{
    Vector3f attenuation;
};

// Ray payload structure for standard ray
struct RayPayloadRadiance
{
    CRandomGenerator rng;   ///< This ray's RNG
    ColorLabxHdr     color; ///< Accumulated variance
    unsigned int     depth; ///< Traversal depth

    bool isInVolume; ///< Tracks ray position 

    float importance; ///< Importance sampling weight
};

}

// End Definition
#endif // CR_RAY_TYPES_H