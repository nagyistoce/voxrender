/* ===========================================================================

    Project: Volume Scatter Renderer

    Description: Defines a clipping plane element

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
#include "VolumeScatterRenderer/Clip/CClipPlane.h"

// ----------------------------------------------------------------------------
//  Intersection callback function
// ----------------------------------------------------------------------------
VOX_DEVICE void CClipPlane_INTERSECT(void * dataPtr, vox::Ray3f & ray)
{
    auto & data = *reinterpret_cast<vox::CClipPlane::Data*>(dataPtr);

    float dt = vox::Vector3f::dot(ray.dir, data.normal);
    float t  = (data.distance - vox::Vector3f::dot(ray.pos, data.normal)) / dt;

    // Cull outwards from normal
    if (dt < 0)
    {
        if (t > ray.min) ray.min = t;
    }
    else
    {
        if (t < ray.max) ray.max = t;
    }
}
VOX_DEVICE vox::ClipFunc symbol_CClipPlane_INTERSECT = CClipPlane_INTERSECT;

namespace vox {

// ----------------------------------------------------------------------------
//  Intersection callback pointer accessor 
// ----------------------------------------------------------------------------
ClipFunc RenderKernel::getCClipPlaneFunc()
{
    ClipFunc result;
    
    VOX_CUDA_CHECK(cudaMemcpyFromSymbol(&result, symbol_CClipPlane_INTERSECT, sizeof(ClipFunc)));

    return result;
}

} // namespace vox