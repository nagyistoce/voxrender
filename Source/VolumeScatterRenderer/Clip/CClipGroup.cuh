/* ===========================================================================

    Project: Volume Scatter Renderer

    Description: Defines a grouping of other clipping elements

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
#include "VolumeScatterRenderer/Clip/CClipGroup.h"

// ----------------------------------------------------------------------------
//  Intersection callback function
// ----------------------------------------------------------------------------
VOX_DEVICE void CClipGroup_INTERSECT(void * dataPtr, vox::Ray3f & ray)
{
    auto & data = *reinterpret_cast<vox::CClipGroup::Data*>(dataPtr);

    size_t length = data.children.size();
    for (size_t i = 0; i < length; i++)
    {
        data.children[i]->clip(ray);
    }
}
VOX_DEVICE vox::ClipFunc symbol_CClipGroup_INTERSECT = CClipGroup_INTERSECT;

namespace vox {

// ----------------------------------------------------------------------------
//  Intersection callback pointer accessor 
// ----------------------------------------------------------------------------
ClipFunc RenderKernel::getCClipGroupFunc()
{
    ClipFunc result;
    
    VOX_CUDA_CHECK(cudaMemcpyFromSymbol(&result, symbol_CClipGroup_INTERSECT, sizeof(ClipFunc)));

    return result;
}

} // namespace vox