/* ===========================================================================

    Project: Volume Scatter Renderer

    Description: Defines a collection of geometric scene elements

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
#include "CClipGeometry.h"

// Include Dependencies
#include "VolumeScatterRenderer/Clip/CClipGroup.h"
#include "VolumeScatterRenderer/Clip/CClipPlane.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Scene/Primitive.h"
#include "VoxLib/Scene/PrimGroup.h"

namespace vox {

// ----------------------------------------------------------------------------
//  Generates a CUDA compatible clipping object from the given handle
// ----------------------------------------------------------------------------
std::shared_ptr<CClipGeometry> CClipGeometry::create(std::shared_ptr<Primitive> primitive)
{
    if (!primitive) return nullptr; // No primitive => no result

    if (primitive->typeId() == PrimGroup::classTypeId())
    {
        auto primGroup = std::dynamic_pointer_cast<PrimGroup>(primitive);
        if (primGroup->children().empty()) return nullptr;
        return std::shared_ptr<CClipGroup>(new CClipGroup(primGroup));
    }
    else if (primitive->typeId() == Plane::classTypeId())
    {
        auto plane = std::dynamic_pointer_cast<Plane>(primitive);
        return std::shared_ptr<CClipPlane>(new CClipPlane(plane));
    }
    else // Unrecognized / non-implemented or version mismatch
    {
        VOX_LOG_ERROR(Error_NotImplemented, VSR_LOG_CATEGORY, 
            format("Ignoring unrecognized clipping primitive type: %1%", primitive->typeId()));
    }

    return nullptr;
}

} // namespace vox