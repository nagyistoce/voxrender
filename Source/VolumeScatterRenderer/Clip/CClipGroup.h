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

// Begin definition
#ifndef VSR_CCLIP_GROUP_H
#define VSR_CCLIP_GROUP_H

// Include Dependencies
#include "VolumeScatterRenderer/Core/Common.h"
#include "VolumeScatterRenderer/Scene/CClipGeometry.h"
#include "VoxLib/Core/Geometry/Ray.h"
#include "VoxLib/Scene/PrimGroup.h"
#include "VolumeScatterRenderer/Core/CBuffer.h"

// API namespace
namespace vox {

class Primitive;

/** Generic clipping geometry class for the device */
class CClipGroup : public CClipGeometry
{
public:
    /** Data structure for clip group */
    struct Data
    {
        Clipper             clipper;
        CBuffer1D<Clipper*> children;
    };

public:
    /** Constructs a clip group associated with the group */
    CClipGroup(std::shared_ptr<PrimGroup> group);

    /** Frees the device memory associated with this group */
    ~CClipGroup()
    {
        m_buffer.reset();

        if (m_handle) cudaFree(m_handle);
    }

    /** Returns a clipper object for the current device */
    virtual Clipper * clipper();

private:
    std::shared_ptr<PrimGroup> m_primGroup;

    std::list<std::shared_ptr<CClipGeometry>> m_children;

    Data * m_handle;

    CBuffer1D<Clipper*> m_buffer;
};

} // namespace vox

// End definition
#endif // VSR_CCLIP_GROUP_H