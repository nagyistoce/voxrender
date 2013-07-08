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

// Begin definition
#ifndef VSR_CCLIP_PLANE_H
#define VSR_CCLIP_PLANE_H

// Include Dependencies
#include "VolumeScatterRenderer/Core/Common.h"
#include "VolumeScatterRenderer/Scene/CClipGeometry.h"
#include "VoxLib/Core/Geometry/Ray.h"
#include "VoxLib/Scene/Primitive.h"

// API namespace
namespace vox {

class Primitive;

/** Generic clipping geometry class for the device */
class CClipPlane : public CClipGeometry
{
public:
    struct Data
    {
        Clipper  clipper;   ///< Clipping object
        Vector3f normal;    ///< Plane normal
        float    distance;  ///< Distance from origin
    };

public:
    /** Constructs a new clipping plane geometry element */
    CClipPlane(std::shared_ptr<Plane> plane);

    /** Frees the device memory for the plane clipper */
    ~CClipPlane()
    {
        if (m_handle) cudaFree(m_handle);
    }

    /** Returns the clipper associated with this element */
    virtual Clipper * clipper();

private:
    std::shared_ptr<Plane> m_plane;

    Data * m_handle;
};

} // namespace vox

// End definition
#endif // VSR_CCLIP_PLANE_H