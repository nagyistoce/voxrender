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

// Begin definition
#ifndef VSR_CGEOMETRY_H
#define VSR_CGEOMETRY_H

// Include Dependencies
#include "VolumeScatterRenderer/Core/Common.h"
#include "VoxLib/Core/Geometry/Ray.h"

// API namespace
namespace vox {

class Primitive;

/** Format of clipping function */
typedef void(*ClipFunc)(void *, Ray3f &);

/** Generic clipping geometry class for the device */
class CClipGeometry
{
public:    
    /** Device side structure for performing clipping operations */
    class Clipper
    {
    public:
        /** Calls the geometry element's appropriate clip function */
        VOX_DEVICE inline void clip(Ray3f & ray) { (*func)(this, ray); }

        ClipFunc func; ///< Device pointer to clip function
    };

public:
    /** Creates a usable clipping object for the render kernel from a primitive element */
    static std::shared_ptr<CClipGeometry> create(std::shared_ptr<Primitive> primitive);

    /** Prerequisite virtualized destructor */
    virtual ~CClipGeometry() { }

    /** Returns the clipper associated with this element */
    virtual Clipper * clipper() = 0;
};

} // namespace vox

// End definition
#endif // VSR_CGEOMETRY_H