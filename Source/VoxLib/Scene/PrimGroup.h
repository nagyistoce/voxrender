/* ===========================================================================

	Project: VoxRender - PrimitiveGroup

	Description: Defines a collection of primitive elements

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
#ifndef VOX_PRIMITIVE_GROUP_H
#define VOX_PRIMITIVE_GROUP_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Geometry/Primitives.h"

// API namespace
namespace vox
{
    /** Buffer class for scene lights */
    class VOX_EXPORT PrimGroup : Primitive
    {
    public:
        /** Ensure initial context change */
        PrimGroup() { }
        
        ~PrimGroup() {}

        /** Returns the UID string classifying this type (classname) */
        virtual Char const* typeId() { return "PrimGroup"; }

        /** Adds a new child element to the group */
        void add(std::shared_ptr<Primitive> child);

        /** Removes an existing child from the group */
        void remove(std::shared_ptr<Primitive> child);

        /** Clears all child elements from the group */
        void clear() { m_children.clear(); setDirty(); }

    private:
        std::map<String,std::shared_ptr<Primitive>> m_children;
    };
}

// End Definition
#endif // VOX_PRIMITIVE_GROUP_H