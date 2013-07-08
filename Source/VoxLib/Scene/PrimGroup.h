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
#include "VoxLib/Scene/Primitive.h"

// API namespace
namespace vox
{
    /** Buffer class for scene lights */
    class VOX_EXPORT PrimGroup : public Primitive, public std::enable_shared_from_this<PrimGroup>
    {
    public:
        /** Ensure initial context change */
        static std::shared_ptr<PrimGroup> create()
        {
            return std::shared_ptr<PrimGroup>(new PrimGroup());
        }

        /** Returns the UID string classifying this type (classname) */
        virtual Char const* typeId();
        
        /** Returns the UID string classifying this type (classname) */
        static Char const* classTypeId();

        /** Adds a new child element to the group */
        void add(std::shared_ptr<Primitive> child);

        /** Removes an existing child from the group */
        void remove(std::shared_ptr<Primitive> child);

        /** Clears all child elements from the group */
        void clear();

        /** Returns a list of the current child nodes */
        std::list<std::shared_ptr<Primitive>> children() 
        {
            return m_children;
        }

    private:
        PrimGroup() { }

        std::list<std::shared_ptr<Primitive>> m_children;
    };
}

// End Definition
#endif // VOX_PRIMITIVE_GROUP_H