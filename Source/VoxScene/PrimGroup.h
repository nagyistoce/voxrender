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

// Internal Dependencies
#include "VoxScene/Common.h"
#include "VoxScene/Primitive.h"

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"

// API namespace
namespace vox
{
    /** Buffer class for scene lights */
    class VOXS_EXPORT PrimGroup : public Primitive, public std::enable_shared_from_this<PrimGroup>
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
        
        /** Exports the plane to a text format */
        static std::shared_ptr<PrimGroup> imprt(boost::property_tree::ptree & node);

        /** Exports the plane to a text format */
        virtual void exprt(boost::property_tree::ptree & node);

        /** Returns a child primitive of this group, if found */
        std::shared_ptr<Primitive> find(int id);

        /** Adds a new child element to the group */
        void add(std::shared_ptr<Primitive> child, bool suppress = false);

        /** Removes an existing child from the group */
        void remove(std::shared_ptr<Primitive> child, bool suppress = false);
        
        /** Sets the callback event for adding a primitive */
        void onAdd(std::function<void(std::shared_ptr<Primitive>, bool)> callback);

        /** Sets the callback event for removing a primitive */
        void onRemove(std::function<void(std::shared_ptr<Primitive>, bool)> callback);

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

        std::function<void(std::shared_ptr<Primitive>, bool)> m_addCallback;
        std::function<void(std::shared_ptr<Primitive>, bool)> m_removeCallback;
    };
}

// End Definition
#endif // VOX_PRIMITIVE_GROUP_H