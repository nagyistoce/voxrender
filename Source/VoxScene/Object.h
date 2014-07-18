/* ===========================================================================

	Project: VoxScene

	Description: Defines a basic identifiable scene element

    Copyright (C) 2012-2014 Lucas Sherman

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
#ifndef VOX_OBJECT_H
#define VOX_OBJECT_H

// Internal Dependencies
#include "VoxScene/Common.h"

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Geometry.h"

namespace boost { class mutex; }

// API namespace
namespace vox
{
    /** Base class for scene graph elements */
    class VOXS_EXPORT Object
    {
    public:
        /** Assigns an ID to the object */
        Object(int id = 0);

        /** Returns the ID of the object */
        int id() { return m_id; }

        /** Sets the ID of the object */
        void setId(int id) { m_id = id; }

        /** Sets the visibility of the object during rendering */
        void setVisible(bool visible = true, bool suppress = false) 
        { 
            m_isVisible = visible;

            if (m_visCallback) m_visCallback(visible, suppress);
        }

        /** Returns the visibility state of the object */
        bool isVisible() { return m_isVisible; }

        /** Returns true if the context change flag is set */
        bool isDirty() const { return m_isDirty; }

        /** Marks the object context as dirty */
        void setDirty() { m_isDirty = true; }

        /** Clears the object's dirty flag */
        void setClean() { m_isDirty = false; }

        /** Sets the visibility change event callback */
        void onVisibilityChanged(std::function<void(bool, bool)> callback) 
        { 
            m_visCallback = callback; 
        }

    protected:
        int m_id; ///< ID of the object

        std::function<void(bool, bool)> m_visCallback; ///< Visibility callback

        bool m_isDirty;     ///< Dirty flag for tracking changes
        bool m_isVisible;   ///< Visibility state of the object
    };

    /** Derived object which locks through a parent */
    class VOXS_EXPORT SubObject : public Object
    {
    public:
        /** SubObject constructor */
        SubObject(int id = 0) : Object(id) { }

        /** Marks both the object and the parent dirty */
        virtual void setDirty() { if (m_parent) m_parent->setDirty(); Object::setDirty(); }

        /** Sets the parent of this object (Should be called by the parent) */
        void setParent(std::shared_ptr<Object> parent) { m_parent = parent; }

    protected:
        std::shared_ptr<Object> m_parent;
    };
}

// End Definition
#endif // VOX_OBJECT_H