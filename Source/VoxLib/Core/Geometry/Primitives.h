/* ===========================================================================

	Project: VoxLib

	Description: Defines the basic primitive object element

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

// :TODO: This should be in the Scene library

// Begin definition
#ifndef VOX_PRIMITIVES_H
#define VOX_PRIMITIVES_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Geometry/Vector.h"
#include "VoxLib/IO/OptionSet.h"

// API namespace
namespace vox
{
    /** Primitive geometry object */
    class Primitive
    {
    public:
        /** Prerequisite virtualized destructor for inheritance */
        virtual ~Primitive() { }

        /** Returns the type identifier for this primitive */
        // :TODO: Register with static member for int
        virtual Char const* typeId() = 0;

        /** Returns the UID string for this primitive */
        String const& id() { return m_id; }

        /** Returns true if the primitive is visible */
        bool isVisible() { return m_visible; }

        /** Sets the visibility status of the primitive */
        void setVisible(bool visible = true)
        {
            if (m_visible == visible) return;

            m_visible = visible; 
            m_dirty = true;
        }

        /** Returns true if the geometry should be updated */
        bool isDirty() { return m_dirty; }

        /** Notifies the object that it's geometry should be updated */
        void setDirty(bool dirty = true) { m_dirty = dirty; }

        /** */

    private:
        bool m_visible;   ///< The visibility status of this primitive
        bool m_dirty;     ///< The update flag for this primitive

    protected:
        String m_id;
    };

	/** Plane structure */
	class Plane : public Primitive
	{
    public:
        Plane() :
          normal(0.0f, 1.0f, 0.0f),
          position(0.0f, 0.0f, 0.0f)
        {
        }

        /** Returns the UID string classifying this type (classname) */
        virtual Char const* typeId() { return "Plane"; }

		Vector3f normal;   ///< Normal vector of the plane
        Vector3f position; ///< Position on the plane
	};

    /** Sphere structure */
	class Sphere : public Primitive
	{
    public:
        Sphere() :
          origin(0.0f, 0.0f, 0.0f),
          radius(50.0f),
          inside(true)
        {
        }

        /** Returns the UID string classifying this type (classname) */
        virtual Char const* typeId() { return "Sphere"; }

		Vector3f origin; ///< Origin of the sphere
        float    radius; ///< Radius of the sphere
        bool     inside; ///< Inner or outer is solid
	};
}

// End Definition
#endif // VOX_PRIMITIVES_H