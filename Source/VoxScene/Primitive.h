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

// Begin definition
#ifndef VOX_PRIMITIVES_H
#define VOX_PRIMITIVES_H

// Internal Dependencies
#include "VoxScene/Common.h"
#include "VoxScene/Object.h"

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Geometry/Vector.h"
#include "VoxLib/IO/OptionSet.h"

#include "boost/property_tree/ptree_fwd.hpp"

// API namespace
namespace vox
{
    class VOXS_EXPORT PrimGroup;

    class VOXS_EXPORT Primitive;

    /**  */
    typedef std::function<std::shared_ptr<Primitive>(boost::property_tree::ptree &)> PrimImporter;

    /** Primitive geometry object */
    class VOXS_EXPORT Primitive : public SubObject
    {
    public:
        /** Prerequisite virtualized destructor for inheritance */
        virtual ~Primitive() {}

        /** Returns the type identifier for this primitive */
        virtual Char const* typeId() = 0;

        /** Clones a primitive used for rendering */
        virtual std::shared_ptr<Primitive> clone() = 0;

        /** Interpolates between states of a primitive */
        virtual std::shared_ptr<Primitive> interp(std::shared_ptr<Primitive> k2, float factor) = 0;

        /**
         * Exports a primitive to a text format
         *
         * @param primitive The primitive to export
         */
        virtual void exprt(boost::property_tree::ptree & tree);
        
        /** Registers a primitive import module */
        static std::shared_ptr<Primitive> imprt(String const& type, boost::property_tree::ptree & node);

        /** Registers a primitive import module */
        static void registerImportModule(String const& type, PrimImporter importer);

        /** Removes a registered import module */
        static void removeImportModule(String const& type);

        /** Returns the UID string for this primitive */
        String const& idString() { return m_id; }

    protected:
        Primitive() {}

        String m_id;
    };

	/** Plane structure */
    class VOXS_EXPORT Plane : public Primitive
	{
    public:
        /** Factory method for plane primitive */
        static std::shared_ptr<Plane> create(
            Vector3f const& normal   = Vector3f(0.0f, 1.0f, 0.0f),
            float           distance = 0.0f
            ) 
        { 
            return std::shared_ptr<Plane>(new Plane(normal, distance));
        }

        /** Returns the UID string classifying this type (classname) */
        virtual Char const* typeId();
 
        /** Clones a copy of the clipping plane */
        virtual std::shared_ptr<Primitive> clone();
        
        /** Interpolates between states of a primitive */
        std::shared_ptr<Primitive> interp(std::shared_ptr<Primitive> k2, float factor);

        /** Exports the plane to a text format */
        static std::shared_ptr<Primitive> imprt(boost::property_tree::ptree & node);

        /** Exports the plane to a text format */
        virtual void exprt(boost::property_tree::ptree & node);

        /** Returns the UID string classifying this type (classname) */              
        static Char const* classTypeId();

        /** Returns the current normal vector for this plane */
        Vector3f const& normal() { return m_normal; }

        /** Sets a new normal vector for this plane */
        void setNormal(Vector3f const& normal) { m_normal = normal; }

        /** Returns the current position vector for this plane */
        float distance() { return m_distance; }

        /** Sets a new position vector for this plane */
        void setDistance(float distance) { m_distance = distance; }

    private:
        Plane(Vector3f const& normal, float distance) :
          m_normal(normal), m_distance(distance)
        {
        }

        Vector3f m_normal;   ///< Normal vector of the plane
        float    m_distance; ///< Normal distance from origin
	};

    /** Sphere structure */
	class VOXS_EXPORT Sphere : public Primitive
	{
    public:
        /** Returns the UID string classifying this type (classname) */
        virtual Char const* typeId();
 
        /** Returns the UID string classifying this type (classname) */              
        static Char const* classTypeId();

    private:
        Sphere() :
          m_origin(0.0f, 0.0f, 0.0f),
          m_radius(50.0f),
          m_inside(true)
        {
        }

        Vector3f m_origin; ///< Origin of the sphere
        float    m_radius; ///< Radius of the sphere
        bool     m_inside; ///< Inner or outer is solid
	};
}

// End Definition
#endif // VOX_PRIMITIVES_H