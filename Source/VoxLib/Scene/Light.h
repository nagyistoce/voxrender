/* ===========================================================================

	Project: VoxScene

	Description: Defines a basic light for use in the scene.

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
#ifndef VOX_LIGHT_H
#define VOX_LIGHT_H

// Include Dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Core/Geometry.h"

// API namespace
namespace vox
{
    class Light;
    class RenderController;

    /** Container class for scene lights */
    class VOX_EXPORT LightSet : public std::enable_shared_from_this<LightSet>
    {
    public:
        /** Convenience factor for shared_ptr construction of a light set */
        static std::shared_ptr<LightSet> create();

        /** Sets the ambient lighting conditions */
        void setAmbientLight(Vector3f const& light) { m_ambientLight = light; }

        /** Returns the ambient lighting conditions */
        Vector3f const& ambientLight() { return m_ambientLight; }

        /** Constructs a new light object for the scene */
        std::shared_ptr<Light> add();

        /** Locks the light set for read/write operations */
        void lock() { m_mutex.lock(); }
        
        /** Unlocks the light set */
        void unlock() { m_mutex.unlock(); }

        /** Adds a new light to the scene */
        void add(std::shared_ptr<Light> light);

        /** Removes an existing light from the scene */
        void remove(std::shared_ptr<Light> light);

        /** Returns the internal light buffer */
        std::list<std::shared_ptr<Light>> const& lights() const { return m_lights;  }

        /** Clears all lights from the set */
        void clear() { m_lights.clear(); }

        /** Returns true if the context change flag is set */
        bool isDirty() const { return m_isDirty; }

        /** Marks the light set as dirty */
        void setDirty() { m_isDirty = true; }

    private:
        /** Marks the light set as synchronized with the render controller */
        void setClean() { m_isDirty = false; }

    private:
        friend RenderController;

        LightSet(); // Private constructor to ensure shared_from_this properly

        Vector3f     m_ambientLight; ///< Ambient light
        boost::mutex m_mutex;        ///< Mutex for scene locking
        bool         m_isDirty;      ///< Dirty flag for tracking scene changes

        std::list<std::shared_ptr<Light>> m_lights; ///< Scene lights
    };

    /** Scene light model */
    class VOX_EXPORT Light
    {
    public:
        /** Convenience factor for shared_ptr construction of a light set */
        static std::shared_ptr<Light> create() { return std::shared_ptr<Light>(new Light()); }
        
        /** Constructor */
        Light();

        /** Locks the light for read/write operations */
        void lock() { if (m_parent) m_parent->lock(); }
        
        /** Unlocks the light */
        void unlock() { if (m_parent) m_parent->unlock(); }

        /** Light position accessor */
        Vector3f const& position() const { return m_position; }
        float positionX() const { return m_position[0]; }
        float positionY() const { return m_position[1]; }
        float positionZ() const { return m_position[2]; }
        
        /** Light color accessor */
        Vector3f const& color() const { return m_color; }

        /** Light position modifiers */
        void setPosition(Vector3f const& position) { m_position = position; }
        void setPositionX(float pos) { m_position[0] = pos; }
        void setPositionY(float pos) { m_position[1] = pos; }
        void setPositionZ(float pos) { m_position[2] = pos; }

        /** Light color modifier */
        void setColor(Vector3f const& color) { m_color = color; }

        /** Returns the dirty flag for the light */
        bool isDirty() { return m_isDirty; }

        /** Sets the dirty flag for the light */
        void setDirty();

    private:
        /** Sets the parent light set for this light */
        void setParent(std::shared_ptr<LightSet> parent);

        /** Marks the light as synchronized with the render */
        void setClean() { m_isDirty = true; }

    private:
        friend LightSet;

        bool     m_isDirty;  ///< Flag for tracking scene changes
        Vector3f m_position; ///< Light position
        Vector3f m_color;    ///< Light color 

        std::shared_ptr<LightSet> m_parent; ///< Parent light set
    };
}

// End Definition
#endif // VOX_LIGHT_H