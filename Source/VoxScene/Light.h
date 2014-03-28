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

// Internal Dependencies
#include "VoxScene/Common.h"
#include "VoxScene/Object.h"

// Include Dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Core/Geometry.h"

// API namespace
namespace vox
{
    class Light;
    class RenderController;

    /** Container class for scene lights */
    class VOXS_EXPORT LightSet : public std::enable_shared_from_this<LightSet>, public Object
    {
    public:
        /** Convenience factor for shared_ptr construction of a light set */
        static std::shared_ptr<LightSet> create();

        /** Sets the ambient lighting conditions */
        void setAmbientLight(Vector3f const& light) { m_ambientLight = light; }

        /** Returns the ambient lighting conditions */
        Vector3f const& ambientLight() { return m_ambientLight; }

        /** Interpolates between transfer function keyframes */
        std::shared_ptr<LightSet> interp(std::shared_ptr<LightSet> k2, float f);

        /** Constructs a new light object for the scene */
        std::shared_ptr<Light> add();

        /** Clones the light set */
        void clone(LightSet & lightSet);

        /** Adds a new light to the scene */
        void add(std::shared_ptr<Light> light, bool suppress = false);

        /** Removes an existing light from the scene */
        void remove(std::shared_ptr<Light> light, bool suppress = false);

        /** Sets the callback event for adding a light */
        void onAdd(std::function<void(std::shared_ptr<Light>, bool)> callback);

        /** Sets the callback event for removing a light */
        void onRemove(std::function<void(std::shared_ptr<Light>, bool)> callback);

        /** Locates a light element within the set */
        std::shared_ptr<Light> find(int id);

        /** Returns the internal light buffer */
        std::list<std::shared_ptr<Light>> const& lights() const { return m_lights;  }

        /** Clears all lights from the set */
        void clear() { m_lights.clear(); }

    private:
        friend RenderController;

        std::function<void(std::shared_ptr<Light>, bool)> m_addLightCallback;
        std::function<void(std::shared_ptr<Light>, bool)> m_remLightCallback;

        LightSet(); // Private constructor to ensure shared_from_this properly

        Vector3f m_ambientLight; ///< Ambient light

        std::list<std::shared_ptr<Light>> m_lights; ///< Scene lights
    };

    /** Scene light model */
    class VOXS_EXPORT Light : public SubObject
    {
    public:
        /** Convenience factor for shared_ptr construction of a light set */
        static std::shared_ptr<Light> create() { return std::shared_ptr<Light>(new Light()); }
        
        /** Constructor */
        Light();

        /** Clone method */
        void clone(Light & light);

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

    private:
        friend LightSet;

        Vector3f m_position; ///< Light position
        Vector3f m_color;    ///< Light color 
    };
}

// End Definition
#endif // VOX_LIGHT_H