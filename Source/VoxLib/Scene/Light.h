/* ===========================================================================

	Project: VoxRender - Light

	Description: Defines a basic light for use in the scene.

    Copyright (C) 2012 Lucas Sherman

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
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Geometry.h"

// API namespace
namespace vox
{
    class Light;
    class RenderController;

    /** Buffer class for scene lights */
    class VOX_EXPORT LightSet : public std::enable_shared_from_this<LightSet>
    {
    public:
        /** Ensure initial context change */
        LightSet() : m_contextChanged(true) { }

        /** Sets the ambient lighting conditions */
        void setAmbientLight(ColorLabHdr const& light)
        {
            m_ambientLight = light;

            m_ambientChanged = true;
        }

        /** Returns the ambient lighting conditions */
        ColorLabHdr const& ambientLight()
        {
            return m_ambientLight;
        }

        /** Adds a new light to the scene */
        std::shared_ptr<Light> addLight();

        /** Removes an existing light from the scene */
        void removeLight(std::shared_ptr<Light> light);

        /** Returns the internal light buffer */
        std::list<std::shared_ptr<Light>> const& lights() const 
        { 
            return m_lights; 
        }

        /** Clears all lights from the set */
        void clear() 
        { 
            m_lights.clear(); 

            m_contextChanged = true; 
        }

        /** Returns true if the context change flag is set */
        bool isDirty() const { return m_contextChanged || m_contentChanged; }

        /** Returns true if the internal content is dirty */
        bool isContentDirty() const
        {
            return m_contextChanged ? true : m_contentChanged;
        }

        /** Returns true if the ambient lighting is dirty */
        bool isAmbientDirty() const { return m_ambientChanged; }

    private:
        friend RenderController;
        friend Light;

        bool m_contextChanged;
        bool m_contentChanged;
        bool m_ambientChanged;

        ColorLabHdr m_ambientLight;

        std::list<std::shared_ptr<Light>> m_lights;
    };

// Context change set
#define VOX_CC m_parent->m_contentChanged = true; m_contextChanged = true;

    /** Scene light model */
    class VOX_EXPORT Light
    {
    public:
        /** Light position accessor */
        Vector3f const& position() const { return m_position; }
        float positionX() const { return m_position[0]; }
        float positionY() const { return m_position[1]; }
        float positionZ() const { return m_position[2]; }
        
        /** Light color accessor */
        ColorLabHdr const& color() const { return m_color; }

        /** Light position modifiers */
        void setPosition(Vector3f const& position) { m_position = position; VOX_CC }
        void setPositionX(float pos) { m_position[0] = pos; VOX_CC }
        void setPositionY(float pos) { m_position[1] = pos; VOX_CC }
        void setPositionZ(float pos) { m_position[2] = pos; VOX_CC }

        /** Light color modifier */
        void setColor(ColorLabHdr const& color) { m_color = color; VOX_CC }

    private:
        friend LightSet;

        Light(std::shared_ptr<LightSet> parent);

        bool m_contextChanged;

        Vector3f     m_position;  ///< Light position
        ColorLabHdr  m_color;     ///< Light color 

        std::shared_ptr<LightSet> m_parent; ///< Parent light set
    };
    
#undef VOX_CC
}

// End Definition
#endif // VOX_LIGHT_H