/* ===========================================================================

	Project: VoxRender - Light

	Description: Defines a light for placement in the scene.

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
#ifndef VSR_CLIGHT_H
#define VSR_CLIGHT_H

// Common Library Header
#include "VolumeScatterRenderer/Core/Common.h"

// VoxLib Dependencies
#include "VoxLib/Core/Geometry/Color.h"
#include "VoxLib/Core/Geometry/Vector.h"
#include "VoxLib/Scene/Light.h"

// API namespace
namespace vox
{
    /** CUDA/Device light model */
    class CLight
    {
    public:
        VOX_HOST_DEVICE CLight() { }
        
        /** Constructs a device light for the specified light */
        VOX_HOST CLight(Light & light) : 
            m_position(light.position())
        {
            m_color = light.color() * light.intensity();
        }

        /** Light position accessor */
        VOX_HOST_DEVICE Vector3f const& position() const { return m_position; }
        
        /** Light color accessor */
        VOX_HOST_DEVICE Vector3f const& color() const { return m_color; }

    private:
        Vector3f  m_position;  ///< Light position
        float     m_packing1;  ///< Alignment component
        Vector3f  m_color;     ///< Light color
        float     m_packing2;  ///< Alignment component
    };
}

// End Definition
#endif // VSR_CLIGHT_H