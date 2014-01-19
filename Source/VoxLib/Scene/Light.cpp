/* ===========================================================================

	Project: VoxRender - Light

	Description: Defines a light for placement in the scene.

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

// Include Header
#include "Light.h"

// API Namespace
namespace vox {

// ------------------------------------------------------------
//  Sets the display image for the render view
// ------------------------------------------------------------
Light::Light(std::shared_ptr<LightSet> parent) :
    m_position(0.0f, 0.0f, 0.0f),
    m_color(255, 255, 255), 
    m_parent(parent)
{
}

// ------------------------------------------------------------
//  Adds a new light to the scene 
// ------------------------------------------------------------
std::shared_ptr<Light> LightSet::addLight()
{
    auto light = std::shared_ptr<Light>(new Light(shared_from_this()));

    m_lights.push_back(light);

    m_contextChanged = true;

    return light;
}

// ------------------------------------------------------------
//  Adds a new light to the scene 
// ------------------------------------------------------------
void LightSet::addLight(std::shared_ptr<Light> light)
{
    m_lights.push_back(light);

    m_contextChanged = true;
}

// ------------------------------------------------------------
//  Removes an existing light from the scene
// ------------------------------------------------------------
void LightSet::removeLight(std::shared_ptr<Light> light)
{ 
    m_lights.remove(light);

    m_contextChanged = true; 
}

} // namespace vox
