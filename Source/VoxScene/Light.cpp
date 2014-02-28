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

namespace vox {
    
// ------------------------------------------------------------
//  Initializes the light object
// ------------------------------------------------------------
Light::Light() :
    m_color(1.0f, 1.0f, 1.0f),
    m_position(0.0f, 0.0f, 0.0f)
{
}

// ------------------------------------------------------------
//  Sets the parent node for the light
// ------------------------------------------------------------
void Light::setParent(std::shared_ptr<LightSet> parent)
{
    m_parent = parent;
}

// ------------------------------------------------------------
//  Sets the dirty flag for the light and its parent node
// ------------------------------------------------------------
void Light::setDirty() 
{
    m_isDirty = true;

    if (m_parent) m_parent->setDirty();
}

// ------------------------------------------------------------
//  Shared ptr factor method for light set
// ------------------------------------------------------------
std::shared_ptr<LightSet> LightSet::create()
{
    return std::shared_ptr<LightSet>(new LightSet());
}

// ------------------------------------------------------------
//  Initializes a new light set to default parameters
// ------------------------------------------------------------
LightSet::LightSet() : 
    m_isDirty(false),
    m_ambientLight(0.1f, 0.1f, 0.1f)
{ 
}

// ------------------------------------------------------------
//  Adds a new light to the scene 
// ------------------------------------------------------------
std::shared_ptr<Light> LightSet::add()
{
    auto light = Light::create();
    m_lights.push_back(light);
    return light;
}

// ------------------------------------------------------------
//  Adds a new light to the scene 
// ------------------------------------------------------------
void LightSet::add(std::shared_ptr<Light> light)
{
    light->setParent(shared_from_this());
    m_lights.push_back(light);
}

// ------------------------------------------------------------
//  Removes an existing light from the scene
// ------------------------------------------------------------
void LightSet::remove(std::shared_ptr<Light> light)
{ 
    m_lights.remove(light);
    light->setParent(nullptr);
}

// ------------------------------------------------------------
//  Interpolates between lighting sets
// ------------------------------------------------------------
std::shared_ptr<LightSet> LightSet::interp(std::shared_ptr<LightSet> k2, float f)
{ 
    BOOST_FOREACH (auto & light, k2->lights())
    {
           
    }

    return k2;
}

} // namespace vox
