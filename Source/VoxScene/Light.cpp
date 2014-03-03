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
    
// ----------------------------------------------------------------------------
//  Initializes the light object
// ----------------------------------------------------------------------------
Light::Light() :
    m_color(1.0f, 1.0f, 1.0f),
    m_position(0.0f, 0.0f, 0.0f)
{
}

// ----------------------------------------------------------------------------
//  Shared ptr factor method for light set
// ----------------------------------------------------------------------------
std::shared_ptr<LightSet> LightSet::create()
{
    return std::shared_ptr<LightSet>(new LightSet());
}

// ----------------------------------------------------------------------------
//  Initializes a new light set to default parameters
// ----------------------------------------------------------------------------
LightSet::LightSet() : m_ambientLight(0.1f, 0.1f, 0.1f)
{ 
}

// ----------------------------------------------------------------------------
//  Performs a deep copy of the light set and its elements
// ----------------------------------------------------------------------------
void LightSet::clone(LightSet & lightSet)
{
    lightSet.setId(id());

    BOOST_FOREACH (auto & light, m_lights)
    {
        auto clone = Light::create();
        lightSet.add(clone);
        light->clone(*clone);
    }

    lightSet.setAmbientLight(ambientLight());
}

// ----------------------------------------------------------------------------
//  Adds a new light to the scene 
// ----------------------------------------------------------------------------
std::shared_ptr<Light> LightSet::add()
{
    auto light = Light::create();
    m_lights.push_back(light);
    return light;
}

// ----------------------------------------------------------------------------
//  Adds a new light to the scene 
// ----------------------------------------------------------------------------
void LightSet::add(std::shared_ptr<Light> light)
{
    light->setParent(shared_from_this());
    m_lights.push_back(light);
}

// ----------------------------------------------------------------------------
//  Locates a child light element by its ID
// ----------------------------------------------------------------------------
std::shared_ptr<Light> LightSet::find(int id)
{
    BOOST_FOREACH (auto & light, m_lights)
    {
        if (light->id() == id) return light;
    }
    
    return nullptr;
}

// ----------------------------------------------------------------------------
//  Removes an existing light from the scene
// ----------------------------------------------------------------------------
void LightSet::remove(std::shared_ptr<Light> light)
{ 
    m_lights.remove(light);
    light->setParent(nullptr);
}

// ----------------------------------------------------------------------------
//  Interpolates between lighting sets
// ----------------------------------------------------------------------------
std::shared_ptr<LightSet> LightSet::interp(std::shared_ptr<LightSet> k2, float f)
{ 
    auto set = LightSet::create();

    BOOST_FOREACH (auto & keyLight, m_lights)
    {
        auto result = set->add();

        auto key1 = keyLight;
        auto key2 = k2->find(keyLight->id());
        key2 = key2 ? key2 : key1;

#define VOX_LERP(ATTR) result->ATTR = key1->ATTR*(1-f) + key2->ATTR*f;

        VOX_LERP(m_position);
        VOX_LERP(m_color);
        
#undef VOX_LERP
    }

    set->m_ambientLight = m_ambientLight * (1.f-f) + k2->m_ambientLight * f;

    return set;
}

// ----------------------------------------------------------------------------
//  Clones the light
// ----------------------------------------------------------------------------
void Light::clone(Light & light)
{
    light.setId(id());
    light.m_color    = m_color;
    light.m_position = m_position;
}

} // namespace vox
