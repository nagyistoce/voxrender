/* ===========================================================================

	Project: VoxLib

	Description: Data structure defining material properties of volume

    Copyright (C) 2012-2013 Lucas Sherman

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
#include "Material.h"

// Include Dependencies
#include "VoxLib/Scene/Transfer.h"

namespace vox {

// ----------------------------------------------------------------------------
//  Implementation structure for Material
// ----------------------------------------------------------------------------
class Material::Impl
{
public:
    Impl::Impl() :
          m_opticalThickness(0.0f),
          m_glossiness(80.0f),
          m_emissiveStrength(0.0f),
          m_emissive(0, 0, 0),
          m_diffuse(255, 255, 255),
          m_specular(0, 0, 0)
        {
        }

    std::list<std::shared_ptr<Node>> m_holders; ///< List of nodes referencing this material

    float m_opticalThickness; ///< Optical thickness of material (-INF, INF)
    float m_glossiness;       ///< Glossiness factor
    float m_emissiveStrength; ///< Emissive light intensity

    Vector<UInt8,3> m_emissive; ///< Emissive color
    Vector<UInt8,3> m_diffuse;  ///< Diffuse reflection color
    Vector<UInt8,3> m_specular; ///< Specular reflection color
};

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
Material::Material() : m_pImpl(new Material::Impl()) { }

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
Material::~Material() { delete m_pImpl; }

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
float Material::opticalThickness() const 
{ 
    return m_pImpl->m_opticalThickness; 
}
        
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void Material::setOpticalThickness(float thickness)
{
    if (m_pImpl->m_opticalThickness == thickness) return;

    m_pImpl->m_opticalThickness = thickness;

    setDirty();
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
float Material::glossiness() const 
{ 
    return m_pImpl->m_glossiness; 
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void Material::setGlossiness(float glossiness)
{
    if (m_pImpl->m_glossiness == glossiness) return;

    m_pImpl->m_glossiness = glossiness;
    
    setDirty();
}
        
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
float Material::emissiveStrength() const 
{ 
    return m_pImpl->m_emissiveStrength; 
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void Material::setEmissiveStrength(float intensity) 
{ 
    if (m_pImpl->m_emissiveStrength == intensity) return;

    m_pImpl->m_emissiveStrength = intensity;
    
    setDirty();
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
Vector<UInt8,3> Material::emissive() const 
{ 
    return m_pImpl->m_emissive; 
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void Material::setEmissive(Vector<UInt8,3> const& emissive)
{
    if (m_pImpl->m_emissive == emissive) return;

    m_pImpl->m_emissive = emissive;
    
    setDirty();
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
Vector<UInt8,3> const& Material::diffuse() const 
{ 
    return m_pImpl->m_diffuse; 
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void Material::setDiffuse(Vector<UInt8,3> const& diffuse)
{
    if (m_pImpl->m_diffuse == diffuse) return;

    m_pImpl->m_diffuse = diffuse;
    
    setDirty();
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
Vector<UInt8,3> const& Material::specular() const 
{ 
    return m_pImpl->m_specular; 
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void Material::setSpecular(Vector<UInt8,3> const& specular)
{
    if (m_pImpl->m_specular == specular) return;

    m_pImpl->m_specular = specular;
    
    setDirty();
}

// ----------------------------------------------------------------------------
//  Marks the material and any watchers dirty
// ----------------------------------------------------------------------------
void Material::setDirty(bool dirty) 
{ 
    m_isDirty = dirty; 

    BOOST_FOREACH(auto & node, m_pImpl->m_holders)
    {
        node->setDirty();
    }
}

// ----------------------------------------------------------------------------
//  Registers a new node with the material (which implies the node is using it)
// ----------------------------------------------------------------------------
void Material::addNode(std::shared_ptr<Node> node)
{
    m_pImpl->m_holders.push_front(node);
}

// ----------------------------------------------------------------------------
//  Registers a new node with the material (which implies the node is using it)
// ----------------------------------------------------------------------------
void Material::removeNode(std::shared_ptr<Node> node)
{
    m_pImpl->m_holders.remove(node);
}

} // namespace vox