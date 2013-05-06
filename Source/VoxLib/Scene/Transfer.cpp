/* ===========================================================================

	Project: Transfer - Transfer Function

	Description: Transfer function applied to volume dataset

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

// Include Header
#include "Transfer.h"

// API namespace
namespace vox
{
    
namespace {
namespace filescope {

    /** Shared_ptr sorting operator */
    template<typename T> bool slt(
        const std::shared_ptr<T>& left,
        const std::shared_ptr<T>& right
        )
    {
       return (*left.get() < *right.get());
    }

} // namespace filescope
} // namespace anonymous

// ----------------------------------------------------------------------------
//  Creates a new node and initializes the material property structure if one
//  is not specified by the user
// ----------------------------------------------------------------------------
Node::Node(std::shared_ptr<Material> material) : m_contextChanged(true)
{ 
    if (material)
    {
        m_material = material;
    }
    else
    {
        m_material = std::make_shared<Material>();
    }
}

// ----------------------------------------------------------------------------
//  Changes the desired resolution of the transfer function's mapping texture 
// ----------------------------------------------------------------------------
void Transfer::setResolution(Vector3u const& resolution)
{
    if (m_resolution != resolution)
    {
        m_resolution = resolution;

        m_contextChanged = true;
    }
}

// ----------------------------------------------------------------------------
//  Adds a new node to the transfer function  
// ----------------------------------------------------------------------------
void Transfer::addNode(std::shared_ptr<Node> node)
{
    m_nodes.push_back(node);
}

// ----------------------------------------------------------------------------
//  Removes a node from the transfer function  
// ----------------------------------------------------------------------------
void Transfer::removeNode(std::shared_ptr<Node> node)
{
    m_nodes.remove(node);
}

// ----------------------------------------------------------------------------
//  Maps the transfer function to a texture of the specified resolution 
//  :TEST: 1D transfer function
// ----------------------------------------------------------------------------
std::shared_ptr<TransferMap> Transfer::generateMap()
{
	std::shared_ptr<TransferMap> map = std::make_shared<TransferMap>();

	m_nodes.sort(filescope::slt<Node>);

    map->diffuse.resize(m_resolution[0], m_resolution[1], m_resolution[2]);

    return map;
}

} // namespace vox