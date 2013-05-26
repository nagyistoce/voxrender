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

// Include Dependencies
#include "VoxLib/Core/Functors.h"

// API namespace
namespace vox
{
    
namespace {
namespace filescope {
    
    // ----------------------------------------------------------------------------
    //  Shared_ptr sorting operator
    // ----------------------------------------------------------------------------
    template<typename T> bool slt(
        const std::shared_ptr<T>& left,
        const std::shared_ptr<T>& right
        )
    {
       return (*left.get() < *right.get());
    }
    
    // ----------------------------------------------------------------------------
    //  Generates an emissive map from the tranfer specification
    // ----------------------------------------------------------------------------
    void mapEmissive(Image3D<Vector4f> & map, std::list<std::shared_ptr<Node>> transfer)
    {
        float samples = static_cast<float>(map.width()) - 1;
        auto buffer = map.data();

        memset(buffer, 0, map.size()*sizeof(Vector4f));

        auto iter = transfer.begin();
        auto curr = *iter; 
        iter++;
        
        while (iter != transfer.end())
        {
            auto next = *iter;

            size_t x1 = static_cast<size_t>(curr->position(0) * samples);
            if (x1 >= map.width()) x1 = map.width()-1;
            size_t x2 = static_cast<size_t>(next->position(0) * samples);
            if (x2 >= map.width()) x2 = map.width()-1;                
            Vector3f s1 = Vector3f(curr->material()->emissive()) / 255.0f * curr->material()->emissiveStrength();
            Vector3f s2 = Vector3f(next->material()->emissive()) / 255.0f * curr->material()->emissiveStrength();
            Vector4f y1(s1[0], s1[1], s1[2], 0.0f);
            Vector4f y2(s2[0], s2[1], s2[2], 0.0f);

            for (size_t i = x1; i <= x2; i++)
            {
                float px = i / samples - curr->position(0);
                float py = next->position(0) - curr->position(0);
                float part = low( high(px / py, 0.0f), 1.0f );
                buffer[i] = y1*(1.f - part) + y2*part;
            }

            curr = next;
            iter++;
        }
    }

    // ----------------------------------------------------------------------------
    //  Generates a diffuse map from the transfer function specification
    // ----------------------------------------------------------------------------
    void mapSpecular(Image3D<Vector<UInt8,4>> & map, std::list<std::shared_ptr<Node>> transfer)
    {
        float samples = static_cast<float>(map.width()) - 1;
        auto buffer = map.data();

        memset(buffer, 0, map.size()*sizeof(Vector<UInt8,4>));

        auto iter = transfer.begin();
        auto curr = *iter; 
        iter++;
        
        while (iter != transfer.end())
        {
            auto next = *iter;

            size_t x1 = static_cast<size_t>(curr->position(0) * samples);
            if (x1 >= map.width()) x1 = map.width()-1;
            size_t x2 = static_cast<size_t>(next->position(0) * samples);
            if (x2 >= map.width()) x2 = map.width()-1;                
            Vector3f s1(curr->material()->specular());
            Vector3f s2(next->material()->specular());
            Vector4f y1(s1[0], s1[1], s1[2], curr->material()->glossiness()*255.0f);
            Vector4f y2(s2[0], s2[1], s2[2], next->material()->glossiness()*255.0f);

            for (size_t i = x1; i <= x2; i++)
            {
                float px = i / samples - curr->position(0);
                float py = next->position(0) - curr->position(0);
                float part = low( high(px / py, 0.0f), 1.0f );
                buffer[i] = static_cast<Vector<UInt8,4>>(y1*(1.f - part) + y2*part);
            }

            curr = next;
            iter++;
        }
    }

    // ----------------------------------------------------------------------------
    //  Generates a diffuse map from the transfer function specification
    // ----------------------------------------------------------------------------
    void mapDiffuse(Image3D<Vector<UInt8,4>> & map, std::list<std::shared_ptr<Node>> transfer)
    {
        float samples = static_cast<float>(map.width()) - 1;
        auto buffer = map.data();

        memset(buffer, 0, map.size()*sizeof(Vector<UInt8,4>));

        auto iter = transfer.begin();
        auto curr = *iter; 
        iter++;
        
        while (iter != transfer.end())
        {
            auto next = *iter;

            size_t x1 = static_cast<size_t>(curr->position(0) * samples);
            if (x1 >= map.width()) x1 = map.width()-1;
            size_t x2 = static_cast<size_t>(next->position(0) * samples);
            if (x2 >= map.width()) x2 = map.width()-1;
            Vector3f s1(curr->material()->diffuse());
            Vector3f s2(next->material()->diffuse());
            Vector4f y1(s1[0], s1[1], s1[2], 0.0f);
            Vector4f y2(s2[0], s2[1], s2[2], 0.0f);

            for (size_t i = x1; i <= x2; i++)
            {
                float px = i / samples - curr->position(0);
                float py = next->position(0) - curr->position(0);
                float part = low( high(px / py, 0.0f), 1.0f );
                buffer[i] = static_cast<Vector<UInt8,4>>(y1*(1.f - part) + y2*part);
            }

            curr = next;
            iter++;
        }
    }

    // ----------------------------------------------------------------------------
    //  Generates a opacity map from the transfer function specification
    // ----------------------------------------------------------------------------
    void mapOpacity(Image3D<float> & map, std::list<std::shared_ptr<Node>> transfer)
    {
        float samples = static_cast<float>(map.width()) - 1;
        auto buffer = map.data();

        memset(buffer, 0, map.size()*sizeof(float));

        auto iter = transfer.begin();
        auto curr = *iter; 
        iter++;

        while (iter != transfer.end())
        {
            auto next = *iter;

            size_t x1 = static_cast<size_t>(curr->position(0) * samples);
            if (x1 >= map.width()) x1 = map.width()-1;
            size_t x2 = static_cast<size_t>(next->position(0) * samples);
            if (x2 >= map.width()) x2 = map.width()-1;
            float  y1 = curr->material()->opticalThickness();
            float  y2 = next->material()->opticalThickness();

            for (size_t i = x1; i <= x2; i++)
            {
                float px = i / samples - curr->position(0);
                float py = next->position(0) - curr->position(0);
                float part = low( high(px / py, 0.0f), 1.0f );
                buffer[i] = - logf( 1.f - (1.f - part) * y1 - part * y2 );
            }

            curr = next;
            iter++;
        }
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
    m_contextChanged = true;

    m_nodes.push_back(node);
}

// ----------------------------------------------------------------------------
//  Removes a node from the transfer function  
// ----------------------------------------------------------------------------
void Transfer::removeNode(std::shared_ptr<Node> node)
{
    m_contextChanged = true;

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

    // Resize the maps to match requested resolution
    map->diffuse.resize(128, 1, 1);
    map->opacity.resize(128, 1, 1);
    map->specular.resize(128, 1, 1);
    map->emissive.resize(64, 1, 1);

    // Generate the opacity mapping
    filescope::mapOpacity(map->opacity, m_nodes);
    filescope::mapDiffuse(map->diffuse, m_nodes);
    filescope::mapSpecular(map->specular, m_nodes);
    filescope::mapEmissive(map->emissive, m_nodes);

    return map;
}

} // namespace vox