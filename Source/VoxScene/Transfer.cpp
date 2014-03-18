/* ===========================================================================

	Project: VoxRender

	Description: Transfer function applied to volume dataset

    Copyright (C) 2013-2014 Lucas Sherman

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
#include "VoxLib/Core/Debug.h"
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

            size_t x1 = static_cast<size_t>(curr->density * samples);
            if (x1 >= map.width()) x1 = map.width()-1;
            size_t x2 = static_cast<size_t>(next->density * samples);
            if (x2 >= map.width()) x2 = map.width()-1;                
            Vector3f s1 = Vector3f(curr->material->emissive) / 255.0f * curr->material->emissiveStrength;
            Vector3f s2 = Vector3f(next->material->emissive) / 255.0f * curr->material->emissiveStrength;
            Vector4f y1(s1[0], s1[1], s1[2], 0.0f);
            Vector4f y2(s2[0], s2[1], s2[2], 0.0f);

            for (size_t i = x1; i <= x2; i++)
            {
                float px = i / samples - curr->density;
                float py = next->density - curr->density;
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

            size_t x1 = static_cast<size_t>(curr->density * samples);
            if (x1 >= map.width()) x1 = map.width()-1;
            size_t x2 = static_cast<size_t>(next->density * samples);
            if (x2 >= map.width()) x2 = map.width()-1;                
            Vector3f s1(curr->material->specular);
            Vector3f s2(next->material->specular);
            Vector4f y1(s1[0], s1[1], s1[2], curr->material->glossiness*255.0f);
            Vector4f y2(s2[0], s2[1], s2[2], next->material->glossiness*255.0f);

            for (size_t i = x1; i <= x2; i++)
            {
                float px = i / samples - curr->density;
                float py = next->density - curr->density;
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

            size_t x1 = static_cast<size_t>(curr->density * samples);
            if (x1 >= map.width()) x1 = map.width()-1;
            size_t x2 = static_cast<size_t>(next->density * samples);
            if (x2 >= map.width()) x2 = map.width()-1;
            Vector3f s1(curr->material->diffuse);
            Vector3f s2(next->material->diffuse);
            Vector4f y1(s1[0], s1[1], s1[2], 0.0f);
            Vector4f y2(s2[0], s2[1], s2[2], 0.0f);

            for (size_t i = x1; i <= x2; i++)
            {
                float px = i / samples - curr->density;
                float py = next->density - curr->density;
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

            size_t x1 = static_cast<size_t>(curr->density * samples);
            if (x1 >= map.width()) x1 = map.width()-1;
            size_t x2 = static_cast<size_t>(next->density * samples);
            if (x2 >= map.width()) x2 = map.width()-1;
            float  y1 = curr->material->opticalThickness;
            float  y2 = next->material->opticalThickness;

            for (size_t i = x1; i <= x2; i++)
            {
                float px = i / samples - curr->density;
                float py = next->density - curr->density;
                float part = low( high(px / py, 0.0f), 1.0f );
                buffer[i] = - logf( 1.f - (1.f - part) * y1 - part * y2 );
            }

            curr = next;
            iter++;
        }
    }
    
    // ----------------------------------------------------------------------------
    //  Helper classes for triangle rasterization
    // ----------------------------------------------------------------------------
    class Edge
    {
    public:
        Edge(Vector2f const& _p1, std::shared_ptr<Material> & _m1,
             Vector2f const& _p2, std::shared_ptr<Material> & _m2)
        {
            if (_p1[1] < _p2[1])
            {
                p1 = Vector2(_p1);
                p2 = Vector2(_p2);
                m1 = _m1;
                m2 = _m2;
            }
            else
            {
                p1 = Vector2(_p2);
                p2 = Vector2(_p1);
                m1 = _m2;
                m2 = _m1;
            }
        }

        Vector2 p1; 
        Vector2 p2;
        std::shared_ptr<Material> m1; 
        std::shared_ptr<Material> m2;
    };
    class Span
    {
    public:
        Span(Material const& _m1, int _x1, Material const& _m2, int _x2)
        {
            if (_x1 < _x2) 
            {
                m1 = _m1;
                x1 = _x1;
                m2 = _m2;
                x2 = _x2;
            } 
            else 
            {
                m1 = _m1;
                x1 = _x2;
                m2 = _m1;
                x2 = _x1;
            }
        }

        Material m1;
        Material m2;
        int x1;
        int x2;
    };
    
    // ----------------------------------------------------------------------------
    //  Interpolates between two materials
    // ----------------------------------------------------------------------------
    Material interp(Material const& m1, Material const& m2, float f)
    {
        Material result;
        result.opticalThickness = m1.opticalThickness * (1-f) + m2.opticalThickness * f;
        result.glossiness       = m1.glossiness       * (1-f) + m2.glossiness       * f;
        result.emissiveStrength = m1.emissiveStrength * (1-f) + m2.emissiveStrength * f;
        result.diffuse          = static_cast<Vector3f>(m1.diffuse) * (1-f) +
                                  static_cast<Vector3f>(m2.diffuse) * f;
        result.emissive         = static_cast<Vector3f>(m1.emissive) * (1-f) +
                                  static_cast<Vector3f>(m2.emissive) * f;
        result.specular         = static_cast<Vector3f>(m1.specular) * (1-f) +
                                  static_cast<Vector3f>(m2.specular) * f;
        return result;
    }
   
    // ----------------------------------------------------------------------------
    //  Draws a single span between two edges
    // ----------------------------------------------------------------------------
    void drawSpan(std::shared_ptr<TransferMap> & map, Span const& span, int y)
    {
        if (y < 0 || y >= map->specular().height()) return;

        int xdiff = span.x2 - span.x1;
        if (xdiff == 0) return;

        float factor = 0.f;
        float factorStep = 1.f / (float)xdiff;

        for (int x = span.x1; x < span.x2; x++)
        {
            if (x < 0 || x >= map->specular().width()) return;

            auto m = interp(span.m1, span.m2, factor);
            map->specular().at(x, y, 0) = Vector<UInt8,4>(m.specular[0], m.specular[1], m.specular[2], m.glossiness*255.0f);
            map->diffuse().at(x, y, 0)  = Vector<UInt8,4>(m.diffuse[0], m.diffuse[1], m.diffuse[2], 0.f);
            map->opacity().at(x, y, 0)  = m.opticalThickness;
            factor += factorStep;
        }
    }

    // ----------------------------------------------------------------------------
    //  Draws all of the spans for 2 edges
    // ----------------------------------------------------------------------------
    void drawSpansBetweenEdges(std::shared_ptr<TransferMap> & map, Edge & e1, Edge & e2)
    {
        // Calculate differences between the y coordinates
        float e1ydiff = (float)(e1.p2[1] - e1.p1[1]);
        if (e1ydiff == 0.0f) return;
        float e2ydiff = (float)(e2.p2[1] - e2.p1[1]);
        if (e2ydiff == 0.0f) return;

        // Calculate differences between the x coordinates
        float e1xdiff = (float)(e1.p2[0] - e1.p1[0]);
        float e2xdiff = (float)(e2.p2[0] - e2.p1[0]);

        // Calculate factors to use for interpolation
        // with the edges and the step values to increase
        // them by after drawing each span
        float factor1 = (float)(e2.p1[1] - e1.p1[1]) / e1ydiff;
        float factorStep1 = 1.0f / e1ydiff;
        float factor2 = 0.0f;
        float factorStep2 = 1.0f / e2ydiff;

        // Loop through the lines between the edges and draw spans
        for(int y = e2.p1[1]; y < e2.p2[1]; y++) 
        {
            // Draw the span
            auto m1 = interp(*e1.m1, *e1.m2, factor1);
            auto m2 = interp(*e2.m1, *e2.m2, factor2);
            drawSpan(map, Span(m1, e1.p1[0] + (int)(e1xdiff * factor1),
                               m2, e2.p1[0] + (int)(e2xdiff * factor2)), y);

            // Increase factors
            factor1 += factorStep1;
            factor2 += factorStep2;
        }
    }
 
    // ----------------------------------------------------------------------------
    //  Renders a triangle to the transfer function map -- 
    //  http://joshbeam.com/articles/triangle_rasterization
    // ----------------------------------------------------------------------------
    void drawTriangle(
        std::shared_ptr<TransferMap> & map,
        Vector2f const& p1, std::shared_ptr<Material> m1,
        Vector2f const& p2, std::shared_ptr<Material> m2,
        Vector2f const& p3, std::shared_ptr<Material> m3)
    {
        // Create edges for the triangle
        Edge edges[3] = {
            Edge(p1, m1, p2, m2),
            Edge(p2, m2, p3, m3),
            Edge(p3, m3, p1, m1)
        };

        int maxLength = 0;
        int longEdge = 0;

        // Find the full height spanning edge
        for(int i = 0; i < 3; i++) 
        {
            int length = edges[i].p2[1] - edges[i].p1[1];
            if(length > maxLength) 
            {
                    maxLength = length;
                    longEdge = i;
            }
        }

        // Draw the spans between other edges and the max edge
        drawSpansBetweenEdges(map, edges[longEdge], edges[(longEdge+1)%3]);
        drawSpansBetweenEdges(map, edges[longEdge], edges[(longEdge+2)%3]);
    }

} // namespace filescope
} // namespace anonymous

// ----------------------------------------------------------------------------
//  Creates a new transfer function node
// ----------------------------------------------------------------------------
std::shared_ptr<Node> Node::create(float density, std::shared_ptr<Material> material) 
{ 
    auto node = std::shared_ptr<Node>(new Node()); 

    node->density = density;

    if (material) node->material = material;
    else          node->material = Material::create();

    return node;
}

// ----------------------------------------------------------------------------
//  Initializes a quad to default settings
// ----------------------------------------------------------------------------
Quad::Quad() : position(0.5f, 0.5f), heights(0.2f, 0.2f), widths(0.2f, 0.2f) 
{ 
    BOOST_FOREACH (auto & material, materials) material = Material::create();
}

// ----------------------------------------------------------------------------
//  Changes the desired resolution of the transfer function's mapping texture 
// ----------------------------------------------------------------------------
void Transfer::setResolution(Vector3u const& resolution)
{
    m_resolution = resolution;
}

// ----------------------------------------------------------------------------
//  Adds a new node to the transfer function  
// ----------------------------------------------------------------------------
void Transfer1D::add(std::shared_ptr<Node> node)
{
    auto iter = m_nodes.begin();
    while (iter != m_nodes.end() && (iter->get()->density <= node->density)) 
        ++iter;

    m_nodes.insert(iter, node);
}

// ----------------------------------------------------------------------------
//  Removes a node from the transfer function  
// ----------------------------------------------------------------------------
void Transfer1D::remove(std::shared_ptr<Node> node)
{
    setDirty();

    m_nodes.remove(node);
}

// ----------------------------------------------------------------------------
//  Performs 1D transfer function interpolation
// ----------------------------------------------------------------------------
std::shared_ptr<Transfer> Transfer1D::interp(std::shared_ptr<Transfer> k2, float f)
{
    auto result = Transfer1D::create();

    if (auto key2 = dynamic_cast<Transfer1D*>(k2.get()))
    {
        BOOST_FOREACH (auto & node, m_nodes)
        {
        }
    }
    
    return result;
}

// ----------------------------------------------------------------------------
//  Maps the transfer function to a texture of the specified resolution 
// ----------------------------------------------------------------------------
void Transfer1D::generateMap(std::shared_ptr<TransferMap> map)
{
    m_nodes.sort(filescope::slt<Node>);

    VOX_ASSERT(map);

    map->lock();

    auto & diffuse = map->diffuse();
    diffuse.resize(128, 1, 1);
    auto & opacity = map->opacity();
    opacity.resize(128, 1, 1);
    auto & specular = map->specular();
    specular.resize(128, 1, 1);
    auto & emissive = map->emissive();
    emissive.resize(64, 1, 1);

    if (m_nodes.size())
    {
        filescope::mapEmissive(emissive, m_nodes);
        filescope::mapSpecular(specular, m_nodes);
        filescope::mapOpacity(opacity, m_nodes);
        filescope::mapDiffuse(diffuse, m_nodes);
    }
    else opacity.clear();

    map->setDirty();

    map->unlock();
}

// ----------------------------------------------------------------------------
//  Adds a new quad to the transfer function  
// ----------------------------------------------------------------------------
void Transfer2D::add(std::shared_ptr<Quad> quad)
{
    m_quads.push_back(quad);
}

// ----------------------------------------------------------------------------
//  Removes a quad from the transfer function
// ----------------------------------------------------------------------------
void Transfer2D::remove(std::shared_ptr<Quad> quad)
{
    setDirty();

    m_quads.remove(quad);
}

// ----------------------------------------------------------------------------
//  Maps the transfer function to a texture of the specified resolution 
// ----------------------------------------------------------------------------
void Transfer2D::generateMap(std::shared_ptr<TransferMap> map)
{
    VOX_ASSERT(map);

    map->lock();

    // Resize the map images as necessary
    auto & diffuse = map->diffuse();
    diffuse.resize(256, 128, 1);
    diffuse.clear();
    auto & opacity = map->opacity();
    opacity.resize(256, 128, 1);
    opacity.clear();
    auto & specular = map->specular();
    specular.resize(256, 128, 1);
    specular.clear();
    auto & emissive = map->emissive();
    emissive.resize(1, 1, 1);
    emissive.clear();

    // Cycle through each quad
    auto res = Vector2f(256.f, 128.f);
    BOOST_FOREACH (auto & quad, m_quads)
    {
        auto h1 = quad->heights[0] / 2;
        auto h2 = quad->heights[1] / 2;
        auto w1 = quad->widths[0] / 2;
        auto w2 = quad->widths[1] / 2;
        filescope::drawTriangle(map, 
            (quad->position + Vector2f(-w1,  h1))*res, quad->materials[Quad::Node_UL],
            (quad->position + Vector2f( w1,  h2))*res, quad->materials[Quad::Node_UR],
            (quad->position + Vector2f(-w2, -h1))*res, quad->materials[Quad::Node_LL]);
        filescope::drawTriangle(map, 
            (quad->position + Vector2f( w1,  h2))*res, quad->materials[Quad::Node_UR],
            (quad->position + Vector2f( w2, -h2))*res, quad->materials[Quad::Node_LR],
            (quad->position + Vector2f(-w2, -h1))*res, quad->materials[Quad::Node_LL]);
    }

    map->setDirty();

    map->unlock();
}

} // namespace vox