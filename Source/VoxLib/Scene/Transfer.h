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

// Begin definition
#ifndef VOX_TRANSFER_H
#define VOX_TRANSFER_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Functors.h"
#include "VoxLib/Core/Geometry/Color.h"
#include "VoxLib/Core/Geometry/Vector.h"
#include "VoxLib/Core/Types.h"
#include "VoxLib/Scene/Material.h"

// API Namespace
namespace vox
{
    class Transfer;

    /** Transfer function node */
    class VOX_EXPORT Node 
    {
    public:
        /** Constructs a new node with an optionally specifiable material */
        Node(std::shared_ptr<Material> material = std::shared_ptr<Material>());

        // Node comparison operators for sorting operations
        bool operator<(Node const& rhs) { return m_position < rhs.m_position; }
        bool operator>(Node const& rhs) { return m_position > rhs.m_position; }
        bool operator==(Node const& rhs) 
        { 
            return m_position == rhs.m_position &&
                   m_material == rhs.m_material; 
        }

        /** Sets the normalized position of the node in the specified dimension */
        void setPosition(int dim, float position) { m_position[dim] = position; }

        /** Returns the normalized position of the node in the specified dimension */
        float position(int dim) const { return m_position[dim]; }

        /** Sets the node's material properties */
        void setMaterial(std::shared_ptr<Material> material) { m_material = material; }
        
        /** Returns the material properties of the node */
        std::shared_ptr<Material> material() { return m_material; }

    private:
        friend Transfer; 

	    Vector3f m_position; ///< Normalized node transfer coordinates

        std::shared_ptr<Material> m_material;   ///< Material properties

        bool  m_contextChanged;  ///< Dirty flag
    };

    /** Transfer Function */
    class VOX_EXPORT Transfer
    {
    public:
        /** Initializes a new transfer function object */
        Transfer() : m_contextChanged(true) { }

        /** Sets the desired resolution of the transfer function */
        void setResolution(Vector3u const& resolution);

        /** Returns the desired resolution of the transfer function */
        Vector3u resolution() const { return m_resolution; }

        /** Adds a new node to the transfer function */
        void addNode(std::shared_ptr<Node> node);

        /** Removes a node from the transfer function */
        void removeNode(std::shared_ptr<Node> node);

        /** Returns a linked list of the transfer function nodes */
        std::list<std::shared_ptr<Node>> nodes() { return m_nodes; }

        /** 
         * Returns a map of the transfer function content 
         *
         * The map is an array with the dimensions specified by the resolution member.
         */
        std::shared_ptr<Material> map();

        uchar4 const* debugMap() { return m_debugMap; }
        void addDebugMap(UInt8 const& density, uchar4 const& color) 
        {
            m_debugMap[density] = color;
        }

        /** Returns true if an unprocessed context change has occured */
        bool isDirty() const { return m_contextChanged; }

    private:
        Vector3u m_resolution;
        uchar4 m_debugMap[256];

        std::list<std::shared_ptr<Node>> m_nodes;

        bool m_contextChanged;
    };
}

// End definition
#endif // VOX_TRANSFER_H