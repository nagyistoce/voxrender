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

// Internal Dependencies
#include "VoxScene/Common.h"
#include "VoxScene/Material.h"
#include "VoxScene/TransferMap.h"
#include "VoxScene/Object.h"

// External Dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Core/Functors.h"
#include "VoxLib/Core/Geometry/Vector.h"
#include "VoxLib/Core/Types.h"

// API Namespace
namespace vox
{
    class RenderController;
    class Transfer;
    class Node;

    typedef std::shared_ptr<Node> NodeH;

    /** Transfer function node */
    class VOXS_EXPORT Node : public SubObject
    {
    public:
        /** Constructs a new transfer function object */
        static std::shared_ptr<Node> create(float density = 1.0f, std::shared_ptr<Material> material = nullptr);

        // Node comparison operators for sorting operations
        bool operator<(Node const& rhs)  { return density < rhs.density; }
        bool operator>(Node const& rhs)  { return density > rhs.density; }
        bool operator==(Node const& rhs) 
        { 
            return density == rhs.density &&
                   material == rhs.material; 
        }

        std::shared_ptr<Material> material;   ///< Material properties

	    float density; ///< Normalized density

    private:
        Node() { }
    };

    /** 2D Transfer function block */
    class VOXS_EXPORT Quad : public SubObject
    {
    public:
        /** Node position indices */
        enum Node
        {
            Node_Begin = 0,
            Node_UL = Node_Begin,
            Node_UR,
            Node_LL,
            Node_LR,
            Node_End
        };

    public:
        /** Constructs a new quad object */
        static std::shared_ptr<Quad> create() { return std::shared_ptr<Quad>(new Quad()); }

        Vector<std::shared_ptr<Material>,4> materials; ///< Corner materials

        Vector2f position;    ///< Center position of the quad
        Vector2f heights;     ///< Edge heights of the quad
        Vector2f widths;      ///< Edge widths of the quad

    private:
        Quad();
    };

    /** Transfer function interface */
    class VOXS_EXPORT Transfer : public std::enable_shared_from_this<Transfer>, public Object
    {
    public: 
        /** Initializes the default transfer function resolution */
        Transfer(int id = 0) : Object(id), m_resolution(256, 128, 8) { }

        /** Interpolates the transfer function towards k2 by a factor f */
        virtual std::shared_ptr<Transfer> interp(std::shared_ptr<Transfer> k2, float f) = 0;

        /** Updates the input map based on the this transfer function */
        virtual void generateMap(std::shared_ptr<TransferMap> map) = 0;

        /** Clones a transfer function instance */
        virtual std::shared_ptr<Transfer> clone() = 0;

        /** Returns the type identifier of the derived class */
        virtual Char const* type() = 0;

        /** Sets the transfer function resolution */
        void setResolution(Vector3u const& resolution);

        /** Returns the resolution of the transfer function */
        Vector3u const& resolution() const { return m_resolution; }

    protected:
        friend RenderController;

        Vector3u m_resolution; ///< Transfer function map resolution
        Vector3f m_range[2];   ///< Subrange of volume to which transfer is applied (anything outside is 0 density)
    };

    /** 1 Dimensional Transfer Function */
    class VOXS_EXPORT Transfer1D : public Transfer
    {
    public:
        /** Constructs a new transfer function object */
        static std::shared_ptr<Transfer1D> create(int id = 0) 
        { 
            return std::shared_ptr<Transfer1D>(new Transfer1D(0)); 
        }

        /** 1D transfer function interpolation */
        virtual std::shared_ptr<Transfer> interp(std::shared_ptr<Transfer> k2, float f);

        /** Sets the 1D transfer function resolution */
        void setResolution(size_t resolution) { Transfer::setResolution(Vector3u(resolution, 1, 1)); }

        /** Generates the associated transfer map */
        virtual void generateMap(std::shared_ptr<TransferMap> map);

        /** Generates a clone of the transfer function object */
        virtual std::shared_ptr<Transfer> clone();

        /** Returns the type of a transfer function */
        virtual Char const* type() { return Transfer1D::typeID(); }

        /** Returns the type of the 1D transfer function */
        static Char const* typeID() { return "Transfer1D"; }

        /** Adds a new node to the transfer function */
        void add(std::shared_ptr<Node> node);

        /** Removes a node from the transfer function */
        void remove(std::shared_ptr<Node> node);

        /** Returns a linked list of the transfer function nodes */
        std::list<std::shared_ptr<Node>> & nodes() { return m_nodes; }

    private:
        /** Initializes a new transfer function object */
        Transfer1D(int id) : Transfer(id) { }

        Transfer1D(Transfer1D & copy);

        std::list<NodeH> m_nodes;
    };

    /** 2 Dimensional Transfer Function */
    class VOXS_EXPORT Transfer2D : public Transfer
    {
    public:
        /** Constructs a new transfer function object */
        static std::shared_ptr<Transfer2D> create(int id = 0) { return std::shared_ptr<Transfer2D>(new Transfer2D(id)); }
        
        /** 2D transfer function interpolation */
        virtual std::shared_ptr<Transfer> interp(std::shared_ptr<Transfer> k2, float f) { return k2; }

        /** Sets the 2D transfer function resolution without requiring a Vec3 */
        void setResolution(Vector2u res) { Transfer::setResolution(Vector3u(res[0], res[1], 1)); }

        /** Generates the associated transfer map */
        virtual void generateMap(std::shared_ptr<TransferMap> map);
        
        /** Generates a clone of the transfer function object */
        virtual std::shared_ptr<Transfer> clone();

        /** Returns the type of a transfer function */
        virtual Char const* type() { return Transfer2D::typeID(); }

        /** Returns the type of the 1D transfer function */
        static Char const* typeID() { return "Transfer2D"; }

        /** Adds a new node to the transfer function */
        void add(std::shared_ptr<Quad> quad);

        /** Removes a node from the transfer function */
        void remove(std::shared_ptr<Quad> quad);

        /** Returns a linked list of the transfer function nodes */
        std::list<std::shared_ptr<Quad>> & quads() { return m_quads; }

    private:
        /** Initializes a new transfer function object */
        Transfer2D(int id) : Transfer(id) { }

        std::list<std::shared_ptr<Quad>> m_quads;
    };
}

// End definition
#endif // VOX_TRANSFER_H