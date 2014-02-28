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
    class VOXS_EXPORT Node
    {
    public:
        /** Constructs a new transfer function object */
        static std::shared_ptr<Node> create(float density = 1.0f, std::shared_ptr<Material> material = nullptr);

        // Node comparison operators for sorting operations
        bool operator<(Node const& rhs) { return density < rhs.density; }
        bool operator>(Node const& rhs) { return density > rhs.density; }
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
    class VOXS_EXPORT Quad
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
        static std::shared_ptr<Quad> create() {
            return std::shared_ptr<Quad>(new Quad());
        }

        Vector<std::shared_ptr<Material>,4> materials; ///< Corner materials

        Vector2f position;    ///< Center position of the quad
        Vector2f heights;     ///< Edge heights of the quad
        Vector2f widths;      ///< Edge widths of the quad

    private:
        Quad();
    };

    /** Transfer function interface */
    class VOXS_EXPORT Transfer : public std::enable_shared_from_this<Transfer>
    {
    public: 
        /** Initializes the default transfer function resolution */
        Transfer() : m_isDirty(true), m_resolution(128, 32, 1) { }

        /** Updates the input map based on the this transfer function */
        virtual void generateMap(std::shared_ptr<TransferMap> map) = 0;

        /** Returns the type identifier of the derived class */
        virtual Char const* type() = 0;

        /** Locks the transfer function for editing */
        void lock() { m_mutex.lock(); }

        /** Unlocks the transfer function after editing */
        void unlock() { m_mutex.unlock(); }

        /** Sets the transfer function resolution */
        void setResolution(Vector3u const& resolution);

        /** Returns the resolution of the transfer function */
        Vector3u const& resolution() const { return m_resolution; }

        /** Returns true if an unprocessed context change has occured */
        bool isDirty() const { return m_isDirty; }

        /** Sets the dirty state of the transfer function */
        void setDirty(bool dirty = true) { m_isDirty = dirty; }

    protected:
        friend RenderController;

        void setClean() { m_isDirty = false; }

        boost::mutex m_mutex; ///< Mutex for scene locking

        Vector3u m_resolution; ///< Transfer function map resolution
        bool     m_isDirty;
    };

    /** 1 Dimensional Transfer Function */
    class VOXS_EXPORT Transfer1D : public Transfer
    {
    public:
        /** Constructs a new transfer function object */
        static std::shared_ptr<Transfer1D> create() { return std::shared_ptr<Transfer1D>(new Transfer1D()); }

        /** Sets the 1D transfer function resolution without requiring a Vec3 */
        void setResolution(size_t resolution)
        {
            Transfer::setResolution(Vector3u(resolution, 1, 1));
        }

        /** Generates the associated transfer map */
        virtual void generateMap(std::shared_ptr<TransferMap> map);

        /** Returns the type of a transfer function */
        virtual Char const* type() { return Transfer1D::typeID(); }

        /** Returns the type of the 1D transfer function */
        static Char const* typeID() { return "Transfer1D"; }

        /** Adds a new node to the transfer function */
        void addNode(std::shared_ptr<Node> node);

        /** Removes a node from the transfer function */
        void removeNode(std::shared_ptr<Node> node);

        /** Returns a linked list of the transfer function nodes */
        std::list<std::shared_ptr<Node>> & nodes() { return m_nodes; }

    private:
        /** Initializes a new transfer function object */
        Transfer1D() { }

        std::list<NodeH> m_nodes;
    };

    /** 2 Dimensional Transfer Function */
    class VOXS_EXPORT Transfer2D : public Transfer
    {
    public:
        /** Constructs a new transfer function object */
        static std::shared_ptr<Transfer2D> create() { return std::shared_ptr<Transfer2D>(new Transfer2D()); }
        
        /** Sets the 1D transfer function resolution without requiring a Vec3 */
        void setResolution(Vector2u resolution)
        {
            Transfer::setResolution(Vector3u(resolution[0], resolution[1], 1));
        }

        /** Generates the associated transfer map */
        virtual void generateMap(std::shared_ptr<TransferMap> map);

        /** Returns the type of a transfer function */
        virtual Char const* type() { return Transfer2D::typeID(); }

        /** Returns the type of the 1D transfer function */
        static Char const* typeID() { return "Transfer2D"; }

        /** Adds a new node to the transfer function */
        void addQuad(std::shared_ptr<Quad> quad);

        /** Removes a node from the transfer function */
        void removeQuad(std::shared_ptr<Quad> quad);

        /** Returns a linked list of the transfer function nodes */
        std::list<std::shared_ptr<Quad>> & quads() { return m_quads; }

    private:
        /** Initializes a new transfer function object */
        Transfer2D() { }

        std::list<std::shared_ptr<Quad>> m_quads;
    };
}

// End definition
#endif // VOX_TRANSFER_H