/* ===========================================================================

	Project: Transfer - Transfer Function

	Description: Transfer function applied to volume dataset

    Copyright (C) 2012 Lucas Sherman

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
#include "VoxLib/Core/Common.h"
#include "VoxLib/Core/Geometry/Color.h"
#include "VoxLib/Core/Geometry/Vector.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/Error/ErrorCodes.h"
#include "VoxLib/Core/Functors.h"

// API Namespace
namespace vox
{
    typedef Vector<float,3> Color; // :TODO: Color header

    class Transfer;

    /**
     * Transfer function node
     */
    class VOX_EXPORT Node 
    {
    public:
        Node( ) :
          m_dirty(false)
        {
        }

        void setPosition( int dim, float position )
        {
            m_position[dim] = clamp( position, 0.f, 1.f ); m_dirty = true;
        }

        void setOpacity( float opacity ) { m_opacity = opacity; m_dirty = true; }
        void setGloss( float glossiness ) { m_glossiness = glossiness; m_dirty = false; } 
 
        void DiffuseColor( Color const& color )  { m_colorDiffuse = color; m_dirty = true; }
        void SpecularColor( Color const& color ) { m_colorSpecular = color; m_dirty = true; }
        void EmissiveColor( Color const& color ) { m_colorEmission = color; m_dirty = true; }
        
        float position( int dim ) const { return m_position[dim]; }

        float opacity( ) const { return m_opacity; }
        float gloss( ) const { return m_glossiness; }

        Color const& diffuse( )  const { return m_colorDiffuse; }
        Color const& specular( ) const { return m_colorSpecular; }
        Color const& emissive( ) const { return m_colorEmission; }

    private:
        friend Transfer; 

	    Vector3f m_position;         ///< Normalized position vector

        std::string m_userData;     ///< Binary user data
	    float m_opacity;		    ///< Opacity of material
	    float m_glossiness;	        ///< Glossiness of material (specular)
	    Color m_colorDiffuse;	    ///< Diffuse color of material
	    Color m_colorSpecular;	    ///< Specular color of material
	    Color m_colorEmission;	    ///< Emissive color of material

        bool  m_dirty;  ///< Dirty flag
    };

    /**
     * Transfer Function Types
     */
    enum TransferType
    {
        TransferType_1D = 0,
        TransferType_2D = 1,
        TransferType_3D = 2
    };

    /**
     * Transfer Function
     */
    class VOX_EXPORT Transfer
    {
    public:
        /**
         * A tranfer region
         *
         * This object contains a collection of nodes which define a region 
         * of the volumes data set to which a transfer function will be applied.
         */
        struct Region
        {
            Node* nodes;
        };

        /** Returns the dimensions of the transfer function */
        virtual Vector3u dimensions() = 0;

        /** Returns a map of the transfer function content */
        virtual std::shared_ptr<ColorLabxHdr> map() = 0;

        /** Returns the dimension of the transfer function */
        inline TransferType type( ) const { return m_type; }

        /** Returns the list of regions composing the transfer function */
        inline std::list<Region*> const& regions( ) { return m_regions; }

        /** Adds a new region to the transfer function */
        Region* addRegion( Region* region = nullptr );

        /** Removes a region from the transfer function */ 
        void removeRegion( Region* region );

    private:
        TransferType m_type;

        std::list<Region*> m_regions;

        bool m_dirty;
    };
}

// End definition
#endif // VOX_TRANSFER_H