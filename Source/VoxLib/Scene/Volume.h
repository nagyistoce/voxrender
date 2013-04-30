/* ===========================================================================

	Project: VoxRender - Volume

	Description: Defines a volume class for use by the Renderer

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
#ifndef VOX_VOLUME_H
#define VOX_VOLUME_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Geometry.h"

// API namespace
namespace vox
{
    
class RenderController;
    
/** 3D/4D Volume Class */
class VOX_EXPORT Volume
{
public:
    typedef UInt8 ValueType; ///< Voxel value type

	/** 
     * Loads the given data set into the volume
     *
     * @param data    Volume density data in a grid x,y,z,t format
     * @param extent  Volume extent in x,y,z,t dimensions respectively
     * @param spacing Volume spacing in x,y,z,t dimensions respectively
     */
    Volume(std::shared_ptr<UInt8> data    = std::shared_ptr<UInt8>(),
           const Vector<size_t,4>&  extent  = Vector<size_t,4>(0),
           const Vector<float,4>&   spacing = Vector<float,4>(0)
          ) : 
        m_data(data), m_extent(extent), m_spacing(spacing)
    { 
        m_voxelSize = 1; // :TODO:
    }

    /** Spacing modifier */     
    void setSpacing(Vector<float,4> const& spacing) { m_spacing = spacing; m_contextChanged = true; }

    /** Extent modifier */     
    void setExtent(Vector<size_t,4> const& extent) 
    { 
        m_extent = extent; m_contextChanged = true; 
    }

    /** Spacing accessor */     
    Vector<float,4> const& spacing() const { return m_spacing; } 

    /** Extent accessor */      
    Vector<size_t,4> const& extent() const { return m_extent; }  
    
    /** Raw voxel data accessor */
    ValueType const& at(size_t x, size_t y, size_t z) const;
   
    /** Data modifier */ 
    void setData(std::shared_ptr<UInt8> const& data, 
                 Vector<size_t,4> const& extent)
    {
        m_data = data; m_extent = extent;
    }

    /** Raw voxel data accessor */
    ValueType * mutableData() { return m_data.get(); }

    /** Raw voxel data accessor */
    ValueType const* data() const { return m_data.get(); }

    /** Returns the format of the data (bytes per voxel) */
    size_t voxelSize() const { return m_voxelSize; }

    /** Returns true if the volume was changed */
    bool isDirty() const { return m_contextChanged; }

private:
    friend RenderController;

    bool m_contextChanged; ///< Context change flag

    std::shared_ptr<UInt8> m_data; ///< Pointer to volume data

    Vector4f m_spacing;     ///< Spacing between voxels (mm)
    Vector4u m_extent;      ///< Size of volume in voxels
    size_t   m_voxelSize;   ///< Volume voxel size (bytes per voxel)
};

}

// End definition
#endif // VOX_VOLUME_H