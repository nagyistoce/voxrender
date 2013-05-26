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
#include "VoxLib/Core/Format.h"
#include "VoxLib/Core/Geometry.h"
#include "VoxLib/Core/Types.h"
#include "VoxLib/Error/Error.h"

// API namespace
namespace vox
{
    
class RenderController;

/** 3D/4D Volume Class */
class VOX_EXPORT Volume
{
public:
    /** Volume data formats */
    enum Type
    {
        Type_Begin,                   ///< Begin iterator for Format enumeration
        Type_UInt8  = Type_Begin,     ///< Unsigned 8-bit integer
        Type_Int8,                    ///< Signed 8-bit integer
        Type_UInt16,                  ///< Unsigned 16-bit integer
        Type_Int16,                   ///< Signed 16-bit integer
        Type_Float32,                 ///< 32-bit floating point numeric
        Type_Float64,                 ///< 64-bit floating point numeric
        Type_End                      ///< End iterator for Format enumeration
    };

    /** Volume data format size conversion (runtime) */
    static size_t typeToSize(Type const& type)
    {
        switch(type)
        {
        case Type_UInt8: case Type_Int8: return 1;
        case Type_UInt16: case Type_Int16: return 2;
        case Type_Float32: return 4;
        case Type_Float64: return 8;
        default: 
            throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
                        format("Invalid volume data type (%1%)", type), 
                        Error_Range);
        }
    }

    /** Volume data format string conversion (runtime) */
    static Char const* typeToString(Type const& type)
    {
        switch(type)
        {
        case Type_UInt8:   return "uint8";
        case Type_Int8:    return "int8";
        case Type_UInt16:  return "uint16";
        case Type_Int16:   return "int16";
        case Type_Float32: return "float32";
        case Type_Float64: return "float64";
        default: 
            throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
                        format("Invalid volume data type (%1%)", type), 
                        Error_Range);
        }
    }

	/** 
     * Loads the given data set into the volume
     *
     * @param data    Volume density data in a grid x,y,z,t format
     * @param extent  Volume extent in x,y,z,t dimensions respectively
     * @param spacing Volume spacing in x,y,z,t dimensions respectively
     */
    Volume(std::shared_ptr<UInt8>   data      = std::shared_ptr<UInt8>(),
           Vector4u const&          extent    = Vector4u(0),
           Vector4f const&          spacing   = Vector4f(0.0f),
           Vector3f const&          offset    = Vector3f(0.0f),
		   Type                     type      = Type_UInt8 
          ) : 
        m_data(data), m_extent(extent), m_spacing(spacing), m_type(type), m_offset(offset)
    { 
    }

    /** Spacing modifier */     
    void setSpacing(Vector4f const& spacing) { m_spacing = spacing; m_contextChanged = true; }

    /** Offset modifier */
    void setOffset(Vector3f const& offset) { m_offset = offset; m_contextChanged = true; }

    /** Spacing accessor */     
    Vector4f const& spacing() const { return m_spacing; } 

    /** Extent accessor */      
    Vector4u const& extent() const { return m_extent; }  
    
    /** Offset accessor */
    Vector3f const& offset() const { return m_offset; }

    /** Raw voxel data accessor */
    void* const& at(size_t x, size_t y, size_t z) const;
   
    /** Data modifier */ 
    void setData(std::shared_ptr<UInt8> const& data, 
                 Vector4u               const& extent,
				 Type                          type)
    {
        m_data = data; m_extent = extent; m_type = type;
    }

    /** Raw voxel data accessor */
    UInt8 * mutableData() { return m_data.get(); }

    /** Raw voxel data accessor */
    UInt8 const* data() const { return m_data.get(); }

    /** Returns the format of the data (bytes per voxel) */
    size_t voxelSize() const { return typeToSize(m_type); }

    /** Returns the format of the data (type) */
    Type type() const { return m_type; }

    /** Returns true if the volume was changed */
    bool isDirty() const { return m_contextChanged; }

private:
    friend RenderController;

    bool m_contextChanged; ///< Context change flag

    std::shared_ptr<UInt8> m_data; ///< Pointer to volume data

    Vector3f m_offset;      ///< Volume offset (mm)
    Vector4f m_spacing;     ///< Spacing between voxels (mm)
    Vector4u m_extent;      ///< Size of volume in voxels
    Type     m_type;        ///< Volume data type
};

}

// End definition
#endif // VOX_VOLUME_H