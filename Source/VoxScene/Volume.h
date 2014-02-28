/* ===========================================================================

	Project: VoxLib

	Description: Defines a volume class for use by the Renderer

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

// Begin definition
#ifndef VOX_VOLUME_H
#define VOX_VOLUME_H

// Internal Dependencies
#include "VoxScene/Common.h"

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
class VOXS_EXPORT Volume
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
    static unsigned int typeToSize(Type const& type)
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
     * @param offset  The offset of the volume in world coordinates (mm^3)
     * @param type    The format of the volume data
     */
    static std::shared_ptr<Volume> create(
           std::shared_ptr<UInt8>   data      = std::shared_ptr<UInt8>(),
           Vector4s const&          extent    = Vector4s(0),
           Vector4f const&          spacing   = Vector4f(0.0f),
           Vector3f const&          offset    = Vector3f(0.0f),
		   Type                     type      = Type_UInt8 
          )
    { 
        return std::shared_ptr<Volume>(new Volume(data, extent, spacing, offset, type));
    }

	/** 
     * Loads the given data set into the volume
     *
     * @param data    Volume density data in a grid x,y,z,t format
     * @param extent  Volume extent in x,y,z,t dimensions respectively
     * @param spacing Volume spacing in x,y,z,t dimensions respectively
     * @param offset  The offset of the volume in world space
     * @param type    The underlying type of the volume data
     */
    Volume(std::shared_ptr<UInt8>   data,
           Vector4s const&          extent,
           Vector4f const&          spacing,
           Vector3f const&          offset,
		   Type                     type
          );

    /** Destructor */
    ~Volume();

    /** Spacing modifier */     
    void setSpacing(Vector4f const& spacing);

    /** Offset modifier */
    void setOffset(Vector3f const& offset);

    /** Sets the current time slice */
    void setTimeSlice(float timeSlice);

    /** Spacing accessor */     
    Vector4f const& spacing() const;

    /** Extent accessor */      
    Vector4s const& extent() const;
    
    /** Offset accessor */
    Vector3f const& offset() const;
    
    /** Returns the time slice of the volume for display */
    float timeSlice();

    /** Returns the normalized value range of the data for the underlying type */
    Vector2f const& valueRange() const;

    /** Updates the value range of the data */
    void updateRange();

    /** Raw voxel data accessor */
    void* const& at(size_t x, size_t y, size_t z) const;
   
    /** Voxel data accessor */
    float fetchNormalized(size_t x, size_t y, size_t z) const;

    /** Data modifier */ 
    void setData(std::shared_ptr<UInt8> const& data, 
                 Vector4s               const& extent,
				 Type                          type);

    /** Raw voxel data accessor */
    UInt8 * mutableData();

    /** Raw voxel data accessor */
    UInt8 const* data() const;

    /** Returns the format of the data (bytes per voxel) */
    size_t voxelSize() const;

    /** Returns the format of the data (type) */
    Type type() const;

    /** Marks the volume display parameters dirty */
    void setDirty();

    /** Returns true if the volume was changed */
    bool isDirty() const;

    /** Locks the volume data for editing */
    void lock();

    /** Unlocks the volume data */
    void unlock();

private:
    friend RenderController;

    /** Clears the dirty flag */
    void setClean();

    class Impl;
    Impl * m_pImpl;
};

}

// End definition
#endif // VOX_VOLUME_H