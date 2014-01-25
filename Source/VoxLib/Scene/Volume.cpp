/* ===========================================================================

	Project: VoxLib

	Description: Defines a 3D volume class

    Copyright (C) 2012-2014 Lucas Sherman

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
#include "Volume.h"

// Standard Includes
#include <iostream>
#include <fstream>

namespace vox {

    namespace {
    namespace filescope {

        // --------------------------------------------------------------------
        //  Computes the maximum normalized range of values within the voxel
        //  type which is occupied by the volume data.
        // --------------------------------------------------------------------
        template<typename T> Vector2f maxValueRange(size_t elements, UInt8 const* raw)
        {
            T const* data = reinterpret_cast<T const*>(raw);
            T low  = *data;
            T high = *data;

            for (size_t i = 1; i < elements; i++)
            {
                if (low > *data) low = *data;
                else if (high < *data) high = *data;

                data++;
            }

            Vector2f result = Vector2f(static_cast<float>(low)+0.5f, static_cast<float>(high)+0.5f);

            return result / static_cast<float>(std::numeric_limits<T>::max());
        }

    } // namespace filescope
    } // namespace anonymous
    
// ----------------------------------------------------------------------------
//  Wraps a volume data buffer to construct a new volume object
// ----------------------------------------------------------------------------
Volume::Volume(std::shared_ptr<UInt8> data, Vector4u const& extent, 
               Vector4f const& spacing, Vector3f const& offset, Type type) : 
    m_data(data), 
    m_extent(extent), 
    m_spacing(spacing), 
    m_type(type), 
    m_offset(offset), 
    m_isDirty(false)
{ 
    updateRange();
}

// ----------------------------------------------------------------------------
//  Sets the volume data
// ----------------------------------------------------------------------------
void Volume::setData(std::shared_ptr<UInt8> const& data, Vector4u const& extent, Type type)
{
    m_data   = data; 
    m_extent = extent; 
    m_type   = type;

    updateRange();
}

// ----------------------------------------------------------------------------
//  Updates the value range for the volume voxels
// ----------------------------------------------------------------------------
void Volume::updateRange()
{
    size_t       elems = m_extent.fold<size_t>(1, &mul);
    UInt8 const* ptr   = m_data.get();

    switch (m_type)
    {
        case Volume::Type_Int8:   m_range = filescope::maxValueRange<Int8>(elems, ptr); break;
        case Volume::Type_UInt8:  m_range = filescope::maxValueRange<UInt8>(elems, ptr); break;
        case Volume::Type_UInt16: m_range = filescope::maxValueRange<UInt16>(elems, ptr); break;
        case Volume::Type_Int16:  m_range = filescope::maxValueRange<Int16>(elems, ptr); break;
        default:
            throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                format("Unsupported volume data type (%1%)", Volume::typeToString(m_type)),
                Error_NotImplemented);
    }
}

// ----------------------------------------------------------------------------
//  Fetches the normalized value at a voxel
// ----------------------------------------------------------------------------
float Volume::fetchNormalized(size_t x, size_t y, size_t z) const
{
    size_t i = x + y * m_extent[0] + z * m_extent[0] * m_extent[1];
        
    float sample;
    switch (m_type)
    {
        case Volume::Type_Int8:  sample = (static_cast<float>(m_data.get()[i]) + 0.5f) / static_cast<float>(std::numeric_limits<Int8>::max()); break;
        case Volume::Type_UInt8:  sample = (static_cast<float>(m_data.get()[i]) + 0.5f) / static_cast<float>(std::numeric_limits<UInt8>::max()); break;
        case Volume::Type_UInt16: sample = (static_cast<float>(reinterpret_cast<UInt16 const*>(m_data.get())[i]) + 0.5f) / static_cast<float>(std::numeric_limits<UInt16>::max());; break;
        case Volume::Type_Int16:  sample = (static_cast<float>(reinterpret_cast<Int16 const*>(m_data.get())[i]) + 0.5f) / static_cast<float>(std::numeric_limits<Int16>::max());; break;
        default:
            throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                format("Unsupported volume data type (%1%)", Volume::typeToString(m_type)),
                Error_NotImplemented);
    }
        
    return (sample - m_range[0]) / (m_range[1] - m_range[0]);
}

} // namespace vox