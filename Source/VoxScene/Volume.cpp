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
#include "VoxLib/Core/Common.h"
#include "VoxLib/IO/Resource.h"

namespace vox {

    namespace {
    namespace filescope {

        // --------------------------------------------------------------------
        //  Computes the maximum normalized range of values within the voxel
        //  type which is occupied by the volume data.
        // --------------------------------------------------------------------
        template<typename T> Vector2f maxValueRange(size_t elements, UInt8 const* raw)
        {
            if (elements == 0) return Vector2f();

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
//  Volume implementation class
// ----------------------------------------------------------------------------
class Volume::Impl
{
public:
    // ----------------------------------------------------------------------------
    //  Wraps a volume data buffer to construct a new volume object
    // ----------------------------------------------------------------------------
    Impl::Impl(std::shared_ptr<UInt8> data, Vector4s const& extent, 
               Vector4f const& spacing, Vector3f const& offset, Type type) : 
        m_data(data), 
        m_extent(extent), 
        m_spacing(spacing), 
        m_type(type), 
        m_offset(offset), 
        m_timeSlice(0.f),
        m_isDataDirty(true)
    { 
        if (!m_data)
        {
            auto bytes = typeToSize(type) * m_extent.fold(&mul);
            m_data = vox::makeSharedArray(bytes);
            memset(m_data.get(), 0, bytes);
            m_range = Vector2f(0.0f);
        }
        else updateRange();
    }

    // ----------------------------------------------------------------------------
    //  Sets the volume data
    // ----------------------------------------------------------------------------
    void setData(std::shared_ptr<UInt8> const& data, Vector4s const& extent, Type type)
    {
        m_data   = data; 
        m_extent = extent; 
        m_type   = type;

        updateRange();
    }

    // ----------------------------------------------------------------------------
    //  Updates the value range for the volume voxels
    // ----------------------------------------------------------------------------
    void updateRange()
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
    //  Returns a pointer to the voxel at the specified index
    // ----------------------------------------------------------------------------
    void* at(size_t x, size_t y, size_t z, size_t t)
    {
        size_t i = x + y * m_extent[0] + z * m_extent[0] * m_extent[1] + t * m_extent[0] * m_extent[1] * m_extent[2];
        
        return m_data.get() + i * typeToSize(m_type);
    }

    // ----------------------------------------------------------------------------
    //  Fetches the normalized value at a voxel
    // ----------------------------------------------------------------------------
    float fetchNormalized(size_t x, size_t y, size_t z) const
    {
        size_t i = x + y * m_extent[0] + z * m_extent[0] * m_extent[1];
        
        float sample;
        switch (m_type)
        {
            case Volume::Type_Int8:   sample = (static_cast<float>(m_data.get()[i]) + 0.5f) / static_cast<float>(std::numeric_limits<Int8>::max()); break;
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
    
    // ----------------------------------------------------------------------------
    //  Clones the volume into another object (shallow copy for data)
    // ----------------------------------------------------------------------------
    void clone(Volume & volume)
    {
        volume.m_pImpl->m_isDataDirty = !(volume.m_pImpl->m_data.get() == m_data.get());
        volume.m_pImpl->m_data        = m_data;
        volume.m_pImpl->m_timeSlice   = m_timeSlice;
        volume.m_pImpl->m_range       = m_range;
        volume.m_pImpl->m_offset      = m_offset;
        volume.m_pImpl->m_spacing     = m_spacing; 
        volume.m_pImpl->m_extent      = m_extent;
        volume.m_pImpl->m_type        = m_type;
    }
    
    // ----------------------------------------------------------------------------
    //  Interpolates between volume settings
    // ----------------------------------------------------------------------------
    std::shared_ptr<Volume> interp(std::shared_ptr<Volume> k2, float f)
    {
        auto volume = Volume::create();
        volume->m_pImpl->m_isDataDirty = false;
        volume->m_pImpl->m_data      = m_data;
        volume->m_pImpl->m_timeSlice = m_timeSlice * (1.f - f) + k2->m_pImpl->m_timeSlice * f; 
        volume->m_pImpl->m_range     = m_range;
        volume->m_pImpl->m_offset    = m_offset * (1.f - f) + k2->m_pImpl->m_offset * f; 
        volume->m_pImpl->m_spacing   = m_spacing * (1.f - f) + k2->m_pImpl->m_spacing * f; 
        volume->m_pImpl->m_extent    = m_extent;
        volume->m_pImpl->m_type      = m_type;

        return volume;
    }

public:
    std::shared_ptr<UInt8> m_data; ///< Pointer to volume data
    
    float m_timeSlice; ///< The current time slice to display
    
    ResourceId diskCache; ///< Local disk cache of this volume dataset

    bool m_isDataDirty; ///< Volume binary data dirty flag

    Vector2f m_range;       ///< Volume value range (normalized to type)
    Vector3f m_offset;      ///< Volume offset (mm)
    Vector4f m_spacing;     ///< Spacing between voxels (mm)
    Vector4s m_extent;      ///< Size of volume in voxels
    Type     m_type;        ///< Volume data type
};

// ----------------------------------------------------------------------------
//  Redirect to implementation
// ----------------------------------------------------------------------------
Volume::Volume(std::shared_ptr<UInt8> data, Vector4s const& extent, 
               Vector4f const& spacing, Vector3f const& offset, Type type) : 
    m_pImpl(new Impl(data, extent, spacing, offset, type)) { }
Volume::~Volume() { delete m_pImpl; }
void Volume::setData(std::shared_ptr<UInt8> const& data, Vector4s const& extent, Type type) 
    { m_pImpl->setData(data, extent, type); }
void Volume::updateRange() { m_pImpl->updateRange(); }
float Volume::fetchNormalized(size_t x, size_t y, size_t z) const { return m_pImpl->fetchNormalized(x, y, z); }
void * Volume::at(size_t x, size_t y, size_t z, size_t t) { return m_pImpl->at(x, y, z, t); }

// ----------------------------------------------------------------------------
//  Getters/Setters
// ----------------------------------------------------------------------------
UInt8 *         Volume::mutableData()      { return m_pImpl->m_data.get(); }
UInt8 const*    Volume::data() const       { return m_pImpl->m_data.get(); }
size_t          Volume::voxelSize() const  { return typeToSize(m_pImpl->m_type); }
Volume::Type    Volume::type() const       { return m_pImpl->m_type; }
Vector4f const& Volume::spacing() const     { return m_pImpl->m_spacing; } 
Vector4s const& Volume::extent() const      { return m_pImpl->m_extent; }  
Vector3f const& Volume::offset() const      { return m_pImpl->m_offset; }
float           Volume::timeSlice() const   { return m_pImpl->m_timeSlice; }
Vector2f const& Volume::valueRange() const  { return m_pImpl->m_range; }
void            Volume::setDataDirty()      { m_pImpl->m_isDataDirty = true; setDirty(); }
void            Volume::setClean()          { Object::setClean(); m_pImpl->m_isDataDirty = false; }
bool            Volume::isDataDirty() const { return m_pImpl->m_isDataDirty; }
void            Volume::setSpacing(Vector4f const& spacing) { m_pImpl->m_spacing = spacing; }
void            Volume::setOffset(Vector3f const& offset)   { m_pImpl->m_offset = offset; }
void            Volume::setTimeSlice(float timeSlice)       { m_pImpl->m_timeSlice = timeSlice; }
void            Volume::clone(Volume & volume) { m_pImpl->clone(volume); }
std::shared_ptr<Volume> Volume::interp(std::shared_ptr<Volume> k2, float f) { return m_pImpl->interp(k2, f); }

} // namespace vox