/* ===========================================================================

	Project: GPU based Volume Scatter Renderer
    
	Description: Wraps the management of a CUDA 3D volume buffer

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
#ifndef VSR_CVOLUME_BUFFER_H
#define VSR_CVOLUME_BUFFER_H

// Common Library Header
#include "VolumeScatterRenderer/Core/Common.h"

// VoxLib Dependencies
#include "VoxLib/Core/Geometry.h"
#include "VoxScene/Volume.h"

// API namespace
namespace vox
{

/** Rendering Volume Class */
class CVolumeBuffer
{
public:
    /** Initializes the buffer for use */
    VOX_HOST void init() { m_handle = nullptr; }

    /** Deallocates the device memory buffer */
    VOX_HOST void reset();

    /** Sets the VolumeBuffer's data content */
    VOX_HOST void setVolume(std::shared_ptr<Volume> volume);

    /** Returns the size of the volume data */
    VOX_HOST_DEVICE inline Vector3f const& size() const { return m_size; }

    /** Returns the offset of the volume data */
    VOX_HOST_DEVICE inline Vector3f const& offset() const { return m_offset; }

    /** Returns the inverse spacing of the volume data */
    VOX_HOST_DEVICE inline Vector3f const& invSpacing() const { return m_invSpacing; }

    /** Returns the currently loaded time slice of the volume */
    VOX_HOST_DEVICE float timeSlice() { return m_timeSlice; }

    /** Returns the cudaArray storing the volume data */
    VOX_HOST inline cudaArray const* handle() const { return m_handle; }

    /** Returns the volume format (bytes per voxel) */
    VOX_HOST_DEVICE inline Volume::Type type() const { return m_type; }

    /** Returns the channel format descriptor */
    VOX_HOST cudaChannelFormatDesc const& formatDescriptor() const { return m_format; }

    /** Normalizes a data value based on the volume data range */
    VOX_HOST_DEVICE float normalizeSample(float sample)
    {
        return (sample - m_dataMin) * m_invRange;
    }

private:
    /** Loads the volume data into the buffer */
    VOX_HOST void loadBuffer(std::shared_ptr<Volume> volume);

private:
    Volume::Type   m_type;         ///< Format of volume data
    Vector3f       m_size;         ///< Size of the volume (mm)
    Vector3f       m_invSpacing;   ///< Inverse of spacing between samples (/mm)
    Vector3f       m_offset;       ///< Volume position offset (mm)
    cudaExtent     m_extent;       ///< Extent of the volume data
    cudaArray *    m_handle;       ///< Handle to volume data array on device
    float          m_invRange;     ///< The range of the volume data values (for normalization)
    float          m_dataMin;      ///< The minimum of the volume data values (for normalization)
    float          m_timeSlice;    ///< The current time slice of the volume

    // :TODO: Store affine transform from world->volume for volume trace Ray3f

    cudaChannelFormatDesc m_format; ///< Texture channel format
};

}

// End definition
#endif // VSR_CVOLUME_BUFFER_H