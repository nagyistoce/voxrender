/* ===========================================================================

	Project: CUDA Renderer - Rendering Camera

	Description: Defines a 3D Camera for use in rendering

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
#ifndef CR_CVOLUME_H
#define CR_CVOLUME_H

// Common Library Header
#include "CudaRenderer/Core/Common.h"

// VoxLib Dependencies
#include "VoxLib/Core/Geometry.h"
#include "VoxLib/Scene/Volume.h"

// API namespace
namespace vox
{

/** Rendering Volume Class */
class CVolumeBuffer
{
public:
    void init();
    void free();

    /** Synchronizes the camera */
    VOX_HOST void synchronize(std::shared_ptr<Volume> const& volume);

    /** Returns the size of the volume data */
    VOX_HOST_DEVICE inline Vector3f const& size() const { return m_size; }

    /** Returns the inverse spacing of the volume data */
    VOX_HOST_DEVICE inline Vector3f const& invSpacing() const { return m_invSpacing; }

    /** Returns the cudaArray storing the volume data */
    VOX_HOST inline cudaArray * handle() { return m_handle; }

    /** Returns the volume format (bytes per voxel) */

private:
    Vector3f    m_size;
    Vector3f    m_invSpacing;
    cudaArray * m_handle;
    size_t      m_format;
};

}

// End definition
#endif // CR_CVOLUME_H