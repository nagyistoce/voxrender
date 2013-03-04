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
class CVolume
{
public:
    VOX_HOST_DEVICE CVolume() { }

    /** Synchronizes the camera */
    VOX_HOST void synchronize(std::shared_ptr<Volume> const& volume);

    /** Returns the extent of the volume data */
    VOX_HOST_DEVICE inline Vector3u const& extent() const { return m_extent; }

    /** Array style access element access */
    /*VOX_HOST_DEVICE inline UInt8 const& operator[](Vector3u i) const 
    { 
        return coord[i]; 
    }*/

private:
    Vector3u    m_extent;
    Vector3f    m_spacing;
    cudaArray * m_data;
};

}

// End definition
#endif // CR_CVOLUME_H