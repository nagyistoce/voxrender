/* ===========================================================================

    Project: Volume Scatter Renderer

    Description: Defines a clipping plane element

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

// Include Header
#include "CClipPlane.h"

// Include Dependencies
#include "VolumeScatterRenderer/Kernels/RenderKernel.h"
#include "VoxLib/Error/CudaError.h"

namespace vox {

// ----------------------------------------------------------------------------
//  Constructs a callback function for performing clipping for the given group
// ----------------------------------------------------------------------------
CClipPlane::CClipPlane(std::shared_ptr<Plane> plane) : m_plane(plane), m_handle(nullptr)
{
}

// ----------------------------------------------------------------------------
//  Creates a Clipper object in device memory for use in clipping operations
// ----------------------------------------------------------------------------
CClipGeometry::Clipper * CClipPlane::clipper()  
{ 
    if (!m_handle)
    {
        VOX_CUDA_CHECK(cudaMalloc((void**)&m_handle, sizeof(Data)));

        Data data;
        data.clipper.func = RenderKernel::getCClipPlaneFunc();
        data.normal       = m_plane->normal();
        data.distance     = m_plane->distance();

        VOX_CUDA_CHECK(cudaMemcpy(m_handle, &data, sizeof(Data), cudaMemcpyHostToDevice));
    }

    return reinterpret_cast<Clipper*>(m_handle); 
}

} // namespace vox