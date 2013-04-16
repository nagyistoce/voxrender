/* ===========================================================================

	Project: GPU based Volume Scatter Renderer
    
	Description: Wraps the management of a CUDA 3D volume buffer

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

// Include Header
#include "CVolumeBuffer.h"

// Include Dependencies
#include "VoxLib/Core/Format.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/Error/CudaError.h"

namespace vox {
    
// --------------------------------------------------------------------
//  Releases any device memory associated with the volume
// --------------------------------------------------------------------
void CVolumeBuffer::reset()
{
    if (m_handle)
    {
        VOX_CUDA_CHECK(cudaFreeArray(m_handle));

        m_handle = nullptr;
    }
}

// --------------------------------------------------------------------
//  Binds the volume data to device buffers for use in rendering
// --------------------------------------------------------------------
void CVolumeBuffer::setVolume(std::shared_ptr<Volume> volume)
{
    reset(); // Ensure previous data is released

    // Verify that the volume format is acceptable
    m_voxelSize = volume->voxelSize();
    if (m_voxelSize != sizeof(unsigned char))
    {
        throw Error(__FILE__, __LINE__, VSR_LOG_CATEGORY,
            format("Unsupported volume voxel size (%1% bits)", m_voxelSize),
            Error_BadFormat);
    }
    
    // Volume parameters
    auto spacing = volume->spacing();
    auto extent  = volume->extent();

    // Record volume extent 
    m_extent.width  = extent[0];
    m_extent.height = extent[1];
    m_extent.depth  = extent[2];

    // Compute volume bounding box
    m_size[0] = extent[0] * spacing[0];
    m_size[1] = extent[1] * spacing[1];
    m_size[2] = extent[2] * spacing[2];

    // Compute inverse spacing vector
    m_invSpacing[0] = 1.0f / spacing[0];
    m_invSpacing[1] = 1.0f / spacing[1];
    m_invSpacing[2] = 1.0f / spacing[2];

    // Specify the format for volume data access
    auto texFormatDesc = cudaCreateChannelDesc(
        m_voxelSize*8, 0, 0, 0, cudaChannelFormatKindUnsigned);

	// Create a 3d array for volume data storage
	VOX_CUDA_CHECK(cudaMalloc3DArray(&m_handle, &texFormatDesc, m_extent));

    // Copy volume data to device
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr.pitch	     = m_extent.width*m_voxelSize;
    copyParams.srcPtr.ptr	     = (void*)volume->data();
    copyParams.dstArray	         = m_handle;
    copyParams.extent	         = m_extent;
    copyParams.kind		         = cudaMemcpyHostToDevice;
    copyParams.srcPtr.xsize	     = m_extent.width;
    copyParams.srcPtr.ysize	     = m_extent.height;
    VOX_CUDA_CHECK(cudaMemcpy3D(&copyParams));
}

} // namespace vox
