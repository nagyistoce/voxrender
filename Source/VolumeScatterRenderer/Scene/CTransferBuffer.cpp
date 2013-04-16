/* ===========================================================================

	Project: GPU based Volume Scatter Renderer
    
	Description: Wraps the management of a CUDA transfer function buffer

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
#include "CTransferBuffer.h"

// Include Dependencies
#include "VoxLib/Core/Format.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/Error/CudaError.h"

namespace vox {
    
// --------------------------------------------------------------------
//  Releases any device memory associated with the volume
// --------------------------------------------------------------------
void CTransferBuffer::reset()
{
    if (m_emissionArray)
    {
        VOX_CUDA_CHECK(cudaFreeArray(m_emissionArray));
        VOX_CUDA_CHECK(cudaFreeArray(m_diffuseArray));

        m_emissionArray = nullptr;
        m_diffuseArray  = nullptr;
    }
}

// --------------------------------------------------------------------
//  Binds the volume data to device buffers for use in rendering
// --------------------------------------------------------------------
void CTransferBuffer::setTransfer(std::shared_ptr<Transfer> transfer)
{
    reset(); // Ensure previous data is released
    
    // Record transfer function resolution
    auto resolution = transfer->resolution();
    m_extent.width  = resolution[0];
    m_extent.height = resolution[1];
    m_extent.depth  = resolution[2];
    /*
    // Specify the format for volume data access
    auto formatDesc = cudaCreateChannelDesc(
        32, 32, 32, 32, cudaChannelFormatKindFloat);

	// Create a 3d array for volume data storage
	VOX_CUDA_CHECK(cudaMalloc3DArray(&m_emissionArray, &formatDesc, m_extent));
	VOX_CUDA_CHECK(cudaMalloc3DArray(&m_absorptionArray, &formatDesc, m_extent));

    // Copy emission data to device
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr.pitch	     = m_extent.width*sizeof(Vector3f);
    copyParams.srcPtr.ptr	     = (void*);
    copyParams.dstArray	         = m_emissionArray;
    copyParams.extent	         = m_extent;
    copyParams.kind		         = cudaMemcpyHostToDevice;
    copyParams.srcPtr.xsize	     = m_extent.width;
    copyParams.srcPtr.ysize	     = m_extent.height;
    VOX_CUDA_CHECK(cudaMemcpy3D(&copyParams));
    
    // Copy emission data to device
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr.pitch	     = m_extent.width*sizeof(Vector3f);
    copyParams.srcPtr.ptr	     = (void*);
    copyParams.dstArray	         = m_emissionArray;
    copyParams.extent	         = m_extent;
    copyParams.kind		         = cudaMemcpyHostToDevice;
    copyParams.srcPtr.xsize	     = m_extent.width;
    copyParams.srcPtr.ysize	     = m_extent.height;
    VOX_CUDA_CHECK(cudaMemcpy3D(&copyParams));
    */
}

} // namespace vox
