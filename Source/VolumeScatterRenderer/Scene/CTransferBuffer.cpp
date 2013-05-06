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
    if (m_diffuseArray)
    {
        VOX_CUDA_CHECK(cudaFreeArray(m_diffuseArray));

        m_diffuseArray  = nullptr;
    }
}

// --------------------------------------------------------------------
//  Binds the transfer data to device buffers for use in rendering
// --------------------------------------------------------------------
void CTransferBuffer::setTransfer(std::shared_ptr<TransferMap> transfer)
{
    reset(); // Ensure previous data is released
 
    bindDiffuseBuffer(transfer);
}

// --------------------------------------------------------------------
//  Binds the diffuse trasfer function buffer to a 3d cudaArray
// --------------------------------------------------------------------
void CTransferBuffer::bindDiffuseBuffer(std::shared_ptr<TransferMap> const& transfer)
{
    // Specify the format for volume data access
    auto formatDesc = cudaCreateChannelDesc(
        8, 8, 8, 8, cudaChannelFormatKindUnsigned);

    // Restructure buffer extent 
    cudaExtent extent;
    extent.width  = transfer->diffuse.width();
    extent.height = transfer->diffuse.height();
    extent.depth  = transfer->diffuse.depth();

	// Create a 3d array for transfer function data storage
	VOX_CUDA_CHECK(cudaMalloc3DArray(&m_diffuseArray, &formatDesc, extent));

    // Copy data to device
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr.pitch	     = extent.width*4;
    copyParams.srcPtr.ptr	     = (void*)transfer->diffuse.data();
    copyParams.dstArray	         = m_diffuseArray;
    copyParams.extent	         = extent;
    copyParams.kind		         = cudaMemcpyHostToDevice;
    copyParams.srcPtr.xsize	     = extent.width;
    copyParams.srcPtr.ysize	     = extent.height;
    VOX_CUDA_CHECK(cudaMemcpy3D(&copyParams));
}

} // namespace vox
