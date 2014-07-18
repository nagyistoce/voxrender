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
#include "VoxScene/TransferMap.h"

namespace vox {
    
// --------------------------------------------------------------------
//  Releases any device memory associated with the volume
// --------------------------------------------------------------------
void CTransferBuffer::reset()
{
    if (m_diffuseArray)  { cudaFreeArray(m_diffuseArray);  m_diffuseArray  = nullptr; }
    if (m_opacityArray)  { cudaFreeArray(m_opacityArray);  m_opacityArray  = nullptr; }
    if (m_specularArray) { cudaFreeArray(m_specularArray); m_specularArray = nullptr; }
    if (m_emissiveArray) { cudaFreeArray(m_emissiveArray); m_emissiveArray = nullptr; }
}

// --------------------------------------------------------------------
//  Binds the transfer data to device buffers for use in rendering
// --------------------------------------------------------------------
void CTransferBuffer::setTransfer(std::shared_ptr<TransferMap> transfer)
{
    reset(); // Ensure previous data is released
 
    bindDiffuseBuffer(transfer);
    bindOpacityBuffer(transfer);
    bindSpecularBuffer(transfer);
    bindEmissiveBuffer(transfer);
}

// --------------------------------------------------------------------
//  Binds the opacity transfer function buffer to a 3d cudaArray
// --------------------------------------------------------------------
void CTransferBuffer::bindOpacityBuffer(std::shared_ptr<TransferMap> const& transfer)
{
    // Specify the format for volume data access
    auto formatDesc = cudaCreateChannelDesc(
        32, 0, 0, 0, cudaChannelFormatKindFloat);

    // Restructure buffer extent 
    auto opacity = transfer->opacity();
    cudaExtent extent;
    extent.width  = opacity.width();
    extent.height = opacity.height();
    extent.depth  = opacity.depth();

	// Create a 3d array for transfer function data storage
	VOX_CUDA_CHECK(cudaMalloc3DArray(&m_opacityArray, &formatDesc, extent));

    // Copy data to device
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr.pitch	     = extent.width*sizeof(float);
    copyParams.srcPtr.ptr	     = (void*)opacity.data();
    copyParams.dstArray	         = m_opacityArray;
    copyParams.extent	         = extent;
    copyParams.kind		         = cudaMemcpyHostToDevice;
    copyParams.srcPtr.xsize	     = extent.width;
    copyParams.srcPtr.ysize	     = extent.height;
    VOX_CUDA_CHECK(cudaMemcpy3D(&copyParams));
}

// --------------------------------------------------------------------
//  Binds the diffuse transfer function buffer to a 3d cudaArray
// --------------------------------------------------------------------
void CTransferBuffer::bindDiffuseBuffer(std::shared_ptr<TransferMap> const& transfer)
{
    // Specify the format for volume data access
    auto formatDesc = cudaCreateChannelDesc(
        8, 8, 8, 8, cudaChannelFormatKindUnsigned);

    // Restructure buffer extent 
    auto diffuse = transfer->diffuse();
    cudaExtent extent;
    extent.width  = diffuse.width();
    extent.height = diffuse.height();
    extent.depth  = diffuse.depth();

	// Create a 3d array for transfer function data storage
	VOX_CUDA_CHECK(cudaMalloc3DArray(&m_diffuseArray, &formatDesc, extent));

    // Copy data to device
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr.pitch	     = extent.width*4;
    copyParams.srcPtr.ptr	     = (void*)diffuse.data();
    copyParams.dstArray	         = m_diffuseArray;
    copyParams.extent	         = extent;
    copyParams.kind		         = cudaMemcpyHostToDevice;
    copyParams.srcPtr.xsize	     = extent.width;
    copyParams.srcPtr.ysize	     = extent.height;
    VOX_CUDA_CHECK(cudaMemcpy3D(&copyParams));
}

// --------------------------------------------------------------------
//  Binds the diffuse transfer function buffer to a 3d cudaArray
// --------------------------------------------------------------------
void CTransferBuffer::bindSpecularBuffer(std::shared_ptr<TransferMap> const& transfer)
{
    // Specify the format for volume data access
    auto formatDesc = cudaCreateChannelDesc(
        8, 8, 8, 8, cudaChannelFormatKindUnsigned);

    // Restructure buffer extent 
    auto specular = transfer->specular();
    cudaExtent extent;
    extent.width  = specular.width();
    extent.height = specular.height();
    extent.depth  = specular.depth();

	// Create a 3d array for transfer function data storage
	VOX_CUDA_CHECK(cudaMalloc3DArray(&m_specularArray, &formatDesc, extent));

    // Copy data to device
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr.pitch	     = extent.width*4;
    copyParams.srcPtr.ptr	     = (void*)specular.data();
    copyParams.dstArray	         = m_specularArray;
    copyParams.extent	         = extent;
    copyParams.kind		         = cudaMemcpyHostToDevice;
    copyParams.srcPtr.xsize	     = extent.width;
    copyParams.srcPtr.ysize	     = extent.height;
    VOX_CUDA_CHECK(cudaMemcpy3D(&copyParams));
}

// --------------------------------------------------------------------
//  Binds the emissive transfer function buffer to a 3d cudaArray
// --------------------------------------------------------------------
void CTransferBuffer::bindEmissiveBuffer(std::shared_ptr<TransferMap> const& transfer)
{
    // Specify the format for volume data access
    auto formatDesc = cudaCreateChannelDesc(
        32, 32, 32, 32, cudaChannelFormatKindUnsigned);

    // Restructure buffer extent 
    auto emissive = transfer->emissive();
    cudaExtent extent;
    extent.width  = emissive.width();
    extent.height = emissive.height();
    extent.depth  = emissive.depth();

	// Create a 3d array for transfer function data storage
	VOX_CUDA_CHECK(cudaMalloc3DArray(&m_emissiveArray, &formatDesc, extent));

    // Copy data to device
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr.pitch	     = extent.width*sizeof(Vector4f);
    copyParams.srcPtr.ptr	     = (void*)emissive.data();
    copyParams.dstArray	         = m_emissiveArray;
    copyParams.extent	         = extent;
    copyParams.kind		         = cudaMemcpyHostToDevice;
    copyParams.srcPtr.xsize	     = extent.width;
    copyParams.srcPtr.ysize	     = extent.height;
    VOX_CUDA_CHECK(cudaMemcpy3D(&copyParams));
}

} // namespace vox
