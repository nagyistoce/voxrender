/* ===========================================================================
                                                                           
   Project: Volume Transform Library                                       
                                                                           
   Description: Performs volume transform operations                       
                                                                           
   Copyright (C) 2014 Lucas Sherman                                        
                                                                           
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
#include "VolumeBlocker.h"

// Include Dependencies
#include "VoxLib/Error/CudaError.h"

namespace vox {
namespace volt {
    
namespace {
namespace filescope {

    // Determines the proper cuda format for a volume data type
    cudaChannelFormatKind getKind(Volume::Type type)
    {
        switch (type)
        {
        case Volume::Type_Float32: case Volume::Type_Float64: 
            return cudaChannelFormatKindFloat;
        case Volume::Type_Int8: case Volume::Type_Int16:
            return cudaChannelFormatKindSigned;
        case Volume::Type_UInt8: case Volume::Type_UInt16:
            return cudaChannelFormatKindUnsigned;
        default:
            throw Error(__FILE__, __LINE__, VOX_VOLT_LOG_CATEGORY,
                format("Invalid volume data type specification (%1%)", 
                    Volume::typeToString(type)), Error_NotImplemented);
        }
    }

} // namespace filescope
} // namespace anonymous

// ----------------------------------------------------------------------------
//  Initializes a volume block loader/scheduler
// ----------------------------------------------------------------------------
VolumeBlocker::VolumeBlocker(Volume & volume, Vector3u apron, Volume::Type outType) : 
    m_volume(volume), m_handle(nullptr), m_handleOut(nullptr), m_typeOut(outType)
{
    // Specify the format for volume data access
    m_format    = cudaCreateChannelDesc(m_volume.voxelSize()*8, 0, 0, 0, filescope::getKind(m_volume.type()));
    m_formatOut = cudaCreateChannelDesc(m_volume.voxelSize()*8, 0, 0, 0, filescope::getKind(outType));
    
    auto extent = m_volume.extent();
    
    // :TODO: Query GPU memory size to determine blocking scheme

    // Compute extents for the volume data blocks
    m_extentOut.width  = extent[0];
    m_extentOut.height = extent[1];
    m_extentOut.depth  = extent[2] ;

    m_numBlocks = Vector4u(
       (extent[0] + m_extentOut.width  - 1) / m_extentOut.width,
       (extent[1] + m_extentOut.height - 1) / m_extentOut.height,
       (extent[2] + m_extentOut.depth  - 1) / m_extentOut.depth,
        extent[3]
        );

    // Compute the optimal extent for the data blocks
    m_extent.width  = low(m_extentOut.width  + m_apron[0] * 2, extent[0]);
    m_extent.height = low(m_extentOut.height + m_apron[1] * 2, extent[1]);
    m_extent.depth  = low(m_extentOut.depth  + m_apron[2] * 2, extent[2]);
}

// ----------------------------------------------------------------------------
//  Releases any device memory associated with the volume
// ----------------------------------------------------------------------------
void VolumeBlocker::reset()
{
    if (m_handle)
    {
        cudaFreeArray(m_handle);
        m_handle = nullptr;
    }

    if (m_handleOut)
    {
        cudaFreeArray(m_handleOut);
        m_handleOut = nullptr;
    }
}

// ----------------------------------------------------------------------------
//  Binds the volume data to device buffers for use in rendering
// ----------------------------------------------------------------------------
void VolumeBlocker::begin()
{
    reset(); // Ensure previous data is released

    m_currBlock = Vector4u(0);
    m_begin = false;

    // Allocate a host buffer for the output volume data set
    size_t bytes = m_volume.extent().fold(mul) * Volume::typeToSize(m_typeOut);
    m_dataOut = makeSharedArray(bytes);

	// Create a 3d array for volume data storage
	VOX_CUDA_CHECK(cudaMalloc3DArray(&m_handle, &m_format, m_extent));
    VOX_CUDA_CHECK(cudaMalloc3DArray(&m_handleOut, &m_formatOut, m_extentOut, cudaArraySurfaceLoadStore));
}

// ----------------------------------------------------------------------------
//  Completes the volume blocking operations and constructs the final output
// ----------------------------------------------------------------------------
std::shared_ptr<Volume> VolumeBlocker::finish()
{
    reset();

    return Volume::create(m_dataOut, m_volume.extent(), m_volume.spacing(), 
                          m_volume.offset(), m_typeOut);
}

// ----------------------------------------------------------------------------
//  Completes the volume blocking operations and constructs the final output
// ----------------------------------------------------------------------------
bool VolumeBlocker::atEnd()
{
    return (m_currBlock[0] == m_numBlocks[0]-1) &&
           (m_currBlock[1] == m_numBlocks[1]-1) &&
           (m_currBlock[2] == m_numBlocks[2]-1) &&
           (m_currBlock[3] == m_numBlocks[3]-1);
}

// ----------------------------------------------------------------------------
//  Loads the specified block section of the volume into GPU memory
// ----------------------------------------------------------------------------
Vector4u VolumeBlocker::loadNext()
{
    if (m_begin)
    {
        // Increment the block index
        m_currBlock[0]++;
        if (m_currBlock[0] >= m_numBlocks[0]) { m_currBlock[0] = 0; m_currBlock[1]++; }
        if (m_currBlock[1] >= m_numBlocks[1]) { m_currBlock[1] = 0; m_currBlock[2]++; }
        if (m_currBlock[2] >= m_numBlocks[2]) { m_currBlock[2] = 0; m_currBlock[3]++; }
        if (m_currBlock[3] >= m_numBlocks[3]) { m_currBlock[3] = 0; 
            throw Error(__FILE__, __LINE__, VOX_VOLT_LOG_CATEGORY, "Exceeded block size", Error_Bug); }
    }
    else m_begin = true;

    auto hostExtent = m_volume.extent();
    cudaPos pos;
    pos.x = m_currBlock[0] * m_extentOut.width;
    pos.y = m_currBlock[1] * m_extentOut.height;
    pos.z = m_currBlock[2] * m_extentOut.depth;

    // Copy volume read data to device
	cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr.pitch	     = hostExtent[0]*m_volume.voxelSize();
    copyParams.srcPtr.ptr	     = (void*)m_volume.data();
    copyParams.srcPtr.xsize	     = hostExtent[0];
    copyParams.srcPtr.ysize	     = hostExtent[1];
    copyParams.srcPos            = pos;
    copyParams.dstArray	         = m_handle;
    copyParams.extent	         = m_extent;
    copyParams.kind		         = cudaMemcpyHostToDevice;
    VOX_CUDA_CHECK(cudaMemcpy3D(&copyParams));

    return m_currBlock;
}

// ----------------------------------------------------------------------------
//  Loads the current block section of the output volume into host memory
// ----------------------------------------------------------------------------
void VolumeBlocker::readNext()
{
    auto hostExtent = m_volume.extent();
    cudaPos pos;
    pos.x = m_currBlock[0] * m_extentOut.width;
    pos.y = m_currBlock[1] * m_extentOut.height;
    pos.z = m_currBlock[2] * m_extentOut.depth;

    // Copy volume write data to host
	cudaMemcpy3DParms copyParams = {0};
    copyParams.srcArray          = m_handleOut;
    copyParams.dstPtr.pitch	     = hostExtent[0]*Volume::typeToSize(m_typeOut);
    copyParams.dstPtr.ptr	     = (void*)m_dataOut.get();
    copyParams.dstPtr.xsize      = hostExtent[0];
    copyParams.dstPtr.ysize      = hostExtent[1];
    copyParams.dstPos            = pos;
    copyParams.extent	         = m_extentOut;
    copyParams.kind		         = cudaMemcpyDeviceToHost;
    VOX_CUDA_CHECK(cudaMemcpy3D(&copyParams));
}

} // namespace volt
} // namespace vox