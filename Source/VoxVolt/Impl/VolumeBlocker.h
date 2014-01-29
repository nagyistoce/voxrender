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

// Begin definition
#ifndef VOX_VOLT_VOLUME_BLOCK_H
#define VOX_VOLT_VOLUME_BLOCK_H

// Include Dependencies
#include "VoxVolt/Common.h"
#include "VoxLib/Scene/Volume.h"

// API namespace
namespace vox {
namespace volt {

/** Implements a sequential block loading scheme for performing volume operations on the GPU */
class VolumeBlocker
{
public:
    VolumeBlocker(Volume & volume, Vector3u apron, Volume::Type outType);

    ~VolumeBlocker() { reset(); }

    void reset();

    void begin();

    Vector4u loadNext();

    void readNext();

    bool atEnd();

    std::shared_ptr<Volume> finish();

    cudaExtent blockSize() { return m_extentOut; }

    cudaArray * arrayOut() { return m_handleOut; }

    cudaArray * arrayIn() { return m_handle; }

    cudaChannelFormatDesc const& formatOut() { return m_formatOut; }
    
    cudaChannelFormatDesc const& formatIn() { return m_format; }

    Volume & volume() { return m_volume; }

private:
    Volume & m_volume; ///< The source volume

    bool m_begin;

    std::shared_ptr<UInt8> m_dataOut; ///< The output volume data
    Volume::Type m_typeOut; ///< The output volume type

    Vector4u m_currBlock; ///< The currently loaded block index
    Vector4u m_blockSize; ///< The size in voxels of a loaded block
    Vector4u m_numBlocks; ///< The number of blocks in each direction (this is cached for simplicity)
    Vector3u m_apron;     ///< The size of the apron in the GPU memory blocks

    cudaChannelFormatDesc m_format;    ///< Texture channel format for input
    cudaChannelFormatDesc m_formatOut; ///< Texture channel format for output

    cudaExtent  m_extent;    ///< Extent of the volume data for input
    cudaExtent  m_extentOut; ///< Extent of the volume data for output
    cudaArray * m_handle;    ///< Handle to device data buffer for input
    cudaArray * m_handleOut; ///< Handle to device data buffer for output
};

} // namespace volt
} // namespace vox

// End definition
#endif // VOX_VOLT_VOLUME_BLOCK_H