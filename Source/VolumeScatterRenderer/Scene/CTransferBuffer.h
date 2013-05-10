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

// Begin definition
#ifndef VSR_CTRANSFER_BUFFER_H
#define VSR_CTRANSFER_BUFFER_H

// Common Library Header
#include "VolumeScatterRenderer/Core/Common.h"

// VoxLib Dependencies
#include "VoxLib/Core/Geometry.h"
#include "VoxLib/Scene/Transfer.h"

// API namespace
namespace vox
{

/** Rendering Volume Class */
class CTransferBuffer
{
public:
    /** Initializes the buffer for use */
    VOX_HOST void init() 
    {
        m_diffuseArray = nullptr;
        m_opacityArray = nullptr;
    }

    /** Deallocates the device memory buffer */
    VOX_HOST void reset();

    /** Sets the VolumeBuffer's data content */
    VOX_HOST void setTransfer(std::shared_ptr<TransferMap> transferMap);

    /** Returns the cudaArray storing the transfer function's opacity characteristic */
    VOX_HOST inline cudaArray const* opacityHandle() const { return m_opacityArray; }
    
    /** Returns the cudaArray storing the transfer function's diffuse characteristic */
    VOX_HOST inline cudaArray const* diffuseHandle() const { return m_diffuseArray; }

private:
    /** Binds the diffuse transfer function data to the active device */
    void bindDiffuseBuffer(std::shared_ptr<TransferMap> const& transfer);

    /** Binds the opacity transfer function data to the active device */
    void bindOpacityBuffer(std::shared_ptr<TransferMap> const& transfer);
    
private:
    cudaArray * m_diffuseArray;     ///< Handle to diffuse data array on device
    cudaArray * m_opacityArray;     ///< Handle to opacity data array on device
};

}

// End definition
#endif // VSR_CTRANSFER_BUFFER_H