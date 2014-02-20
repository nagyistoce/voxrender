/* ===========================================================================

	Project: VoxRender - Cuda Device interface

	Description:
	 Performs cuda device querying and attribute reading

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
#include "Devices.h"

// Include Dependencies
#include "VoxLib/Error/ErrorCodes.h"
#include "VoxLib/Error/CudaError.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Core/Format.h"

// Standard Library
#include <iostream>

// Static members
std::vector<vox::Device> vox::DeviceManager::m_devices;
int vox::DeviceManager::m_deviceCount = 0;

// -------------------------------------------------------------
// Constructor - Performs initial device count and loads device 
// properties using the runtime api
// -------------------------------------------------------------
void vox::DeviceManager::loadDeviceInfo( )
{
    cudaError errorId;

	// Get CUDA enabled device count
    if( errorId = cudaGetDeviceCount( &m_deviceCount ) ) 
	{
		m_deviceCount = 0;

        CudaError error( __FILE__, __LINE__, VOX_LOG_CATEGORY, 
            "cudaGetDeviceCount", errorId );
        Logger::addEntry( error, Severity_Error );
        
        throw error;
    }

    if( m_deviceCount == 0 )
    {
        Logger::addEntry( Severity_Warning, Error_Device, VOX_LOG_CATEGORY,
            "No CUDA capable devices found.", __FILE__, __LINE__ );
    }
    else
    {
        Logger::addEntry( Severity_Info, Error_None, 
            VOX_LOG_CATEGORY, format("Found %1% CUDA capable device(s)", m_deviceCount), 
            __FILE__, __LINE__ );
    }

	// Load device properties
	m_devices.resize(m_deviceCount);
	for (int dev = 0; dev < m_deviceCount; dev++)
	{	
		m_devices[dev].id = dev;
        VOX_CUDA_CHECK(cudaGetDeviceProperties(&m_devices[dev].props, dev));
	}
}