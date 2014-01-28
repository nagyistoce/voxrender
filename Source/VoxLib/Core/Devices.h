/* ===========================================================================

	Project: VoxRender - Cuda Device interface

	Description: Performs CUDA device queries 

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
#ifndef VOX_DEVICES_H
#define VOX_DEVICES_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"

// API namespace
namespace vox
{
	/** 
	 * Cuda Device Handle
     *
     * A Cuda enabled device handle which contains the
     * device properties structure and a system unique
     * identifier code.
	 */
	struct VOX_EXPORT Device
	{
		cudaDeviceProp props;   ///< CUDA provided props for this device
		int id;                 ///< VoxRender specific device ID
	};

	/**
	 * Device Management Interface
	 */
	class VOX_EXPORT DeviceManager
	{
	public:
		/** 
         * Queries the CUDA runtime api for CUDA enabled device info 
         *
         * @throws CudaError
         */
		VOX_HOST static void loadDeviceInfo();

		/** Returns the number of available CUDA enabled devices on the system */
		VOX_HOST static int getDeviceCount() { return m_deviceCount; }

		/** Returns a handle to the specified CUDA enabled device. */
		VOX_HOST static Device const& getDevice(int dev) { return m_devices[dev]; }

        /** Returns an std::vector containing all available devices */
        VOX_HOST static std::vector<Device> const& devices() { return m_devices; }

	private:
		VOX_HOST DeviceManager( ) { }

		static std::vector<Device> m_devices;	///< Device list
		static int m_deviceCount;				///< Device count
	};

}

#endif // VOX_DEVICES_H
