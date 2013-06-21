/* ===========================================================================

	Project: VoxLib

	Description: Information structure for managed plugins

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
#ifndef VOX_PLUGIN_INFO_H
#define VOX_PLUGIN_INFO_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Types.h"

// API namespace
namespace vox
{
    /** Information structure describing a loaded plugin */
    class PluginInfo
    {
    public:
        /** Version structure for plugin */   
        struct Version 
        {
            unsigned int major;
            unsigned int minor;
            unsigned int patch;
        };

        String name;        ///< Plugin name
        String vendor;      ///< Plugin vendor
        String file;        ///< File location
        String description; ///< Plugin description
        String url;         ///< Reference URL

        Version version;        ///< Plugin version
        Version apiVersionMax;  ///< Max API version
        Version apiVersionMin;  ///< Min API version
    };

} // namespace vox

// End definition
#endif // VOX_PLUGIN_INFO_H