/* ===========================================================================

    Project: PVM Volume Import Module
    
	Description: Defines a VoxScene import module for .pvm format volumes

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
#ifndef PVMI_PLUGIN_H
#define PVMI_PLUGIN_H

// Include dependencies
#include "PvmVolumeImporter/Common.h"

// Plugin interface 
extern "C"
{
    // VoxLib Plugin - Interface Requirements
    PVMI_EXPORT void         initPlugin();       ///< Called when the plugin is loaded
    PVMI_EXPORT void         freePlugin();       ///< Called immediately before plugin is unloaded
    PVMI_EXPORT char const*  apiVersionMin();    ///< Returns the minimum plugin API version supported
    PVMI_EXPORT char const*  apiVersionMax();    ///< Returns the maximum plugin API version supported
    PVMI_EXPORT char const*  name();             ///< Returns the name of the plugin
    PVMI_EXPORT char const*  vendor();           ///< Returns the creator of the plugin
    PVMI_EXPORT char const*  description();       ///< Returns a description of what the plugin does
    PVMI_EXPORT char const*  referenceUrl();     ///< Returns a plugin reference URL
    PVMI_EXPORT char const*  version();          ///< Returns the version tag of the plugin (dot delimited)
    PVMI_EXPORT void         enable();           ///< Enables a plugin / signals application OK to run plugin
    PVMI_EXPORT void         disable();          ///< Disables an enabled plugin
}

// End definition
#endif // PVMI_PLUGIN_H