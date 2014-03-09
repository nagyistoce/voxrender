/* ===========================================================================

	Project: Standard Volume Filters
    
	Description: Exposes some standard filters provided by the volt library

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
#ifndef SVF_PLUGIN_H
#define SVF_PLUGIN_H

// Include dependencies
#include "StdVolumeFilter/Common.h"

// Plugin interface 
extern "C"
{
    // VoxLib Plugin - Interface Requirements
    SVF_EXPORT void         initPlugin();       ///< Called when the plugin is loaded
    SVF_EXPORT void         freePlugin();       ///< Called immediately before plugin is unloaded
    SVF_EXPORT char const*  apiVersionMin();    ///< Returns the minimum plugin API version supported
    SVF_EXPORT char const*  apiVersionMax();    ///< Returns the maximum plugin API version supported
    SVF_EXPORT char const*  name();             ///< Returns the name of the plugin
    SVF_EXPORT char const*  vendor();           ///< Returns the creator of the plugin
    SVF_EXPORT char const*  description();       ///< Returns a description of what the plugin does
    SVF_EXPORT char const*  referenceUrl();     ///< Returns a plugin reference URL
    SVF_EXPORT char const*  version();          ///< Returns the version tag of the plugin (dot delimited)
    SVF_EXPORT void         enable();           ///< Enables a plugin / signals application OK to run plugin
    SVF_EXPORT void         disable();          ///< Disables an enabled plugin
}

// End definition
#endif // SVF_PLUGIN_H