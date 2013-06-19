/* ===========================================================================

	Project: LibCurl IO Library Wrapper
    
	Description: Defines a VoxIO compatible plugin interface

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
#ifndef SIO_PLUGIN_H
#define SIO_PLUGIN_H

// Include dependencies
#include "StandardIO/Common.h"

// Plugin interface 
extern "C"
{
    // VoxLib Plugin - Interface Requirements
    SIO_EXPORT void         initPlugin();       ///< Called when the plugin is loaded
    SIO_EXPORT void         freePlugin();       ///< Called immediately before plugin is unloaded
    SIO_EXPORT char const*  apiVersionMin();    ///< Returns the minimum plugin API version supported
    SIO_EXPORT char const*  apiVersionMax();    ///< Returns the maximum plugin API version supported
    SIO_EXPORT char const*  name();             ///< Returns the name of the plugin
    SIO_EXPORT char const*  vendor();           ///< Returns the creator of the plugin
    SIO_EXPORT char const*  description();       ///< Returns a description of what the plugin does
    SIO_EXPORT char const*  referenceUrl();     ///< Returns a plugin reference URL
    SIO_EXPORT char const*  version();          ///< Returns the version tag of the plugin (dot delimited)
    SIO_EXPORT void         enable();           ///< Enables a plugin / signals application OK to run plugin
    SIO_EXPORT void         disable();          ///< Disables an enabled plugin
    SIO_EXPORT bool         canUnload();        ///< Indicates whether the plugin is ready to unload safely
}

// End definition
#endif // SIO_PLUGIN_H