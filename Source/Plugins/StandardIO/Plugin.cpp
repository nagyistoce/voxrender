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

// Include Header
#include "Plugin.h"

// Include Dependencies
#include "StandardIO/Common.h"
#include "StandardIO/StandardIO.h"
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Logging.h"

// LibCurl Library
#include <curl/curl.h>

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
void initPlugin() { curl_global_init(CURL_GLOBAL_ALL); }

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
void freePlugin() { curl_global_cleanup(); }

// --------------------------------------------------------------------
//  Returns the dot delimited version string for this build
// --------------------------------------------------------------------
char const* version() { return SIO_VERSION_STRING; }

// --------------------------------------------------------------------
//  Returns a reference URL for the plugin
// --------------------------------------------------------------------
char const* referenceUrl() { return ""; }

// --------------------------------------------------------------------
//  Returns the minimum compatible version of the plugin API
// --------------------------------------------------------------------
char const* apiVersionMin() { return SIO_API_VERSION_MIN_STR; }

// --------------------------------------------------------------------
//  Returns the minimum compatible version of the plugin API
// --------------------------------------------------------------------
char const* apiVersionMax() { return SIO_API_VERSION_MAX_STR; }

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
char const* name() { return "standard_io"; }

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
char const* vendor() { return "vox"; }

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
void enable() 
{  
    VOX_LOGF(vox::Severity_Info, vox::Error_None, SIO_LOG_CATEGORY, "TEST vox.standard_io");
}

// --------------------------------------------------------------------
//  Releases the specified resource module handle
// --------------------------------------------------------------------
void disable() 
{ 
}