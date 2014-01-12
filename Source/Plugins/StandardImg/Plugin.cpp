/* ===========================================================================

    Project: Standard Image ExIm Module
    
	Description: Defines an image import module for common LDR image formats

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
#include "StandardImg/Common.h"
#include "StandardImg/StandardImg.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Plugin/PluginManager.h"

using namespace vox;

namespace {
namespace filescope {

    std::shared_ptr<StandardImg> exim;
    std::shared_ptr<void> handle;

} // namespace filescope
} // namespace anonymous

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
void initPlugin() 
{
    VOX_LOG_INFO(VOX_SIMG_LOG_CATEGORY, "Loading the 'Vox.Standard Img ExIm' plugin");
    
    filescope::handle = PluginManager::instance().acquirePluginHandle();
}

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
void freePlugin() 
{
    VOX_LOG_INFO(VOX_SIMG_LOG_CATEGORY, "Unloading the 'Vox.Standard Img ExIm' plugin");
}

// --------------------------------------------------------------------
//  Returns the dot delimited version string for this build
// --------------------------------------------------------------------
char const* version() { return VOX_SIMG_VERSION_STRING; }

// --------------------------------------------------------------------
//  Returns a reference URL for the plugin
// --------------------------------------------------------------------
char const* referenceUrl() { return "http://code.google.com/p/voxrender/"; }

// --------------------------------------------------------------------
//  Returns the minimum compatible version of the plugin API
// --------------------------------------------------------------------
char const* apiVersionMin() { return VOX_SIMG_API_VERSION_MIN_STR; }

// --------------------------------------------------------------------
//  Returns the minimum compatible version of the plugin API
// --------------------------------------------------------------------
char const* apiVersionMax() { return VOX_SIMG_API_VERSION_MAX_STR; }

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
char const* name() { return "Standard Img ExIm"; }

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
char const* vendor() { return "Vox"; }

// --------------------------------------------------------------------
//  Returns a description of the plugin
// --------------------------------------------------------------------
char const* description() 
{
    return  "The Standard Image ExIm plugin provides image import and export modules "
            "for various image formats include: png"
            ;
}

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
void enable() 
{  
    VOX_LOG_INFO(VOX_SIMG_LOG_CATEGORY, "Enabling the 'Vox.Standard Img ExIm' plugin");
    
    filescope::exim = std::shared_ptr<StandardImg>(new StandardImg(filescope::handle));

    vox::RawImage::registerImportModule(".png", filescope::exim);
    vox::RawImage::registerExportModule(".png", filescope::exim);
}

// --------------------------------------------------------------------
//  Releases the specified resource module handle
// --------------------------------------------------------------------
void disable() 
{ 
    VOX_LOG_INFO(VOX_SIMG_LOG_CATEGORY, "Disabling the 'Vox.Standard Img ExIm' plugin");

    vox::RawImage::removeImportModule(filescope::exim);
    vox::RawImage::removeExportModule(filescope::exim);

    filescope::exim.reset();
    filescope::handle.reset();
}