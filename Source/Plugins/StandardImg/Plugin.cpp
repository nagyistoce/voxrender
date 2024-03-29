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
#include "StandardImg/BmpImg.h"
#include "StandardImg/PngImg.h"
#include "StandardImg/JpegImg.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Plugin/PluginManager.h"

using namespace vox;

namespace {
namespace filescope {

    std::shared_ptr<PngImg>  pngExim;
    std::shared_ptr<JpegImg> jpegExim;
    std::shared_ptr<JpegImg> jpsExim;
    std::shared_ptr<JpegImg> mpoExim;
    std::shared_ptr<BmpImg>  bmpExim;
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
            "for various image formats including: png, jpeg"
            ;
}

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
void enable() 
{  
    VOX_LOG_INFO(VOX_SIMG_LOG_CATEGORY, "Enabling the 'Vox.Standard Img ExIm' plugin");
    
    filescope::pngExim  = std::shared_ptr<PngImg> (new PngImg (filescope::handle));
    filescope::jpegExim = std::shared_ptr<JpegImg>(new JpegImg(filescope::handle, JpegImg::Container_JPG));
    filescope::mpoExim  = std::shared_ptr<JpegImg>(new JpegImg(filescope::handle, JpegImg::Container_MPO));
    filescope::jpsExim  = std::shared_ptr<JpegImg>(new JpegImg(filescope::handle, JpegImg::Container_JPS));
    filescope::bmpExim  = std::shared_ptr<BmpImg> (new BmpImg (filescope::handle));
    
    vox::Bitmap::registerImportModule(".bmp",  filescope::bmpExim);
    vox::Bitmap::registerExportModule(".bmp",  filescope::bmpExim);
    vox::Bitmap::registerImportModule(".png",  filescope::pngExim);
    vox::Bitmap::registerExportModule(".png",  filescope::pngExim);
    vox::Bitmap::registerImportModule(".jpeg", filescope::jpegExim);
    vox::Bitmap::registerExportModule(".jpeg", filescope::jpegExim);
    vox::Bitmap::registerImportModule(".jpg",  filescope::jpegExim);
    vox::Bitmap::registerExportModule(".jpg",  filescope::jpegExim);
    vox::Bitmap::registerImportModule(".mpo",  filescope::mpoExim);
    vox::Bitmap::registerExportModule(".mpo",  filescope::mpoExim);
    vox::Bitmap::registerImportModule(".jps",  filescope::jpsExim);
    vox::Bitmap::registerExportModule(".jps",  filescope::jpsExim);
}

// --------------------------------------------------------------------
//  Releases the specified resource module handle
// --------------------------------------------------------------------
void disable() 
{ 
    VOX_LOG_INFO(VOX_SIMG_LOG_CATEGORY, "Disabling the 'Vox.Standard Img ExIm' plugin");
    
    vox::Bitmap::removeImportModule(filescope::bmpExim);
    vox::Bitmap::removeExportModule(filescope::bmpExim);
    vox::Bitmap::removeImportModule(filescope::pngExim);
    vox::Bitmap::removeExportModule(filescope::pngExim);
    vox::Bitmap::removeImportModule(filescope::jpegExim);
    vox::Bitmap::removeExportModule(filescope::jpegExim);
    vox::Bitmap::removeImportModule(filescope::mpoExim);
    vox::Bitmap::removeExportModule(filescope::mpoExim);
    vox::Bitmap::removeImportModule(filescope::jpsExim);
    vox::Bitmap::removeExportModule(filescope::jpsExim);
    
    filescope::bmpExim.reset();
    filescope::pngExim.reset();
    filescope::jpegExim.reset();
    filescope::mpoExim.reset();
    filescope::jpsExim.reset();
    filescope::handle.reset();
}