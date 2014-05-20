/* ===========================================================================

    Project: Standard Video Import/Export
    
	Description: Provides an import/export module for video formats

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

// Include Header
#include "Plugin.h"

// Include Dependencies
#include "StandardVid/Common.h"
#include "StandardVid/AviVid.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Plugin/PluginManager.h"
#include "VoxLib/Video/VidStream.h"

// FFMPEG Headers
extern "C"
{
    #include "libavformat/avformat.h"
    #include "libswscale/swscale.h"
}

using namespace vox;

namespace {
namespace filescope {

    std::shared_ptr<AviVid> ffmpegExim;
    std::shared_ptr<void> handle;

} // namespace filescope
} // namespace anonymous

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
void initPlugin() 
{
    VOX_LOG_INFO(VOX_SVID_LOG_CATEGORY, "Loading the 'Vox.Standard Vid ExIm' plugin");
    
    filescope::handle = PluginManager::instance().acquirePluginHandle();

    av_register_all();
}

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
void freePlugin() 
{
    VOX_LOG_INFO(VOX_SVID_LOG_CATEGORY, "Unloading the 'Vox.Standard Vid ExIm' plugin");
}

// --------------------------------------------------------------------
//  Returns the dot delimited version string for this build
// --------------------------------------------------------------------
char const* version() { return VOX_SVID_VERSION_STRING; }

// --------------------------------------------------------------------
//  Returns a reference URL for the plugin
// --------------------------------------------------------------------
char const* referenceUrl() { return "http://code.google.com/p/voxrender/"; }

// --------------------------------------------------------------------
//  Returns the minimum compatible version of the plugin API
// --------------------------------------------------------------------
char const* apiVersionMin() { return VOX_SVID_API_VERSION_MIN_STR; }

// --------------------------------------------------------------------
//  Returns the minimum compatible version of the plugin API
// --------------------------------------------------------------------
char const* apiVersionMax() { return VOX_SVID_API_VERSION_MAX_STR; }

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
char const* name() { return "Standard Vid ExIm"; }

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
char const* vendor() { return "Vox"; }

// --------------------------------------------------------------------
//  Returns a description of the plugin
// --------------------------------------------------------------------
char const* description() 
{
    return  "The Standard Video ExIm plugin provides image import and export modules "
            "for various video formats including: avi"
            ;
}

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
void enable() 
{  
    VOX_LOG_INFO(VOX_SVID_LOG_CATEGORY, "Enabling the 'Vox.Standard Vid ExIm' plugin");
    
    filescope::ffmpegExim  = std::shared_ptr<AviVid> (new AviVid(filescope::handle));
    
    // Encoder formats
    vox::VidOStream::registerEncoder(".avi", filescope::ffmpegExim);

    vox::VidOStream::registerEncoder(".mpg", filescope::ffmpegExim);

    vox::VidOStream::registerEncoder(".asf", filescope::ffmpegExim);
    vox::VidOStream::registerEncoder(".wmv", filescope::ffmpegExim);
    vox::VidOStream::registerEncoder(".wma", filescope::ffmpegExim);

    vox::VidOStream::registerEncoder(".caf", filescope::ffmpegExim);

    vox::VidOStream::registerEncoder(".flv", filescope::ffmpegExim);

    vox::VidOStream::registerEncoder(".ogg", filescope::ffmpegExim);
    vox::VidOStream::registerEncoder(".ogv", filescope::ffmpegExim);
    vox::VidOStream::registerEncoder(".ogx", filescope::ffmpegExim);
    vox::VidOStream::registerEncoder(".oga", filescope::ffmpegExim);
    vox::VidOStream::registerEncoder(".ogm", filescope::ffmpegExim);
    vox::VidOStream::registerEncoder(".spx", filescope::ffmpegExim);
    vox::VidOStream::registerEncoder(".opus", filescope::ffmpegExim);

    // Decoder formats
    vox::VidIStream::registerDecoder(".avi", filescope::ffmpegExim);

    vox::VidIStream::registerDecoder(".mpg", filescope::ffmpegExim);

    vox::VidIStream::registerDecoder(".asf", filescope::ffmpegExim);
    vox::VidIStream::registerDecoder(".wmv", filescope::ffmpegExim);
    vox::VidIStream::registerDecoder(".wma", filescope::ffmpegExim);

    vox::VidIStream::registerDecoder(".caf", filescope::ffmpegExim);

    vox::VidIStream::registerDecoder(".flv", filescope::ffmpegExim);

    vox::VidIStream::registerDecoder(".ogg", filescope::ffmpegExim);
    vox::VidIStream::registerDecoder(".ogv", filescope::ffmpegExim);
    vox::VidIStream::registerDecoder(".ogx", filescope::ffmpegExim);
    vox::VidIStream::registerDecoder(".oga", filescope::ffmpegExim);
    vox::VidIStream::registerDecoder(".ogm", filescope::ffmpegExim);
    vox::VidIStream::registerDecoder(".spx", filescope::ffmpegExim);
    vox::VidIStream::registerDecoder(".opus", filescope::ffmpegExim);
}

// --------------------------------------------------------------------
//  Releases the specified resource module handle
// --------------------------------------------------------------------
void disable() 
{ 
    VOX_LOG_INFO(VOX_SVID_LOG_CATEGORY, "Disabling the 'Vox.Standard Vid ExIm' plugin");
    
    vox::VidOStream::removeEncoder(filescope::ffmpegExim);
    vox::VidIStream::removeDecoder(filescope::ffmpegExim);
    
    filescope::ffmpegExim.reset();
    filescope::handle.reset();
}