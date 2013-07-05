/* ===========================================================================

    Project: StandardIO - Standard IO protocols for VoxIO
    
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
#include "VoxLib/Plugin/PluginManager.h"

// LibCurl Library
#include <curl/curl.h>

namespace {
namespace filescope {

    static std::shared_ptr<vox::StandardIO> io;
    std::shared_ptr<void> handle;

} // namespace filescope
} // namespace anonymous

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
void initPlugin() 
{
    VOX_LOG_INFO(SIO_LOG_CATEGORY, "Loading the 'Vox.Standard IO' plugin");
    
    filescope::handle = vox::PluginManager::instance().acquirePluginHandle(); 
}

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
void freePlugin() 
{
    VOX_LOG_INFO(SIO_LOG_CATEGORY, "Unloading the 'Vox.Standard IO' plugin"); 
}

// --------------------------------------------------------------------
//  Returns the dot delimited version string for this build
// --------------------------------------------------------------------
char const* version() { return SIO_VERSION_STRING; }

// --------------------------------------------------------------------
//  Returns a reference URL for the plugin
// --------------------------------------------------------------------
char const* referenceUrl() { return "http://code.google.com/p/voxrender/"; }

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
char const* name() { return "Standard IO"; }

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
char const* vendor() { return "Vox"; }

// --------------------------------------------------------------------
//  Returns a description of the plugin
// --------------------------------------------------------------------
char const* description() 
{
    return  "The StandardIO plugin provides resource modules for the following internet protocols:\n\n"
		
		    "HTTP, HTTPS, FTP, FTPS, DICT, LDAP, LDAPS, IMAP, IMAPS, POP3, POP3S, SMTP, SMTPS, GOPHER, TELNET, TFTP, SFTP, RTMP, RTSP, and SCP.\n\n"
		  
		    "Individual protocols can be suppressed by editing the StandardIO.pin XML file's restrict attribute."
        ;
}

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
void enable() 
{  
    VOX_LOG_INFO(SIO_LOG_CATEGORY, "Enabling the 'Vox.Standard IO' plugin");

    std::shared_ptr<vox::StandardIO> io(new vox::StandardIO());

    vox::Resource::registerModule("http",   io);
    vox::Resource::registerModule("https",  io);
    vox::Resource::registerModule("ftp",    io);
    vox::Resource::registerModule("ftps",   io);
    vox::Resource::registerModule("sftp",   io);
    vox::Resource::registerModule("tftp",   io);
    vox::Resource::registerModule("rtmp",   io);
    vox::Resource::registerModule("rtsp",   io);
    vox::Resource::registerModule("smtp",   io);
    vox::Resource::registerModule("smtps",  io);
    vox::Resource::registerModule("dict",   io);
    vox::Resource::registerModule("scp",    io);
    vox::Resource::registerModule("imap",   io);
    vox::Resource::registerModule("imaps",  io);
    vox::Resource::registerModule("pop3",   io);
    vox::Resource::registerModule("pop3s",  io);
    vox::Resource::registerModule("ldap",   io);
    vox::Resource::registerModule("ldaps",  io);
    vox::Resource::registerModule("gopher", io);
    vox::Resource::registerModule("telnet", io);

    filescope::io = io;
}

// --------------------------------------------------------------------
//  Releases the specified resource module handle
// --------------------------------------------------------------------
void disable() 
{ 
    VOX_LOG_INFO(SIO_LOG_CATEGORY, "Disabling the 'Vox.Standard IO' plugin");

    vox::Resource::removeModule(filescope::io);

    filescope::io.reset();
    filescope::handle.reset();
}