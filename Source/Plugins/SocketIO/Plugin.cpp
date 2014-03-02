/* ===========================================================================

    Project: SocketIO                                                       
                                                                           
    Description: Provides an IO module for low level socket IO              
                                                                           
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
#include "SocketIO/Common.h"
#include "SocketIO/TcpSocketIO.h"
#include "SocketIO/UdpSocketIO.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Plugin/PluginManager.h"

using namespace vox;

namespace {
namespace filescope {

    static std::shared_ptr<TcpSocketIO> tcpio;
    static std::shared_ptr<UdpSocketIO> udpio;
    std::shared_ptr<void> handle;

} // namespace filescope
} // namespace anonymous

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
void initPlugin() 
{
    VOX_LOG_INFO(SOKIO_LOG_CAT, "Loading the 'Vox.Socket IO' plugin"); 
    
    filescope::handle = PluginManager::instance().acquirePluginHandle();
}

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
void freePlugin()
{ 
    VOX_LOG_INFO(SOKIO_LOG_CAT, "Unloading the 'Vox.Socket IO' plugin");
}

// --------------------------------------------------------------------
//  Returns the dot delimited version string for this build
// --------------------------------------------------------------------
char const* version() { return SOKIO_VERSION_STRING; }

// --------------------------------------------------------------------
//  Returns a reference URL for the plugin
// --------------------------------------------------------------------
char const* referenceUrl() { return "http://code.google.com/p/voxrender/"; }

// --------------------------------------------------------------------
//  Returns the minimum compatible version of the plugin API
// --------------------------------------------------------------------
char const* apiVersionMin() { return SOKIO_API_VERSION_MIN_STR; }

// --------------------------------------------------------------------
//  Returns the minimum compatible version of the plugin API
// --------------------------------------------------------------------
char const* apiVersionMax() { return SOKIO_API_VERSION_MAX_STR; }

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
char const* name() { return "Socket IO"; }

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
char const* vendor() { return "Vox"; }

// --------------------------------------------------------------------
//  Returns a description of the plugin
// --------------------------------------------------------------------
char const* description() 
{
    return  "The SocketIO plugin provides a resource module which handles low level TCP/UDP streaming "
            "using the ResourceIO interface."
            ;
}

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
void enable() 
{  
    VOX_LOG_INFO(SOKIO_LOG_CAT, "Enabling the 'Vox.Socket IO' plugin");

    std::shared_ptr<TcpSocketIO> tcpio(new TcpSocketIO());
    std::shared_ptr<UdpSocketIO> udpio(new UdpSocketIO());
    
    vox::Resource::registerModule("tcp", tcpio);
    vox::Resource::registerModule("udp", udpio);
    
    filescope::tcpio = tcpio;
    filescope::udpio = udpio;
}

// --------------------------------------------------------------------
//  Releases the specified resource module handle
// --------------------------------------------------------------------
void disable() 
{ 
    VOX_LOG_INFO(SOKIO_LOG_CAT, "Disabling the 'Vox.Socket IO' plugin");

    vox::Resource::removeModule(filescope::tcpio);
    vox::Resource::removeModule(filescope::udpio);

    filescope::tcpio.reset();
    filescope::udpio.reset();
    filescope::handle.reset();
}