/* ===========================================================================

	Project: VoxServer

	Description: Implements a WebSocket based server for interactive rendering

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
#include "Interface.h"

// Include Dependencies
#include "VoxServer/Session.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Core/System.h"
#include "VoxLib/Plugin/PluginManager.h"
#include "VoxLib/IO/ResourceHelper.h"

#include <boost/filesystem.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>

using namespace vox;
using boost::asio::ip::tcp;

namespace {
namespace filescope {

    std::shared_ptr<Session> session;

    std::ofstream logFileStream;

    // ------------------------------------------------------------------------
    //  Configures the log file stream for the server
    // ------------------------------------------------------------------------
    void configureLogging()
    {
        // Compose default filename and path for session log file
        String logLocation = vox::System::currentDirectory() + "/Logs/"; 
        String logFilename = vox::System::computerName() + ".log";
    
        // Ensure log directory exists for local filesystem 
        if (!boost::filesystem::exists(logLocation))
        {
            boost::filesystem::create_directory(logLocation);
        }

        // Create log file for this session 
        logFileStream.open(logLocation + logFilename, std::ios_base::app);

        // Redirect std::clog stream to session log sink
        if (logFileStream) std::clog.set_rdbuf(logFileStream.rdbuf());
    }

} // namespace filescope
} // namespace anonymous

// --------------------------------------------------------------------
//  Initializes the render controllers
// --------------------------------------------------------------------
int voxServerStart(char const* directory, bool logToFile) 
{
    try 
    {
        // Configure the log file output 
        boost::filesystem::current_path(directory);
        if (logToFile) filescope::configureLogging();

        // Display and log the library version info and startup time
        VOX_LOG_INFO(VOX_SERV_LOG_CAT, format("VoxServer Version: %1%", VOX_SERV_VERSION_STRING));

        // Load the configuration file (Scene plugins, etc)
        ResourceHelper::loadConfigFile("VoxServer.config");

        return Error_None;
    }
    catch (Error & e)
    {
        VOX_LOG_EXCEPTION(Severity_Fatal, e);

        return e.code;
    }
    catch (std::exception & e)
    {
        VOX_LOG_ERROR(Error_Unknown, VOX_SERV_LOG_CAT, e.what());

        return Error_Unknown;
    }
}

// --------------------------------------------------------------------
//  terminates the render
// --------------------------------------------------------------------
void voxServerEnd()
{ 
    VOX_LOG_INFO(VOX_SERV_LOG_CAT, "Terminating render service");
    
    PluginManager::instance().unloadAll();
    filescope::logFileStream.close();
}

// --------------------------------------------------------------------
//  Begins rendering the specified scene file
// --------------------------------------------------------------------
int voxServerBeginStream(uint16_t * portOut, uint64_t * keyOut, char const* rootDir)
{
    // Select a port and key for the stream
    UInt16 port = 8000;
    UInt64 key  = 0;
    
    // Prepare the WebSocket to accept an incoming connection
    try
    {
        filescope::session = std::make_shared<Session>(port, key, rootDir);
        filescope::session->start();
    }
    catch (std::exception& e)
    {
        VOX_LOG_ERROR(Error_Unknown, VOX_SERV_LOG_CAT, e.what());
        return Error_Unknown;
    }

    // Return the connection info to the user
    *portOut = port;
    *keyOut  = key;

    return Error_None;
}

// --------------------------------------------------------------------
//  Returns the dot delimited version string for this build
// --------------------------------------------------------------------
char const* voxServerVersion() { return VOX_SERV_VERSION_STRING; }
