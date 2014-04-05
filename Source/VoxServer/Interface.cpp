/* ===========================================================================

    Project: VoxServer
    
	Description: Rendering library for VoxRenderWeb

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
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Core/System.h"
#include "VoxLib/Plugin/PluginManager.h"
#include "VoxLib/IO/ResourceHelper.h"

// Include WebSocket Server Dependencies
#include <boost/asio.hpp>

// Include Scene Components
#include "VoxScene/RenderController.h"
#include "VoxScene/Volume.h"
#include "VoxScene/Camera.h"
#include "VoxScene/Light.h"
#include "VoxScene/RenderParams.h"
#include "VoxScene/PrimGroup.h"
#include "VoxScene/Transfer.h"

// Standard Renderers for the Application
#include "VolumeScatterRenderer/Core/VolumeScatterRenderer.h"

// Server log category
#define LOG_CAT "VoxServer"

// Stringify macro
#define VSERV_XSTR(v) #v
#define VSERV_STR(v) VSERV_XSTR(v)

// Plugin version info
#define VSERV_VERSION_MAJOR 1
#define VSERV_VERSION_MINOR 0
#define VSERV_VERSION_PATCH 0

// API support version info
#define VSERV_API_VERSION_MIN_STR "0.0.0"
#define VSERV_API_VERSION_MAX_STR "999.999.999"

// Plugin version string
#define VSERV_VERSION_STRING VSERV_STR(VSERV_VERSION_MAJOR) \
	"." VSERV_STR(VSERV_VERSION_MINOR) "." VSERV_STR(VSERV_VERSION_PATCH)

using namespace vox;
using boost::asio::ip::tcp;

namespace {
namespace filescope {

    std::shared_ptr<VolumeScatterRenderer> renderer;

    std::ofstream logFileStream;

    RenderController renderController;
    Scene            scene;

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
    
    // ------------------------------------------------------------------------
    //  Callback function on frame ready
    // ------------------------------------------------------------------------
    void onFrameReady(std::shared_ptr<vox::FrameBuffer> frame)
    {
    }

} // namespace filescope
} // namespace anonymous

// --------------------------------------------------------------------
//  Initializes the render controllers
// --------------------------------------------------------------------
int voxServerStart(char const* directory) 
{
    try 
    {
        // Configure the log file output 
        boost::filesystem::current_path(directory);
        filescope::configureLogging();

        // Display and log the library version info and startup time
        VOX_LOG_INFO(LOG_CAT, format("VoxServer Version: %1%", VSERV_VERSION_STRING));

        // Load the configuration file (Scene plugins, etc)
        ResourceHelper::loadConfigFile("VoxServer.config");

        // Configure the volume renderer and generate the usage info
        filescope::renderer = vox::VolumeScatterRenderer::create();
        filescope::renderer->setRenderEventCallback(&filescope::onFrameReady);

        return Error_None;
    }
    catch (Error & e)
    {
        VOX_LOG_EXCEPTION(Severity_Fatal, e);

        return e.code;
    }
    catch (std::exception & e)
    {
        VOX_LOG_ERROR(Error_Unknown, LOG_CAT, e.what());

        return Error_Unknown;
    }
}

// --------------------------------------------------------------------
//  terminates the render
// --------------------------------------------------------------------
void voxServerEnd()
{ 
    VOX_LOG_INFO(LOG_CAT, "Terminating render service");

    filescope::renderer.reset();
}

// --------------------------------------------------------------------
//  Begins rendering the specified scene file
// --------------------------------------------------------------------
int voxServerStream(char const* filename)
{
    try
    {
        VOX_LOG_INFO(LOG_CAT, format("Loading scene file: %1%", filename));

        // Load the specified scene file
        auto & scene = filescope::scene;
        scene = Scene();
        scene.imprt(filename);
        if (!scene.volume) throw Error(__FILE__, __LINE__, LOG_CAT, "Scene is missing volume data", Error_MissingData);
        if (!scene.parameters)   scene.parameters   = RenderParams::create();
        if (!scene.clipGeometry) scene.clipGeometry = PrimGroup::create();
        if (!scene.lightSet)     scene.lightSet     = LightSet::create();
        if (!scene.transferMap)  scene.transferMap  = TransferMap::create(); // :TODO: Only required because of a bug
        if (!scene.transfer)     scene.transfer     = Transfer1D::create();
        if (!scene.camera)       scene.camera       = Camera::create();
        if (!scene.animator)     scene.animator     = Animator::create();

        // Begin rendering the scene
        filescope::renderController.render(
            filescope::renderer, filescope::scene, 100000);

        return Error_None;
    }
    catch (Error & e)
    {
        VOX_LOG_EXCEPTION(Severity_Error, e);

        return e.code;
    }
    catch (std::exception &)
    {
        return Error_Unknown;
    }
}

// --------------------------------------------------------------------
//  Returns the dot delimited version string for this build
// --------------------------------------------------------------------
char const* voxServerVersion() { return VSERV_VERSION_STRING; }
