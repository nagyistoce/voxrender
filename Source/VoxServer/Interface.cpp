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
#include "VoxServer/WebSocket.h"
#include "VoxServer/Base64.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Core/System.h"
#include "VoxLib/Plugin/PluginManager.h"
#include "VoxLib/IO/ResourceHelper.h"
#include "VoxLib/Bitmap/Bitmap.h"

#include <boost/asio.hpp>
#include <boost/bind.hpp>

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

using namespace vox;
using boost::asio::ip::tcp;

namespace {
namespace filescope {

    std::shared_ptr<VolumeScatterRenderer> renderer;

    boost::asio::io_service ioService;
    std::shared_ptr<WebSocket> webSocket;

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
        VOX_LOG_INFO(VOX_SERV_LOG_CAT, "RENDER FRAME CALLBACK");

        auto message = format("Render Time %1%", renderer->renderTime());

        frame->data();

        std::ostringstream imageStream;
        auto buffer = std::shared_ptr<void>((void*)frame->data(), [] (void *) {});
        Bitmap image(Bitmap::Format_RGBX, frame->width(), frame->height(), 8, frame->stride(), buffer);
        image.exprt(imageStream, ".png");

        auto imageData = "data:image/png;base64," + Base64::encode(imageStream.str());

        filescope::webSocket->write(imageData);

        boost::this_thread::sleep(boost::posix_time::milliseconds(50));
    }
    
    // ------------------------------------------------------------------------
    //  Begins streaming the specified file
    // ------------------------------------------------------------------------
    void beginStream(String const& filename)
    {
        VOX_LOG_INFO(VOX_SERV_LOG_CAT, format("Loading scene file: %1%", filename));

        // Load the specified scene file
        auto & scene = filescope::scene;
        scene = Scene::imprt(filename);
        if (!scene.volume) throw Error(__FILE__, __LINE__, VOX_SERV_LOG_CAT, "Scene is missing volume data", Error_MissingData);
        if (!scene.parameters)   scene.parameters   = RenderParams::create();
        if (!scene.clipGeometry) scene.clipGeometry = PrimGroup::create();
        if (!scene.lightSet)     scene.lightSet     = LightSet::create();
        if (!scene.transferMap)  scene.transferMap  = TransferMap::create(); // :TODO: Only required because of a bug
        if (!scene.transfer)     scene.transfer     = Transfer1D::create();
        if (!scene.camera)       scene.camera       = Camera::create();
        if (!scene.animator)     scene.animator     = Animator::create();

        // Begin rendering the scene
        filescope::renderController.render(filescope::renderer, filescope::scene, 100000);
    }

    // ------------------------------------------------------------------------
    //  Connected event callback from WebSocket
    // ------------------------------------------------------------------------
    void onConnect()
    {
        try
        {
            beginStream("file:///C:/Users/Lucas/Documents/Projects/voxrender/trunk/Models/Examples/smoke.xml");
        }
        catch (Error & error)
        {
            VOX_LOG_EXCEPTION(Severity_Error, error);
        }
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
    
    filescope::renderController.stop();
    filescope::webSocket.reset();
    PluginManager::instance().unloadAll();
    filescope::logFileStream.close();

    filescope::renderer.reset();
}

// --------------------------------------------------------------------
//  Begins rendering the specified scene file
// --------------------------------------------------------------------
int voxServerBeginStream(uint16_t * portOut, uint64_t * keyOut)
{
    // Select a port and key for the stream
    UInt16 port = 8000;
    UInt64 key  = 0;
    
    // Prepare the WebSocket to accept an incoming connection
    try
    {
        filescope::webSocket = std::make_shared<WebSocket>(filescope::ioService, port);
        filescope::webSocket->onConnected(&filescope::onConnect);
        //filescope::webSocket->onClosed();
        //filescope::webSocket->onMessage();

        filescope::webSocket->start();

        filescope::ioService.run();
    }
    catch (std::exception& e)
    {
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
