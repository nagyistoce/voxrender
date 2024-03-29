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
#include "Session.h"

// Include Dependencies
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Core/Format.h"
#include "VoxLib/Bitmap/Bitmap.h"
#include "VoxServer/Base64.h"
#include "VoxServer/OpCodes.h"

// Include Scene Components
#include "VoxScene/RenderController.h"
#include "VoxScene/Volume.h"
#include "VoxScene/Camera.h"
#include "VoxScene/Light.h"
#include "VoxScene/RenderParams.h"
#include "VoxScene/PrimGroup.h"
#include "VoxScene/Transfer.h"

#include <chrono>

namespace vox {
    
// ----------------------------------------------------------------------------
//  Begins awaiting a new render session with a client over WebSocket protocol
// ----------------------------------------------------------------------------
Session::Session(UInt16 port, UInt64 key, char const* rootDir) :
    m_socket(m_ioService, port), m_rootDir(rootDir)
{
    // Configure the volume renderer
    m_renderer = vox::VolumeScatterRenderer::create();
    m_renderer->setRenderEventCallback(std::bind(&Session::onFrameReady, this, 
        std::placeholders::_1));
    
    // Configure the WebSocket
    m_socket.onConnected(std::bind(&Session::onConnect, this));
    m_socket.onClosed(std::bind(&Session::onClose, this));
    m_socket.onMessage(std::bind(&Session::onMessage, this, 
        std::placeholders::_1));
}

// :DEBUG:
void Session::start() 
{ 
    m_socket.start();
    m_ioService.run();
}

// ----------------------------------------------------------------------------
//  When the WebSocket has connected, upload the available scenes in the root
// ----------------------------------------------------------------------------
void Session::onConnect()
{
    auto path = m_rootDir.path.substr(1);

    using namespace boost::filesystem;
    if (!exists(path)) return;

    // Scene file directory listing
    String results;
    directory_iterator end_itr; // default construction yields past-the-end
    for (directory_iterator itr(path); itr != end_itr; ++itr)
    if (!is_directory(itr->status())) // Ignore subdirectories
    {
        auto name = itr->path().filename();
        if (name.extension() != ".xml") continue;
        if (!results.empty()) results.push_back('|');
        results = results + name.generic_string();
    }

    char opCode = (char)OpCode_DirList;
    results = String(&opCode, 1) + results;
    m_socket.write(results);

    // Filter lists
    std::ostringstream filterOs;
    std::list<std::shared_ptr<volt::Filter>> filters;
    volt::FilterManager::instance().getFilters(filters);
    if (filters.size())
    {
        char opCode = (char)OpCode_FilterList;
        filterOs << String(&opCode, 1);
        filterOs << "[";
        auto iter = filters.begin();
        filterOs << "\"" << (*iter)->name() << "\"";
        ++iter;
        while (iter != filters.end())
        {
            filterOs << ",\"" << (*iter)->name() << "\"";
            ++iter;
        }
        filterOs << "]";

        m_socket.write(filterOs.str());
    }
}

// ----------------------------------------------------------------------------
//  Callback event for when the WebSocket connection has recieved a message
// ----------------------------------------------------------------------------
void Session::onMessage(String const& message)
{
    try
    {
        if (message.size() < 0) 
            VOX_LOG_ERROR(Error_BadFormat, VOX_SERV_LOG_CAT,
                "Message is missing OpCode");
        auto opCode = (UInt8)message[0];

        switch (opCode)
        {
        case OpCode_Filter:
            VOX_LOG_DEBUG(VOX_SERV_LOG_CAT, "Recieved OpCode_Filter");
            applyFilter(message);
            break;
        case OpCode_BegStream:
            VOX_LOG_DEBUG(VOX_SERV_LOG_CAT, "Recieved OpCode_BegStream");
            unloadScene();
            loadScene(message);
            break;
        case OpCode_EndStream:
            VOX_LOG_DEBUG(VOX_SERV_LOG_CAT, "Recieved OpCode_EndStream");
            unloadScene();
            break;
        case OpCode_Update:
            VOX_LOG_DEBUG(VOX_SERV_LOG_CAT, "Recieved OpCode_Update");
            updateScene(message);
            break;
        default:
            throw Error(__FILE__, __LINE__, VOX_SERV_LOG_CAT,
                "Message contains invalid OpCode", Error_BadFormat);
        }
    }
    catch (std::exception & e)
    {
        VOX_LOG_ERROR(Error_Unknown, VOX_SERV_LOG_CAT, e.what());
    }
}

// ----------------------------------------------------------------------------
//  Loads the specified scene file and begins streaming on the WebSocket
// ----------------------------------------------------------------------------
void Session::loadScene(String const& message)
{
    if (message.size() < 2) throw Error(__FILE__, __LINE__, VOX_SERV_LOG_CAT,
        "Invalid BegStream message header", Error_BadFormat);
    
    auto * filename = message.c_str() + 1;

    VOX_LOG_INFO(VOX_SERV_LOG_CAT, format("Loading scene file: %1%", filename));

    ResourceId uri = m_rootDir.applyRelativeReference(ResourceId(filename));

    // Load the specified scene file
    m_scene = Scene::imprt(uri);
    m_scene->pad();

    // Extract the scene id tag for post back frames
    size_t filenameLen = strlen(filename);
    auto * idStr = filename + filenameLen + 1;
    m_id = String(idStr, strlen(idStr)) + "\x01"; // This is stupid but some javascript implementations are apparently flagrantly disregarding 
                                                  // standards so we use 0x01 as our null terminator character for seperating strings.
    
    // Return the scene data to the user (as JSON)
    auto flags = std::ios::in|std::ios::out|std::ios::binary;
    auto buffer = std::make_shared<std::stringbuf>(flags);
    ResourceStream outputStream(buffer);
    auto opCode = (Char)OpCode_Scene;
    outputStream << String(&opCode, 1) << m_id;
    m_scene->exprt(outputStream, OptionSet(), ".json");
    m_socket.write(buffer->str());

    // Begin rendering the scene
    m_renderController.render(m_renderer, m_scene, 100000);
}
    
// ----------------------------------------------------------------------------
//  Updates the internal scene data
// ----------------------------------------------------------------------------
void Session::updateScene(String const& message)
{
    // Parse the scene file specification
    auto buffer = std::make_shared<std::stringbuf>(message); // :TODO: Roll this out to work w/o copying
    buffer->sbumpc();
    ResourceStream sceneFile(buffer);
    auto scene = Scene::imprt(sceneFile, OptionSet(), ".json");

    // Update the local scene
    if (scene->camera) scene->camera->clone(*m_scene->camera);
}

// ------------------------------------------------------------------------
//  Terminates the render and unloads the active scene
// ------------------------------------------------------------------------
void Session::unloadScene()
{
    m_renderController.stop();
    m_scene.reset();
}

// ------------------------------------------------------------------------
//  Applies a filter to the volume
// ------------------------------------------------------------------------
void Session::applyFilter(String const& message)
{
    if (message.size() < 2) return;

    if (!m_renderController.isActive()) return;

    try
    {
        m_renderController.stop();
        
        std::list<std::shared_ptr<volt::Filter>> filters;
        volt::FilterManager::instance().getFilters(filters);
        BOOST_FOREACH (auto & filter, filters)
        if (filter->name() == &message[1])
        {
            filter->execute(*m_scene, OptionSet());
            break;
        }
    }
    catch (Error & error)
    {
        error.message = "Filtering error: " + error.message;

        VOX_LOG_EXCEPTION(Severity_Error, error);
    }
    catch (std::exception & error)
    {
        VOX_LOG_ERROR(Error_Unknown, VOX_SERV_LOG_CAT, String("Filtering error: ") + error.what());
    }
    
    // Send a scene update to the client

    // Begin rendering the scene
    m_renderController.render(m_renderer, m_scene, 100000);
}

// ------------------------------------------------------------------------
//  Callback function on frame ready
// ------------------------------------------------------------------------
void Session::onFrameReady(std::shared_ptr<vox::FrameBuffer> frame)
{
    //:TODO: - Write output directly to socket stream

    frame->data();
    
    //auto tbeg = std::chrono::high_resolution_clock::now();

    // :TODO: Permanent JPEG TX buffer
    std::ostringstream imageStream;
    auto buffer = std::shared_ptr<void>((void*)frame->data(), [] (void *) {});
    Bitmap image(Bitmap::Format_RGBX, frame->width(), frame->height(), 8, frame->stride(), buffer);
    image.exprt(imageStream, ".jpg");
    
    //auto tend = std::chrono::high_resolution_clock::now();
    //auto time = std::chrono::duration_cast<std::chrono::milliseconds>(tend-tbeg);

    auto opCode = (Char)OpCode_Frame;
    auto imageData = String(&opCode, 1) + m_id + "data:image/jpg;base64," + Base64::encode(imageStream.str());

    //VOX_LOG_INFO(VOX_SERV_LOG_CAT, format("TX: %1% ms", time.count()));
    //VOX_LOG_INFO(VOX_SERV_LOG_CAT, format("Size: %1% KB", imageStream.str().length() / 1024));

    m_socket.write(imageData);
}

} // namespace vox 