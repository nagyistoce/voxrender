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

// Begin definition
#ifndef VOX_SESSION_H
#define VOX_SESSION_H

// Include Dependencies
#include "VoxServer/Common.h"
#include "VoxServer/WebSocket.h"
#include "VoxLib/Core/Types.h"
#include "VoxLib/Core/Functors.h"
#include "VoxScene/RenderController.h"

// Standard Renderers for the Application
#include "VolumeScatterRenderer/Core/VolumeScatterRenderer.h"

#include <boost/asio.hpp>

namespace vox {

/** Manages a session with a single client */
class Session
{
public:
    /** Initializes a new render session */
    Session(UInt16 port, UInt64 key, char const* rootDir = "");

    void start();

private:
    /** Callback from WebSocket with client control messages */
    void onMessage(String const& message);

    /** Callback from WebSocket notifying the socket is CONNECTED */
    void onConnect();

    /** Callback from the WebSocket notifying the socket is CLOSED */
    void onClose() { }

    /** Callback from render thread for frame ready event */
    void onFrameReady(std::shared_ptr<FrameBuffer> frame);

    /** Imports and begins rendering the specified scene file */
    void loadScene(String const& message);

    /** Unloads the current scenefile and stops the render */
    void unloadScene();

private:
    boost::asio::io_service m_ioService; ///< IO service object for this session
    WebSocket m_socket; ///< Underlying websocket for the connection

    Scene m_scene;
    std::shared_ptr<VolumeScatterRenderer> m_renderer;
    RenderController m_renderController;

    ResourceId m_rootDir;       ///< Root directory to which the client has permissions
    UInt64     m_key;           ///< Access key for the client's websocket connection
    bool       m_authenticated; ///< Tracks if the user has authenticated themselves
};

} // namespace vox

#endif // VOX_SESSION_H
