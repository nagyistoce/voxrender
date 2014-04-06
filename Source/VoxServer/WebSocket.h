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
#ifndef VOX_WEB_SOCKET_H
#define VOX_WEB_SOCKET_H

// Include Dependencies
#include "VoxServer/Common.h"
#include "VoxLib/Core/Types.h"
#include "VoxLib/Core/Functors.h"

// Boost dependencies
#include <boost/bind.hpp>
#include <boost/asio.hpp>

namespace vox {

/** 
 * Manages a WebSocket protocol connection to a client.
 * See https://tools.ietf.org/html/rfc6455
 */
class WebSocket
{
public:
    /** Initializes a new session */
    WebSocket(boost::asio::io_service & io_service, UInt16 port) : 
        m_socket(io_service), 
        m_acceptor(io_service, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port)),
        m_status(Status_Head), 
        m_bytesLeft(0)
    {
    }
    
    /** Returns the TCP socket associated with this session */
    boost::asio::ip::tcp::socket & socket() { return m_socket; }
    
    /** Awaits the client connection on the specified port */
    void start()
    {
        m_acceptor.async_accept(m_socket,
            boost::bind(&WebSocket::handleAccept, this,
                        boost::asio::placeholders::error));
    }

    /** Writes a block of data to the WebSocket */
    void write(std::shared_ptr<UInt8> data, size_t bytes);

    /** Writes a text message to the web socket */
    void write(String const& message, bool isBinary = true);

    /** Closes the WebSocket and the underlying connection */
    void close(String const& message = "");

    /** Sends a pong control message */
    void pong(String const& message = "");

    /** Registers the callback for when WebSocket status is CONNECTED */
    void onConnected(std::function<void()> callback) { m_connectCallback = callback; }

    /** Registers the callback for a message event on the WebSocket */
    void onMessage(std::function<void(String const&)> callback) { m_messageCallback = callback; }
    
    /** Registers the callback for when WebSocket status is CLOSED */
    void onClosed(std::function<void()> callback) { m_closedCallback = callback; }

private:
    /** Uses the mask and the transformed message to compute the original message */
    void unmaskMessage();

    /** Waits for the initial handshake request on the WebSocket connection */
    void handleAccept(boost::system::error_code const& error);

    /** Processes and responds to the initial handshake for the WebSocket connection */
    void handleConnect(boost::system::error_code const& error, size_t bytes);
    
    /** Verifies that the initial handshake response was carried out successfully */
    void handleConnected(boost::system::error_code const& error);
    
    /** Processes framed packets sent over the socket into coherent messages */
    void handleRead(boost::system::error_code const& error, size_t bytes);

    /** Handles an invalid request from the client */
    void handleError();

private:
    boost::asio::ip::tcp::socket   m_socket;   ///< TCP socket for the WebSocket connection
    boost::asio::ip::tcp::acceptor m_acceptor; ///< TCP socket acceptor for detecting clients
    
    /** OpCodes for message content */
    enum OpCode
    {
        OpCode_Text   = 0,
        OpCode_Binary = 1, 
        OpCode_Close  = 8,
        OpCode_Ping   = 9,
        OpCode_Pong   = 10
    };

    /** Read status on the current message */
    enum Status
    {
        Status_Head,    // Read header byte
        Status_Pay,     // Read initial payload byte
        Status_Pay16,   // Read 16 bit payload
        Status_Pay64,   // Read 64 bit payload
        Status_Mask,    // Read mask
        Status_Read     // Read content
    };

    Status m_status; ///< Read status for the current message

    UInt64 m_payloadLength; ///< Length of the current message payload
    UInt32 m_mask;          ///< The mask for the payload bytes
    int    m_opCode;        ///< OpCode of the current message
    bool   m_isLastFrame;   ///< Marks the last frame of a message
    bool   m_masking;       ///< Denotes whether the current message uses masking

    std::string m_message; ///< Buffer for deframing messages

    std::function<void(String const&)> m_messageCallback;
    std::function<void()>              m_connectCallback;
    std::function<void()>              m_closedCallback;

    enum { max_length = 1024 };
    char m_dataBuf[max_length]; ///< Buffer for raw socket read data (framed data)
    unsigned int m_bytesLeft;   ///< Number of bytes left in the read buffer
};

} // namespace vox

#endif // VOX_WEB_SOCKET_H
