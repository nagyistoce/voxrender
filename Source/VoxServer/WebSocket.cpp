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
#include "WebSocket.h"

// Include Dependencies
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Core/Format.h"
#include "VoxServer/Base64.h"

// Boost SHA1 Hash (its ::detail so keep an eye on it)
#include <boost/uuid/sha1.hpp>

namespace vox {

using boost::asio::ip::tcp;

// Filescope functions
namespace {
namespace filescope {

    // Computes a SHA1 hash of the input
    std::string sha1_hash(std::string const& s)
    {
        UInt32 hash[5];
        boost::uuids::detail::sha1 sha;
        sha.process_bytes(s.c_str(), s.size());
        sha.get_digest(hash);

        for (int i = 0; i < 5; i++) hash[i] = htonl(hash[i]);

        return std::string((char*)hash, sizeof(unsigned int)*5);
    }

    // Performs 8 byte network endian conversion
    UInt64 ntohll(UInt64 value)
    {
        static const int num = 42;

        // Check the endianness
        if (*reinterpret_cast<const char*>(&num) == num)
        {
            const UInt32 high_part = htonl(static_cast<UInt32>(value >> 32));
            const UInt32 low_part  = htonl(static_cast<UInt32>(value & 0xFFFFFFFFLL));

            return (static_cast<UInt64>(low_part) << 32) | high_part;
        } 
        else return value;
    }
    
    // Performs 8 byte network endian conversion
    UInt64 htonll(UInt64 value)
    {
        return ntohll(value);
    }

    static const UInt8 FIN_MASK = 0x80;
    static const UInt8 OPC_MASK = 0x0F;
    static const UInt8 PAY_MASK = 0x7F;
    static const UInt8 MASK_BIT = 0x80;

} // namespace filescope
} // namespace anonymous

// ------------------------------------------------------------------------
//  Uses the mask and the transformed message to compute the original
// ------------------------------------------------------------------------
void WebSocket::unmaskMessage()
{
    if (!m_masking) return;

    size_t chunks = m_message.size() / 4;
    UInt32 * iter1 = (UInt32*)&m_message[0];
    for (size_t i = 0; i < chunks; ++i, ++iter1)
    {
        *iter1 ^= m_mask;
    }
        
    size_t bits = m_message.size() - chunks*4;
    size_t offb = chunks * 4;
    UInt8 * iter2 = (UInt8*)&m_message[0] + offb;
    UInt8 * masks = (UInt8*)&m_mask;
    for (size_t i = 0; i < bits; ++i)
    {
        iter2[i] ^= masks[i];
    }
}

// ------------------------------------------------------------------------
//  Waits for the initial handshake request on the WebSocket connection
// ------------------------------------------------------------------------
void WebSocket::handleAccept(boost::system::error_code const& error)
{
    m_socket.async_read_some(
        boost::asio::buffer(m_dataBuf, max_length),
        boost::bind(&WebSocket::handleConnect, this,
                    boost::asio::placeholders::error,
                    boost::asio::placeholders::bytes_transferred));
}

// ------------------------------------------------------------------------
//  Handles the initial handshake of the WebSocket connection
// ------------------------------------------------------------------------
void WebSocket::handleConnect(boost::system::error_code const& error, size_t bytes)
{
    std::string const KEY_HEADER = "Sec-WebSocket-Key: ";
    std::string const END_LINE   = "\r\n";
    std::string const MAGIC_STR  = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    std::string const RESP_STR   = 
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Accept: ";

    if (!error)
    {
        std::string requestText(m_dataBuf, bytes);

        VOX_LOG_DEBUG(VOX_SERV_LOG_CAT, format("WebSocket Client Handshake: %1%", requestText));

        // Extract the Base64 encoded key in the Sec-WebSocket-Key header
        auto offHead = requestText.find(KEY_HEADER);
        offHead += KEY_HEADER.size();
        auto offKey = requestText.find(END_LINE, offHead);
        auto inKey = requestText.substr(offHead, offKey-offHead); 
            
        // Generate a response based on the request key
        auto key = filescope::sha1_hash(inKey + MAGIC_STR);
        auto response = RESP_STR + Base64::encode(key) + END_LINE + END_LINE;
        if (response.size() > max_length) return;
        memcpy(m_dataBuf, response.c_str(), response.size());
        
        VOX_LOG_DEBUG(VOX_SERV_LOG_CAT, format("WebSocket Server Response: %1%", response));

        // Write the handshake response to the client
        boost::asio::async_write(m_socket,
            boost::asio::buffer(m_dataBuf, response.size()),
            boost::bind(&WebSocket::handleConnected, this,
            boost::asio::placeholders::error));
    }
    else
    {
        VOX_LOG_ERROR(Error_Unknown, VOX_SERV_LOG_CAT, error.message());
    }
}
    
// ------------------------------------------------------------------------
//  Processes framed packets over the socket into coherent messages
//  https://tools.ietf.org/html/rfc6455#page-28
// ------------------------------------------------------------------------
void WebSocket::handleRead(boost::system::error_code const& error, size_t bytes)
{
    if (!error)
    {
        auto bytePtr = m_dataBuf;
        m_bytesLeft += bytes;

        #define CLOMP(X) m_bytesLeft -= X; bytePtr += X;

        // Header byte contains OpCode and FIN bit
        if (m_bytesLeft && m_status == Status_Head)
        {
            m_isLastFrame = (bool)((*bytePtr) & filescope::FIN_MASK); 
            m_opCode      = (int) ((*bytePtr) & filescope::OPC_MASK);

            m_status = Status_Pay;

            CLOMP(1)
        }

        // Second byte contains payload length data
        if (m_bytesLeft && m_status == Status_Pay)
        {
            m_masking       = (bool)((*bytePtr) & filescope::MASK_BIT); 
            m_payloadLength = (int) ((*bytePtr) & filescope::PAY_MASK);

            if      (m_payloadLength == 126) m_status = Status_Pay16;
            else if (m_payloadLength == 127) m_status = Status_Pay64;
            else m_status = m_masking ? Status_Mask : Status_Read;

            CLOMP(1)
        }

        // Payload length extension for 16 bit payload
        if (m_bytesLeft >= 2 && m_status == Status_Pay16)
        {
            UInt16 length = *(UInt16*)(bytePtr);
            m_payloadLength = (UInt64)ntohs(length);

            m_status = m_masking ? Status_Mask : Status_Read;

            CLOMP(2)
        }

        // Payload length extension for 64 bit payload
        if (m_bytesLeft >= 8 && m_status == Status_Pay64)
        {
            UInt64 length = *(UInt64*)(bytePtr);
            m_payloadLength = filescope::ntohll(length);

            m_status = m_masking ? Status_Mask : Status_Read;

            CLOMP(8)
        }

        // Read the payload content from the data buffer
        if (m_bytesLeft >= 4 && m_status == Status_Mask)
        {
            m_mask = *(UInt32*)(bytePtr);

            m_status = Status_Read;

            CLOMP(4)
        }

        // Read the payload content from the data buffer
        if (m_bytesLeft && m_status == Status_Read)
        {
            auto bytesToRead = low<UInt64>(m_bytesLeft, m_payloadLength);
            m_payloadLength -= bytesToRead;
            m_message += std::string(bytePtr, bytesToRead);

            CLOMP(bytesToRead);

            // Detect EOF condition
            if (m_payloadLength == 0)
            {
                if (m_isLastFrame)
                {
                    unmaskMessage();
                    VOX_LOG_DEBUG(VOX_SERV_LOG_CAT, format("Message Recieved: %1%", m_message));
                    if (m_messageCallback) m_messageCallback(m_message);
                    m_message.clear();
                }

                m_status = Status_Head;
            }
        }

        // Shift the remaining bits back to the buffer head
        if (m_bytesLeft)
        {
            char temp[max_length];
            memcpy(temp, bytePtr, m_bytesLeft);
            memcpy(m_dataBuf, temp, m_bytesLeft);
        }

        // Check if we have also read the beginning of a new message
        if (m_bytesLeft && m_status == Status_Head)
        {
            handleRead(error, 0);
        }
        else // Await the next message packet from the server
        {
            m_socket.async_read_some(
                boost::asio::buffer(m_dataBuf + m_bytesLeft, max_length - m_bytesLeft),
                boost::bind(&WebSocket::handleRead, this,
                            boost::asio::placeholders::error,
                            boost::asio::placeholders::bytes_transferred));
        }
    }
    else
    {
        VOX_LOG_ERROR(Error_Unknown, VOX_SERV_LOG_CAT, error.message());
    }
}
    
// ------------------------------------------------------------------------
//  Writes a block of data to the WebSocket
// ------------------------------------------------------------------------
void WebSocket::write(std::shared_ptr<UInt8> data, size_t bytes)
{
}
  
// ------------------------------------------------------------------------
//  Writes a block of data to the WebSocket
// ------------------------------------------------------------------------
void WebSocket::write(String const& message, bool isBinary)
{
    std::ostringstream os;

    // Header byte (FIN bit, extensions, and OpCode)
    UInt8 header = 0;
    header |= filescope::FIN_MASK;                    // Single frame message
    header |= isBinary ? OpCode_Binary : OpCode_Text; // Message type
    os.write((char*)&header, 1);
    
    // Initial padding byte
    UInt8 initialFrame = 0;
    int length = 0;
    if (message.size() <= 125)
    {
        length = 0;
        initialFrame |= (UInt8)message.size();
    }
    else
    {
        bool isLong = message.size() > std::numeric_limits<UInt16>::max();
        length = isLong ? 2 : 1;
        initialFrame |= isLong ? 0x7F : 0x7E;
    }
    os.write((char*)&initialFrame, 1);

    // Extended padding bytes
    if (length == 1) 
    {
        UInt16 size = htons((UInt16)message.size());
        os.write((char*)&size, sizeof(UInt16));
    }
    else if (length == 2) 
    {
        UInt64 size = filescope::htonll((UInt64)message.size());
        os.write((char*)&size, sizeof(UInt64));
    }

    // Message content
    os.write(&message[0], message.size());

    // Write the frame message
    auto framedMessage = os.str();
    m_socket.write_some(boost::asio::buffer(
        &framedMessage[0], framedMessage.size()));
}

// ------------------------------------------------------------------------
//  Verifies that the initial handshake response was sent successfully
// ------------------------------------------------------------------------
void WebSocket::handleConnected(boost::system::error_code const& error)
{
    if (!error)
    {
        if (m_connectCallback) m_connectCallback();

        m_socket.async_read_some(boost::asio::buffer(m_dataBuf, max_length),
            boost::bind(&WebSocket::handleRead, this,
            boost::asio::placeholders::error,
            boost::asio::placeholders::bytes_transferred));
    }
    else
    {
        VOX_LOG_ERROR(Error_Unknown, VOX_SERV_LOG_CAT, error.message());
    }
}

} // namespace vox