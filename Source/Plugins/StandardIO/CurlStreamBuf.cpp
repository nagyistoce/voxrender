/* ===========================================================================

    Project: StandardIO - Module definition for exported interface

    Description: A libcurl wrapper compatible with the VoxIO library

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
#include "CurlStreamBuf.h"

// Include Dependencies
#include "VoxLib/Error/PluginError.h"
#include "VoxLib/Core/Logging.h"

// 3rd Party Includes
#include <boost/lock
#include <boost/asio.hpp>

#define CHECK_CURL(x) if ((x) != CURLE_OK) { throw PluginError(__FILE__, __LINE__, SIO_LOG_CATEGORY, "call to libcurl api has returned error"); }

namespace vox {

namespace {
namespace filescope {
    
    // ----------------------------------------------------------------------------
    //  Helper class which encapsulates curl multihandle as an io service
    // ----------------------------------------------------------------------------
    class MultiWrapper
    {
    public:
        // ----------------------------------------------------------------------------
        //  Creates a multi handle and initializes the io service and socket callbacks 
        // ----------------------------------------------------------------------------
        MultiWrapper() : m_running(0), m_handle(curl_multi_init()), m_timer(m_ioService)
        {
            if (!m_handle) // Ensure proper initialization
            {  
                throw PluginError(__FILE__, __LINE__, SIO_LOG_CATEGORY, "curl_multi_init failed");
            }

            curl_multi_setopt(m_handle, CURLMOPT_TIMERDATA,  this);
            curl_multi_setopt(m_handle, CURLMOPT_SOCKETDATA, this);

            curl_multi_setopt(m_handle, CURLMOPT_SOCKETFUNCTION, socketCallbackRedirect);
            curl_multi_setopt(m_handle, CURLMOPT_TIMERFUNCTION,  multiTimerCallbackRedirect);
        }

        // ----------------------------------------------------------------------------
        //  Ensures cleanup of the multi handle
        // ----------------------------------------------------------------------------
        ~MultiWrapper()
        {
            if (m_handle) 
            {
                auto code = curl_multi_cleanup(m_handle); 

                VOX_LOGF(Severity_Error, Error_Unknown, SIO_LOG_CATEGORY, 
                    format("Call to curl_multi_cleanup failed: %1%", curl_multi_strerror(code)));

                m_handle = nullptr;
            }
        }
        
        // ----------------------------------------------------------------------------
        //  Returns the multi handle underlying this wrapper
        // ----------------------------------------------------------------------------
        CURLM * handle() 
        {
            return m_handle;
        }

        // ----------------------------------------------------------------------------
        //  Adds a new easy handle to the multihandle 
        // ----------------------------------------------------------------------------
        void addRequest(CURL * easyhandle)
        {
            CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_OPENSOCKETDATA, this));
            CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_CLOSESOCKETDATA, this));

            CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_OPENSOCKETFUNCTION,  openSocketRedirect));
            CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_CLOSESOCKETFUNCTION, closeSocketRedirect));

            CHECK_CURL(curl_multi_add_handle(m_handle, easyhandle));   
        }

    private:
        // ----------------------------------------------------------------------------
        //  Opens a socket for libcurl (CURLOPT_OPENSOCKETFUNCTION)
        // ----------------------------------------------------------------------------
        curl_socket_t openSocket(curlsocktype purpose, struct curl_sockaddr * address)
        {
            VOX_LOG_INFO(SIO_LOG_CATEGORY, "Opening socket");
 
            curl_socket_t sockfd = CURL_SOCKET_BAD;

            // Restriction to IPv4 for now
            if (purpose == CURLSOCKTYPE_IPCXN && address->family == AF_INET)
            {
                // Create a new TCP socket
                boost::asio::ip::tcp::socket * tcp_socket = new boost::asio::ip::tcp::socket(m_ioService);
 
                // Open to acquire a native handle for libcurl
                boost::system::error_code ec;
                tcp_socket->open(boost::asio::ip::tcp::v4(), ec);
 
                if (ec)
                {
                    VOX_LOGF(Severity_Error, Error_Unknown, SIO_LOG_CATEGORY, 
                        format("Failed to open socket: %1%", ec.message()));
                }
                else
                {
                    sockfd = tcp_socket->native_handle();
 
                    m_sockets.insert(std::pair<curl_socket_t, boost::asio::ip::tcp::socket *>(sockfd, tcp_socket));
                }
            }
            else
            {
                VOX_LOGF(Severity_Error, Error_Unknown, SIO_LOG_CATEGORY, "Failed to open socket: IPv6 not supported");
            }

            return sockfd;
        }

        // ----------------------------------------------------------------------------
        //  Closes a previously opened socket for libcurl (CURLOPT_CLOSESOCKETFUNCTION)
        // ----------------------------------------------------------------------------
        int closeSocket(curl_socket_t item)
        {
            VOX_LOG_INFO(SIO_LOG_CATEGORY, format("Closing socket: %1%", item));

            auto iter = m_sockets.find(item);
            if (iter != m_sockets.end())
            {
                delete iter->second;
                m_sockets.erase(iter);
            }
            else
            {
                VOX_LOGF(Severity_Error, Error_Unknown, VOX_LOG_CATEGORY, 
                    format("libcurl requested closure of unknown socket: %1%", item));
            }

            return 0;
        }

        // ----------------------------------------------------------------------------
        // Check for completed transfers, and remove their easy handles
        // ----------------------------------------------------------------------------
        void checkInfo()
        {
            VOX_LOG_INFO(SIO_LOG_CATEGORY, format("Requests remaining: %1%", m_running));

            int messages; ///< Message count
 
            while (auto * msg = curl_multi_info_read(m_handle, &messages))
            {
                if (msg->msg == CURLMSG_DONE)
                {
                    auto easyhandle = msg->easy_handle;
                    auto result     = msg->data.result;

                    CurlIStreamBuf * request;
                    curl_easy_getinfo(easyhandle, CURLINFO_PRIVATE, &request);

                    char * effectiveUrl;
                    curl_easy_getinfo(easyhandle, CURLINFO_EFFECTIVE_URL, &effectiveUrl);
                    
                    VOX_LOG_INFO(SIO_LOG_CATEGORY, "Request complete: ");

                    //request->cleanup();
                }
            }
        }

        // ----------------------------------------------------------------------------
        //  Called by boost::asio timer when the libcurl specified timeout occurs
        // ----------------------------------------------------------------------------
        void timerCallback(boost::system::error_code const& error)
        {
            if (!error)
            {
                auto code = curl_multi_socket_action(m_handle, CURL_SOCKET_TIMEOUT, 0, &m_running);
                if (code != CURLM_OK)
                {
                    VOX_LOGF(Severity_Error, Error_Unknown, SIO_LOG_CATEGORY, 
                        format("Call to curl_multi_socket_action failed: %1%", curl_multi_strerror(code)));
                }

                checkInfo();
            }
            else
            {
                // :TODO: Log error
            }
        }

        // ----------------------------------------------------------------------------
        //  Updates the event timer with the timeout issued by libcurl
        // ----------------------------------------------------------------------------
        int multiTimerCallback(long timeoutMs)
        {
            VOX_LOG_INFO(SIO_LOG_CATEGORY, format("MultiTimer Callback: %1% ms", timeoutMs));

            m_timer.cancel();
 
            if (timeoutMs > 0)
            {
                // Reset the timer to the specified timeout 
                m_timer.expires_from_now(boost::posix_time::milliseconds(timeoutMs));
                m_timer.async_wait(boost::bind(&MultiWrapper::timerCallback, this, _1));
            }
            else
            {
                // Call timeout function immediately 
                boost::system::error_code error; // Success
                timerCallback(error);
            }

            return 0;
        }
        
        // ----------------------------------------------------------------------------
        //  Event callback from asio on socket event completion
        // ----------------------------------------------------------------------------
        void eventCallback(boost::asio::ip::tcp::socket * tcpSocket, int action)
        {
            auto code = curl_multi_socket_action(m_handle, tcpSocket->native_handle(), action, &m_running);
            if (code != CURLM_OK)
            {
                VOX_LOGF(Severity_Error, Error_Unknown, SIO_LOG_CATEGORY, 
                    format("Call to curl_multi_socket_action failed: %1%", curl_multi_strerror(code)));
            }

            checkInfo(); ///< Process request completions
 
            if (m_running <= 0) m_timer.cancel();
        }

        // ----------------------------------------------------------------------------
        //  Waits for the specified option on the socket specified by libcurl
        // ----------------------------------------------------------------------------
        void socketCallback(curl_socket_t s, int action)
        {
            // Acquire the handle to the overhead asio tcp socket
            auto iter = m_sockets.find(s);
            if (iter == m_sockets.end())
            {
                VOX_LOG_INFO(SIO_LOG_CATEGORY, "Socket is c-ares socket; ignoring.");

                return;
            }
            auto * tcpSocket = iter->second;
 
            // Wait on the socket for the specified action/event/status
            switch (action)
            {
            case CURL_POLL_IN:
                tcpSocket->async_read_some(boost::asio::null_buffers(), boost::bind(&MultiWrapper::eventCallback, this, tcpSocket, action));
                break;
                
            case CURL_POLL_OUT:
                tcpSocket->async_write_some(boost::asio::null_buffers(), boost::bind(&MultiWrapper::eventCallback, this, tcpSocket, action));
                break;
                
            case CURL_POLL_INOUT:
                tcpSocket->async_read_some(boost::asio::null_buffers(), boost::bind(&MultiWrapper::eventCallback, this, tcpSocket, action));
                tcpSocket->async_write_some(boost::asio::null_buffers(), boost::bind(&MultiWrapper::eventCallback, this, tcpSocket, action));
                break;

            default:
                VOX_LOGF(Severity_Error, Error_Bug, SIO_LOG_CATEGORY, 
                    format("Unrecognized action requested on libcurl socket: %1%", action));
            }
        }

    private: // Libcurl Callback Redirects 

        // ----------------------------------------------------------------------------
        //  Updates the even timer after a curl_multi_* call
        // ----------------------------------------------------------------------------
        static int multiTimerCallbackRedirect(CURLM *multihandle, long timeoutMs, void * userp)
        {
            return reinterpret_cast<MultiWrapper*>(userp)->multiTimerCallback(timeoutMs);
        }

        // ----------------------------------------------------------------------------
        //  Closes a previously opened socket for libcurl (CURLOPT_CLOSESOCKETFUNCTION)
        // ----------------------------------------------------------------------------
        static int closeSocketRedirect(void * clientp, curl_socket_t item)
        {
            return reinterpret_cast<MultiWrapper*>(clientp)->closeSocket(item);
        }
        
        // ----------------------------------------------------------------------------
        //  Opens a socket for libcurl (CURLOPT_OPENSOCKETFUNCTION)
        // ----------------------------------------------------------------------------
        static curl_socket_t openSocketRedirect(void * clientp, curlsocktype purpose, struct curl_sockaddr * address)
        {
            return reinterpret_cast<MultiWrapper*>(clientp)->openSocket(purpose, address);
        }

        // ----------------------------------------------------------------------------
        //  Socket callback from libcurl about socket status change (CURLMOPT_SOCKETFUNCTION)
        // ----------------------------------------------------------------------------
        static int socketCallbackRedirect(CURL * easyhandle, curl_socket_t s, int action, void *userp, void *sockp)
        {
            static const char * actionStr[] = {"none", "IN", "OUT", "INOUT", "REMOVE"};

            VOX_LOG_INFO(SIO_LOG_CATEGORY, format("Socket callback: socket=%1%, action=%2%", s, actionStr[action]));

            if (action == CURL_POLL_REMOVE) return 0;

            reinterpret_cast<MultiWrapper*>(userp)->socketCallback(s, action);

            return 0;
        }

    private:
        std::map<curl_socket_t, boost::asio::ip::tcp::socket *> m_sockets;

        boost::asio::io_service     m_ioService;
        boost::asio::deadline_timer m_timer; 

        boost::mutex m_mutex;

        CURLM * m_handle;     ///< Curl multi handle
        int     m_running;    ///< Active requests
    };

    boost::thread_specific_ptr<MultiWrapper> multiWrapper;

} // namespace filescope
} // namespace anonymous

// ----------------------------------------------------------------------------
//  Initializes a request, creating and registering the curl handle
// ----------------------------------------------------------------------------
CurlOStreamBuf::CurlOStreamBuf(
    ResourceId &     identifier,
    OptionSet const& options
    )
{
    // Create the libcurl request handle
    if (!(m_easyhandle = curl_easy_init()))
    {
        throw PluginError(__FILE__, __LINE__, SIO_LOG_CATEGORY,
            "Failed to initialize handle for libCurl session");
    }
    
    // Set the request options using the libcurl client API (see http://curl.haxx.se/libcurl/c/curl_easy_setopt.html)
    CHECK_CURL(curl_easy_setopt(m_easyhandle, CURLOPT_USERAGENT,      "voxlib-agent/" SIO_VERSION_STRING));
    CHECK_CURL(curl_easy_setopt(m_easyhandle, CURLOPT_URL,            identifier.asString())); 
    CHECK_CURL(curl_easy_setopt(m_easyhandle, CURLOPT_FOLLOWLOCATION, 1L));
    CHECK_CURL(curl_easy_setopt(m_easyhandle, CURLOPT_UPLOAD,         1L));
    CHECK_CURL(curl_easy_setopt(m_easyhandle, CURLOPT_READDATA,       this));
    CHECK_CURL(curl_easy_setopt(m_easyhandle, CURLOPT_READFUNCTION,   &CurlOStreamBuf::onGetDataCallback));
    CHECK_CURL(curl_easy_setopt(m_easyhandle, CURLOPT_PRIVATE,        this));

    // The Size option is a generic option across all protocols
    auto sizeIter = options.find("Size");
    if (sizeIter != options.end())
    {
        auto size = boost::lexical_cast<size_t>(sizeIter->second);
        CHECK_CURL(curl_easy_setopt(m_easyhandle, CURLOPT_INFILESIZE_LARGE, size));
    }
}

// ----------------------------------------------------------------------------
//  Terminates the session if it is still active
// ----------------------------------------------------------------------------
void CurlOStreamBuf::cleanup()
{
}

// ----------------------------------------------------------------------------
//  Copies data from a libcurl request into the internal memory buffer
// ----------------------------------------------------------------------------
size_t CurlOStreamBuf::onGetData(void * ptr, size_t maxBytes)
{
    return 0;
}

// ----------------------------------------------------------------------------
//  Redirects the data callback to the associated request object
// ----------------------------------------------------------------------------
size_t CurlOStreamBuf::onGetDataCallback(void* ptr, size_t size, size_t nmemb, void* buf)
{
    return reinterpret_cast<CurlOStreamBuf*>(buf)->onGetData(ptr, size*nmemb);
}

// ----------------------------------------------------------------------------
//  Initializes a request, creating and registering the curl handle
// ----------------------------------------------------------------------------
CurlIStreamBuf::CurlIStreamBuf(
    ResourceId &     identifier,
    OptionSet const& options
    ) 
{
    // Initialize the buf ptrs to ensure immediate underflow/overflow 
    setg(nullptr, nullptr, nullptr); 
    setp(nullptr, nullptr, nullptr);

    // Create the libcurl request handle
    if (!(m_easyhandle = curl_easy_init()))
    {
        throw PluginError(__FILE__, __LINE__, SIO_LOG_CATEGORY,
            "Failed to initialize handle for libCurl session");
    }

    // Set the request options using the libcurl client API (see http://curl.haxx.se/libcurl/c/curl_easy_setopt.html)
    CHECK_CURL(curl_easy_setopt(m_easyhandle, CURLOPT_USERAGENT,        "voxlib-agent/" SIO_VERSION_STRING));
    CHECK_CURL(curl_easy_setopt(m_easyhandle, CURLOPT_URL,              identifier.asString().c_str()));
    CHECK_CURL(curl_easy_setopt(m_easyhandle, CURLOPT_FOLLOWLOCATION,   1L));
    CHECK_CURL(curl_easy_setopt(m_easyhandle, CURLOPT_WRITEDATA,        this));
    CHECK_CURL(curl_easy_setopt(m_easyhandle, CURLOPT_WRITEFUNCTION,    &CurlIStreamBuf::onDataReadyCallback));
    CHECK_CURL(curl_easy_setopt(m_easyhandle, CURLOPT_PRIVATE,          this));

    // Issue the request to the Multi-Interface wrapper
    if (filescope::multiWrapper.get() == nullptr) 
    {
        filescope::multiWrapper.reset(new filescope::MultiWrapper());
    }
    filescope::multiWrapper->addRequest(m_easyhandle);
}

// ----------------------------------------------------------------------------
//  Underflow method which awaits a chunk and resets the read head
// ----------------------------------------------------------------------------
int CurlIStreamBuf::underflow()
{
    boost::mutex::scoped_lock lock(m_mutex);

    // Pop the old buffer from the list
    if (!m_data.empty()) 
    {
        m_data.front().free();
        m_data.pop_front();
    }

    while (m_data.empty()) 
    {
       
    }

    // :TODO: Check exception_ptr for error

    if (m_data.front().size == 0) return EOF; /// nullbuf indicates eof

    // Reset the get pointers to the next buffer chunk
    char* data = reinterpret_cast<char*>(m_data.front().data);
    size_t len = m_data.front().size;
    setg(data, data, data+len);

    return *data;
}

// ----------------------------------------------------------------------------
//  Copies data from a libcurl request into the internal memory buffer
// ----------------------------------------------------------------------------
size_t CurlIStreamBuf::onDataReady(char * ptr, size_t bytes)
{
    boost::mutex::scoped_lock lock(m_mutex);

    m_data.push_back( DataBuffer(ptr, bytes) );

    return bytes;
}

// ----------------------------------------------------------------------------
//  Redirects the data callback to the associated request object
// ----------------------------------------------------------------------------
size_t CurlIStreamBuf::onDataReadyCallback(char * ptr, size_t size, size_t nmemb, void * buf)
{
    return reinterpret_cast<CurlIStreamBuf*>(buf)->onDataReady(ptr, size*nmemb);
}
    
} // namespace vox