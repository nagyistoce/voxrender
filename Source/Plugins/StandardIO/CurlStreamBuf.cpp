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
#include "VoxLib/Error/Error.h"
#include "VoxLib/Core/Logging.h"

// 3rd Party Includes
#include <boost/lockfree/queue.hpp>
#include <boost/asio.hpp>

#define CHECK_CURL(x) if ((x) != CURLE_OK) { throw Error(__FILE__, __LINE__, SIO_LOG_CATEGORY, "call to libcurl api has returned error"); }

namespace vox {

namespace {
namespace filescope {
    
    // ----------------------------------------------------------------------------
    //  IO service class which encapsulates the curl multi interface with asio
    // ----------------------------------------------------------------------------
    class CurlAsyncIOService
    {
    public:
        // ----------------------------------------------------------------------------
        //  Creates a multi handle and initializes the io service and socket callbacks 
        // ----------------------------------------------------------------------------
        CurlAsyncIOService() : 
          m_running(0), 
          m_timer(m_ioService), 
          m_work(m_ioService), 
          m_serviceThread(boost::bind(&CurlAsyncIOService::ioServiceLoop, this))
        {
            // Initialize the multihandle
            m_handle = curl_multi_init();
            if (!m_handle)
            {  
                throw Error(__FILE__, __LINE__, SIO_LOG_CATEGORY, "curl_multi_init failed");
            }

            curl_multi_setopt(m_handle, CURLMOPT_TIMERDATA,  this);
            curl_multi_setopt(m_handle, CURLMOPT_SOCKETDATA, this);

            curl_multi_setopt(m_handle, CURLMOPT_SOCKETFUNCTION, socketCallbackRedirect);
            curl_multi_setopt(m_handle, CURLMOPT_TIMERFUNCTION,  multiTimerCallbackRedirect);
        }

        // ----------------------------------------------------------------------------
        //  Ensures cleanup of the multi handle
        // ----------------------------------------------------------------------------
        ~CurlAsyncIOService()
        {
            if (m_handle) 
            {
                // :TODO: Make sure cancels are issued

                // Ensure termination of IO services
                m_ioService.stop(); m_serviceThread.join();

                // Cleanup the libcurl environment 
                auto code = curl_multi_cleanup(m_handle); 
                if (code != CURLM_OK)
                {
                    VOX_LOG(Severity_Error, Error_Unknown, SIO_LOG_CATEGORY, 
                        format("Call to curl_multi_cleanup failed: %1%", curl_multi_strerror(code)));
                }

                m_handle = nullptr;
            }
        }
        
        // ----------------------------------------------------------------------------
        //  IOService loop function
        // ----------------------------------------------------------------------------
        void ioServiceLoop()
        {
            VOX_LOG_TRACE(SIO_LOG_CATEGORY, "Entering new IO service thread");

            try
            {
                m_ioService.run();
            }
            catch (Error & error)
            {
                VOX_LOG_EXCEPTION(Severity_Error, error);
            }
            catch (...)
            {
                VOX_LOG_ERROR(Error_Unknown, SIO_LOG_CATEGORY, "IO Service thread has failed");
            }

            VOX_LOG_TRACE(SIO_LOG_CATEGORY, "Exiting IO service thread");
        }

        // ----------------------------------------------------------------------------
        //  Adds a new easy handle to the multihandle 
        // ----------------------------------------------------------------------------
        void addRequest(CurlAsyncRequest * request)
        {
            CURL* easyhandle = request->handle();

            CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_OPENSOCKETDATA, this));
            CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_CLOSESOCKETDATA, this));

            CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_OPENSOCKETFUNCTION,  openSocketRedirect));
            CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_CLOSESOCKETFUNCTION, closeSocketRedirect));
            
            m_ioService.post(boost::bind(&CurlAsyncIOService::addRequestHandler, this, request));
        }

        // ----------------------------------------------------------------------------
        //  Cancels an in-progress request, terminating the transfer prematurely
        // ----------------------------------------------------------------------------
        void cancelRequest(CurlAsyncRequest * request)
        {
            m_ioService.post(boost::bind(&CurlAsyncIOService::cancelRequestHandler, this, request));
        }

    private:
        // ----------------------------------------------------------------------------
        //  Checks the queue and extracts a pending request if possible
        // ----------------------------------------------------------------------------
        void addRequestHandler(CurlAsyncRequest * request)
        {
            CURL* easyhandle = request->handle();

            CHECK_CURL(curl_multi_add_handle(m_handle, easyhandle));
        }
        
        // ----------------------------------------------------------------------------
        //  Checks the queue and extracts a pending request if possible
        // ----------------------------------------------------------------------------
        void cancelRequestHandler(CurlAsyncRequest * request)
        {
        }

        // ----------------------------------------------------------------------------
        //  Opens a socket for libcurl (CURLOPT_OPENSOCKETFUNCTION)
        // ----------------------------------------------------------------------------
        curl_socket_t openSocket(curlsocktype purpose, struct curl_sockaddr * address)
        {
            curl_socket_t sockfd = CURL_SOCKET_BAD;

            // Restriction to IPv4 for now
            if (purpose == CURLSOCKTYPE_IPCXN && address->family == AF_INET)
            {
                // Create a new TCP socket
                boost::asio::ip::tcp::socket * tcp_socket = new boost::asio::ip::tcp::socket(m_ioService);
 
                // Open to acquire a native handle for libcurl
                boost::system::error_code ec; // Don't let an exception through
                if (tcp_socket->open(boost::asio::ip::tcp::v4(), ec))
                {
                    VOX_LOG_ERROR(Error_Unknown, SIO_LOG_CATEGORY, format("Failed to open socket: %1%", ec.message()));
                }
                else
                {
                    sockfd = tcp_socket->native_handle();
 
                    m_sockets.insert(std::pair<curl_socket_t, boost::asio::ip::tcp::socket *>(sockfd, tcp_socket));
                }
                
                VOX_LOG_TRACE(SIO_LOG_CATEGORY, format("Opening socket: socket=%1%", sockfd));
            }
            else
            {
                VOX_LOG_ERROR(Error_Range, SIO_LOG_CATEGORY, "Failed to open socket: IPv6 not supported");
            }

            return sockfd;
        }

        // ----------------------------------------------------------------------------
        //  Closes a previously opened socket for libcurl (CURLOPT_CLOSESOCKETFUNCTION)
        // ----------------------------------------------------------------------------
        int closeSocket(curl_socket_t item)
        {
            VOX_LOG_TRACE(SIO_LOG_CATEGORY, format("Closing socket: %1%", item));

            auto iter = m_sockets.find(item);
            if (iter != m_sockets.end())
            {
                delete iter->second;
                m_sockets.erase(iter);
            }
            else
            {
                VOX_LOG_ERROR(Error_Bug, SIO_LOG_CATEGORY, format("libcurl requested closure of unrecognized socket: %1%", item));
            }

            return 0;
        }

        // ----------------------------------------------------------------------------
        // Check for completed transfers, and remove their easy handles
        // ----------------------------------------------------------------------------
        void checkInfo()
        {
            VOX_LOG_TRACE(SIO_LOG_CATEGORY, format("Requests remaining: %1%", m_running));

            int messages; ///< Message count

            while (auto * msg = curl_multi_info_read(m_handle, &messages))
            {
                if (msg->msg == CURLMSG_DONE)
                {
                    auto easyhandle = msg->easy_handle;
                    auto result     = msg->data.result;

                    if (curl_multi_remove_handle(m_handle, easyhandle) != CURLM_OK)
                    {
                        VOX_LOG_ERROR(Error_Unknown, SIO_LOG_CATEGORY, "Error calling curl_multi_remove_handle");
                    }

                    CurlAsyncRequest * request;
                    curl_easy_getinfo(easyhandle, CURLINFO_PRIVATE, &request);

                    char * effectiveUrl;
                    curl_easy_getinfo(easyhandle, CURLINFO_EFFECTIVE_URL, &effectiveUrl);
                    // :TODO: Feed this back to the user somehow

                    request->complete(nullptr);
                }
            }
        }

        // ----------------------------------------------------------------------------
        //  Called by boost::asio timer when the libcurl specified timeout occurs
        // ----------------------------------------------------------------------------
        void timerCallback(boost::system::error_code const& error)
        {
            VOX_LOG_TRACE(SIO_LOG_CATEGORY, "Timeout: Calling curl_multi_socket_action");

            if (!error)
            {
                auto code = curl_multi_socket_action(m_handle, CURL_SOCKET_TIMEOUT, 0, &m_running);
                if (code != CURLM_OK)
                {
                    VOX_LOG(Severity_Error, Error_Unknown, SIO_LOG_CATEGORY, 
                        format("Call to curl_multi_socket_action failed: %1%", curl_multi_strerror(code)));
                }

                checkInfo();
            }
            else
            {
                VOX_LOG(Severity_Error, Error_Unknown, SIO_LOG_CATEGORY, 
                    format("Error performing asio async_wait: %1%", error));
            }
        }

        // ----------------------------------------------------------------------------
        //  Updates the event timer with the timeout issued by libcurl
        // ----------------------------------------------------------------------------
        int multiTimerCallback(long timeoutMs)
        {
            VOX_LOG_TRACE(SIO_LOG_CATEGORY, format("MultiTimer Callback: %1% ms", timeoutMs));

            m_timer.cancel();

            if (timeoutMs > 0)
            {
                // Reset the timer to the specified timeout 
                m_timer.expires_from_now(boost::posix_time::milliseconds(timeoutMs));
                m_timer.async_wait(boost::bind(&CurlAsyncIOService::timerCallback, this, _1));
            }
            else
            {
                // Call timeout function immediately 
                timerCallback(boost::system::error_code());
            }

            return 0;
        }
        
        // ----------------------------------------------------------------------------
        //  Event callback from asio on socket event completion
        // ----------------------------------------------------------------------------
        void eventCallback(curl_socket_t fd, int action)
        {
            VOX_LOG_TRACE(SIO_LOG_CATEGORY, format("Socket ready: socket=%1%", fd));

            auto code = curl_multi_socket_action(m_handle, fd, action, &m_running);
            if (code != CURLM_OK)
            {
                VOX_LOG(Severity_Error, Error_Unknown, SIO_LOG_CATEGORY, 
                    format("Call to curl_multi_socket_action failed: %1%", curl_multi_strerror(code)));
            }

            checkInfo(); ///< Process request completions
 
            if (m_running <= 0) m_timer.cancel();
        }

        // ----------------------------------------------------------------------------
        //  Waits for the specified option on the socket specified by libcurl
        // ----------------------------------------------------------------------------
        void socketCallback(curl_socket_t s, int action, boost::asio::ip::tcp::socket * socketHandle)
        {
            boost::asio::ip::tcp::socket * tcpSocket = socketHandle;

            // Acquire the handle to the overhead asio tcp socket
            if (tcpSocket == nullptr)
            {
                auto iter = m_sockets.find(s);
                if (iter == m_sockets.end())
                {
                    VOX_LOG_WARNING(SIO_LOG_CATEGORY, "Socket callback on presumed c-ares socket; ignoring.");
             
                    return;
                }
                tcpSocket = iter->second;

                curl_multi_assign(m_handle, s, tcpSocket); // Cache a private handle to the socket
            }

            // Wait on the socket for the specified action/event/status
            switch (action)
            {
            case CURL_POLL_IN:
                tcpSocket->async_read_some(boost::asio::null_buffers(), boost::bind(&CurlAsyncIOService::eventCallback, this, s, action));
                break;
                
            case CURL_POLL_OUT:
                tcpSocket->async_write_some(boost::asio::null_buffers(), boost::bind(&CurlAsyncIOService::eventCallback, this, s, action));
                break;
                
            case CURL_POLL_INOUT:
                tcpSocket->async_read_some(boost::asio::null_buffers(), boost::bind(&CurlAsyncIOService::eventCallback, this, s, action));
                tcpSocket->async_write_some(boost::asio::null_buffers(), boost::bind(&CurlAsyncIOService::eventCallback, this, s, action));
                break;

            default:
                VOX_LOG(Severity_Error, Error_Bug, SIO_LOG_CATEGORY, 
                    format("Unrecognized action requested on libcurl socket: %1%", action));
                break;
            }
        }

    private: // Libcurl Callback Redirects 

        // ----------------------------------------------------------------------------
        //  Updates the even timer after a curl_multi_* call
        // ----------------------------------------------------------------------------
        static int multiTimerCallbackRedirect(CURLM *multihandle, long timeoutMs, void * userp)
        {
            return reinterpret_cast<CurlAsyncIOService*>(userp)->multiTimerCallback(timeoutMs);
        }

        // ----------------------------------------------------------------------------
        //  Closes a previously opened socket for libcurl (CURLOPT_CLOSESOCKETFUNCTION)
        // ----------------------------------------------------------------------------
        static int closeSocketRedirect(void * clientp, curl_socket_t item)
        {
            return reinterpret_cast<CurlAsyncIOService*>(clientp)->closeSocket(item);
        }
        
        // ----------------------------------------------------------------------------
        //  Opens a socket for libcurl (CURLOPT_OPENSOCKETFUNCTION)
        // ----------------------------------------------------------------------------
        static curl_socket_t openSocketRedirect(void * clientp, curlsocktype purpose, struct curl_sockaddr * address)
        {
            return reinterpret_cast<CurlAsyncIOService*>(clientp)->openSocket(purpose, address);
        }

        // ----------------------------------------------------------------------------
        //  Socket callback from libcurl about socket status change (CURLMOPT_SOCKETFUNCTION)
        // ----------------------------------------------------------------------------
        static int socketCallbackRedirect(CURL * easyhandle, curl_socket_t s, int action, void *userp, void *sockp)
        {
            static const char * actionStr[] = {"none", "IN", "OUT", "INOUT", "REMOVE"};

            VOX_LOG_TRACE(SIO_LOG_CATEGORY, format("Socket callback: socket=%1%, action=%2%", s, actionStr[action]));

            if (action == CURL_POLL_REMOVE) return 0;

            reinterpret_cast<CurlAsyncIOService*>(userp)->socketCallback(s, action, reinterpret_cast<boost::asio::ip::tcp::socket*>(sockp));

            return 0;
        }

    private: // DON'T move these around, the some variables have construction dependencies

        std::map<curl_socket_t, boost::asio::ip::tcp::socket *> m_sockets;

        boost::asio::io_service         m_ioService;    ///< Asio service which services libcurl's async wait requests
        boost::asio::deadline_timer     m_timer;        ///< Event timer from asio services
        boost::asio::io_service::work   m_work;         ///< Keep-alive work for the service 

        boost::thread m_serviceThread;  ///< Service thread for requests
        boost::mutex  m_mutex;

        CURLM * m_handle;     ///< Curl multi handle
        int     m_running;    ///< Active requests
    };

    std::unique_ptr<CurlAsyncIOService> asyncIOService; ///< This will be a service pool at some point 
    
    void initThreadPool() // :DEBUG:
    {
        static boost::mutex mutex;

        if (!asyncIOService)
        {
            boost::mutex::scoped_lock lock(mutex);

            if (!asyncIOService)
            {
                asyncIOService.reset(new CurlAsyncIOService());            
            }
        }
    }

} // namespace filescope
} // namespace anonymous

// ----------------------------------------------------------------------------
//  Initializes a libcurl easy handle for use by the AsyncIO service
// ----------------------------------------------------------------------------
CurlAsyncRequest::CurlAsyncRequest(
    ResourceId &     identifier, ///< The resource identifier
    OptionSet const& options     ///< The advanced access options
    )
{
    curl_global_init(CURL_GLOBAL_ALL);
    filescope::initThreadPool();

    VOX_LOG_TRACE(SIO_LOG_CATEGORY, format("Initiating CurlAsyncRequest: url=%1% id=%2%", identifier.asString(), (void*)this));

    CURL * easyhandle; // Request handle

    // Create the libcurl request handle
    if (!(easyhandle = curl_easy_init()))
    {
        throw Error(__FILE__, __LINE__, SIO_LOG_CATEGORY,
            "Failed to initialize handle for libCurl session");
    }
    m_handle = easyhandle;

    // Set the global request options using the libcurl client API (see http://curl.haxx.se/libcurl/c/curl_easy_setopt.html)
    CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_USERAGENT,       "voxlib-agent/" SIO_VERSION_STRING));
    CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_URL,             identifier.asString().c_str()));
    CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_FOLLOWLOCATION,  1L));
    CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_PRIVATE,         this));
    CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_LOW_SPEED_TIME,  3L));
    CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_LOW_SPEED_LIMIT, 10L));
    curl_easy_setopt(easyhandle, CURLOPT_VERBOSE, 1L);
}

// ----------------------------------------------------------------------------
//  Terminates an async request if necessary
// ----------------------------------------------------------------------------
CurlAsyncRequest::~CurlAsyncRequest()
{
    if (m_handle)
    {
        filescope::asyncIOService->cancelRequest(this);

        // :TODO: Wait on completion
    }
}

// ---------------------------------------------------------------------------- 
//  Cleans up the libcurl request handle
// ----------------------------------------------------------------------------
void CurlAsyncRequest::complete(std::exception_ptr ex)
{
    curl_easy_cleanup(reinterpret_cast<CURL*>(m_handle));
}

// ----------------------------------------------------------------------------
//  Initializes a request, creating and registering the curl handle
// ----------------------------------------------------------------------------
CurlOStreamBuf::CurlOStreamBuf(
    ResourceId &     identifier,
    OptionSet const& options
    ) :
    CurlAsyncRequest(identifier, options),
    m_error(nullptr)
{
    // Initialize the buf ptrs to ensure immediate underflow/overflow 
    setg(nullptr, nullptr, nullptr); 
    setp(nullptr, nullptr, nullptr);
    
    CURL * easyhandle = reinterpret_cast<CURL*>(m_handle);

    // Set the request options using the libcurl client API (see http://curl.haxx.se/libcurl/c/curl_easy_setopt.html)
    CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_UPLOAD,         1L));
    CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_READDATA,       this));
    CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_READFUNCTION,   &CurlOStreamBuf::onGetDataCallback));

    // The BufSize option is a generic option across all protocols
    auto bufSizeIter = options.find("BufSize");
    if (bufSizeIter != options.end())
    {
        auto size = boost::lexical_cast<size_t>(bufSizeIter->second);
        m_buffer.resize(size);
    }
    else m_buffer.resize(512);

    // The Size option is a generic option across all protocols
    auto sizeIter = options.find("Size");
    if (sizeIter != options.end())
    {
        auto size = boost::lexical_cast<size_t>(sizeIter->second);
        CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_INFILESIZE_LARGE, size));
    }

    // Issue the request to the Multi-Interface wrapper
    filescope::asyncIOService->addRequest(this);
}
    
// ----------------------------------------------------------------------------
//  Recieves notification of request completion from the io service thread
//  This includes completion by failure
// ----------------------------------------------------------------------------
void CurlOStreamBuf::complete(std::exception_ptr ex)
{
    VOX_LOG_TRACE(SIO_LOG_CATEGORY, format("Completing request: id=%1%", (void*)this));

    CurlAsyncRequest::complete(ex);
}

// ----------------------------------------------------------------------------
//  Copies data from a libcurl request into the internal memory buffer
// ----------------------------------------------------------------------------
size_t CurlOStreamBuf::onGetData(void * ptr, size_t maxBytes)
{
    VOX_LOG_TRACE(SIO_LOG_CATEGORY, format("Awaiting upload of request content: id=%1%", (void*)this));

    // :TODO: PAUSE_IF_NECESSARY for data upload waiting, prevent hang on service thread

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
    )  :
    CurlAsyncRequest(identifier, options),
    m_error(nullptr)
{
    // Initialize the buf ptrs to ensure immediate underflow/overflow 
    setg(nullptr, nullptr, nullptr); 
    setp(nullptr, nullptr, nullptr);

    CURL * easyhandle = reinterpret_cast<CURL*>(m_handle);

    // Set the request options using the libcurl client API (see http://curl.haxx.se/libcurl/c/curl_easy_setopt.html)
    CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_WRITEDATA,        this));
    CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_WRITEFUNCTION,    &CurlIStreamBuf::onDataReadyCallback));

    // Issue the request to the Multi-Interface wrapper
    filescope::asyncIOService->addRequest(this);
    //curl_easy_perform(easyhandle);
    //complete(nullptr);
    //filescope::asyncIOService.reset();
}

// ----------------------------------------------------------------------------
//  Underflow method which awaits a chunk and resets the read head
// ----------------------------------------------------------------------------
int CurlIStreamBuf::underflow()
{
    boost::mutex::scoped_lock lock(m_mutex);

    // Pop the old buffer from the list ...
    if (gptr() != nullptr) 
    {
        // ... but leave terminal chunk for EOF detection
        if (m_data.front().size != 0)
        {
            m_data.front().free();
            m_data.pop_front();
        }
    }

    while (m_data.empty()) m_cond.wait(lock);

    if (!(m_error == nullptr)) std::rethrow_exception(m_error); 

    if (m_data.front().size == 0) return EOF; /// nullbuf indicates eof

    // Reset the get pointers to the next buffer chunk
    char* data = reinterpret_cast<char*>(m_data.front().data);
    size_t len = m_data.front().size;
    setg(data, data, data+len);

    return *data;
}

// ----------------------------------------------------------------------------
//  Recieves notification of request completion from the io service thread
//  This includes completion by failure
// ----------------------------------------------------------------------------
void CurlIStreamBuf::complete(std::exception_ptr ex)
{
    VOX_LOG_TRACE(SIO_LOG_CATEGORY, format("Completing request: id=%1%", (void*)this));

    CurlAsyncRequest::complete(ex);

    boost::mutex::scoped_lock lock(m_mutex);

    m_error = ex; // Copy potential exception

    m_data.push_back( DataBuffer(nullptr, 0) );

    m_cond.notify_all();
}

// ----------------------------------------------------------------------------
//  Copies data from a libcurl request into the internal memory buffer
// ----------------------------------------------------------------------------
size_t CurlIStreamBuf::onDataReady(char * ptr, size_t bytes)
{
    VOX_LOG_TRACE(SIO_LOG_CATEGORY, format("Receiving request content: id=%1%", (void*)this));

    boost::mutex::scoped_lock lock(m_mutex);

    m_data.push_back( DataBuffer(ptr, bytes) );

    m_cond.notify_all();

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