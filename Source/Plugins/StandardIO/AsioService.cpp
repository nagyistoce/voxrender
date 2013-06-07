/* ===========================================================================

    Project: AsioService

    Description: A foundation Async IO service for developing WebClient APIs

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
#include "AsioService.h"
#include "VoxLib/Core/Logging.h"

// 3rd Party Dependencies
#include <boost/asio.hpp>
#include <curl/curl.h>

#define CHECK_CURL(x) if ((x) != CURLE_OK) { throw Error(__FILE__, __LINE__, SIO_LOG_CATEGORY, "call to libcurl api has returned error"); }

namespace vox {

namespace {
namespace filescope {

    // ----------------------------------------------------------------------------
    //  IO service class which encapsulates the curl multi interface with asio
    // ----------------------------------------------------------------------------
    class AsioService
    {
    public:
        // ----------------------------------------------------------------------------
        //  Creates a multi handle and initializes the io service and socket callbacks 
        // ----------------------------------------------------------------------------
        AsioService() : 
            m_running(0), 
            m_timer(m_ioService), 
            m_work(m_ioService), 
            m_serviceThread(boost::bind(&AsioService::ioServiceLoop, this))
        {
            CHECK_CURL(curl_global_init(CURL_GLOBAL_ALL));

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
        ~AsioService()
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

            curl_global_cleanup();
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
        void addRequest(AsioRequest * request)
        {
            CURL* easyhandle = request->handle();

            CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_OPENSOCKETDATA, this));
            CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_CLOSESOCKETDATA, this));

            CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_OPENSOCKETFUNCTION,  openSocketRedirect));
            CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_CLOSESOCKETFUNCTION, closeSocketRedirect));
            
            m_ioService.post(boost::bind(&AsioService::addRequestHandler, this, request));
        }

        // ----------------------------------------------------------------------------
        //  Cancels an in-progress request, terminating the transfer prematurely
        // ----------------------------------------------------------------------------
        void cancelRequest(std::shared_ptr<AsioRequest> request)
        {
            m_ioService.post(boost::bind(&AsioService::cancelRequestHandler, this, request));
        }

    private:
        // ----------------------------------------------------------------------------
        //  Checks the queue and extracts a pending request if possible
        // ----------------------------------------------------------------------------
        void addRequestHandler(AsioRequest * request)
        {
            CURL* easyhandle = request->handle();

            CHECK_CURL(curl_multi_add_handle(m_handle, easyhandle));
        }
        
        // ----------------------------------------------------------------------------
        //  Checks the queue and extracts a pending request if possible
        // ----------------------------------------------------------------------------
        void cancelRequestHandler(std::shared_ptr<AsioRequest> request)
        {
            request->onComplete(nullptr);
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
                boost::system::error_code ec; 
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

                    AsioRequest * request;
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
                m_timer.async_wait(boost::bind(&AsioService::timerCallback, this, _1));
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
        void eventCallback(curl_socket_t fd, int action, int * actionHandle)
        {
            VOX_LOG_TRACE(SIO_LOG_CATEGORY, format("Socket ready: socket=%1%", fd));

            // Issue another call to curl multi now that we can perform on the socket
            auto code = curl_multi_socket_action(m_handle, fd, action, &m_running);
            if (code != CURLM_OK)
            {
                VOX_LOG(Severity_Error, Error_Unknown, SIO_LOG_CATEGORY, 
                    format("Call to curl_multi_socket_action failed: %1%", curl_multi_strerror(code)));
            }

            // Check if we need to resume the previous action (ie action handle is not changed)
            if (action == *actionHandle)
            {
                VOX_LOG_TRACE(SIO_LOG_CATEGORY, format("No change in socket action, resuming wait: socket=%1% action=%2%", fd, action));

                socketCallback(fd, action, actionHandle);
            }

            checkInfo(); ///< Process request completions
 
            if (m_running <= 0) m_timer.cancel();
        }

        // ----------------------------------------------------------------------------
        //  Waits for the specified option on the socket specified by libcurl
        // ----------------------------------------------------------------------------
        void socketCallback(curl_socket_t s, int action, int * previousAction)
        {
            // Acquire the handle to the overhead asio tcp socket
            auto iter = m_sockets.find(s);
            if (iter == m_sockets.end())
            {
                VOX_LOG_WARNING(SIO_LOG_CATEGORY, "Socket callback on presumed c-ares socket; ignoring.");
             
                return;
            }
            auto tcpSocket = iter->second;

            // Store the action handle for reference in eventCallback
            int * actionHandle = previousAction;
            if (actionHandle == nullptr)
            {
                actionHandle = new int(action);

                curl_multi_assign(m_handle, s, actionHandle);
            }
            else
            {
                *actionHandle = action;
            }

            // Wait on the socket for the specified action/event/status
            switch (action)
            {
            case CURL_POLL_IN:
                tcpSocket->async_read_some(boost::asio::null_buffers(), boost::bind(&AsioService::eventCallback, this, s, action, actionHandle));
                break;
                
            case CURL_POLL_OUT:
                tcpSocket->async_write_some(boost::asio::null_buffers(), boost::bind(&AsioService::eventCallback, this, s, action, actionHandle));
                break;
                
            case CURL_POLL_INOUT:
                tcpSocket->async_read_some(boost::asio::null_buffers(), boost::bind(&AsioService::eventCallback, this, s, action, actionHandle));
                tcpSocket->async_write_some(boost::asio::null_buffers(), boost::bind(&AsioService::eventCallback, this, s, action, actionHandle));
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
            return reinterpret_cast<AsioService*>(userp)->multiTimerCallback(timeoutMs);
        }

        // ----------------------------------------------------------------------------
        //  Closes a previously opened socket for libcurl (CURLOPT_CLOSESOCKETFUNCTION)
        // ----------------------------------------------------------------------------
        static int closeSocketRedirect(void * clientp, curl_socket_t item)
        {
            return reinterpret_cast<AsioService*>(clientp)->closeSocket(item);
        }
        
        // ----------------------------------------------------------------------------
        //  Opens a socket for libcurl (CURLOPT_OPENSOCKETFUNCTION)
        // ----------------------------------------------------------------------------
        static curl_socket_t openSocketRedirect(void * clientp, curlsocktype purpose, struct curl_sockaddr * address)
        {
            return reinterpret_cast<AsioService*>(clientp)->openSocket(purpose, address);
        }

        // ----------------------------------------------------------------------------
        //  Socket callback from libcurl about socket status change (CURLMOPT_SOCKETFUNCTION)
        // ----------------------------------------------------------------------------
        static int socketCallbackRedirect(CURL * easyhandle, curl_socket_t s, int action, void *userp, void *sockp)
        {
            static const char * actionStr[] = {"none", "IN", "OUT", "INOUT", "REMOVE"};

            VOX_LOG_TRACE(SIO_LOG_CATEGORY, format("Socket callback: socket=%1%, action=%2% previous_action=%3%", 
                s, actionStr[action], actionStr[(sockp ? *(int*)sockp : 0)]));

            if (action == CURL_POLL_REMOVE) { delete sockp; return 0; }

            reinterpret_cast<AsioService*>(userp)->socketCallback(s, action, reinterpret_cast<int*>(sockp));

            return 0;
        }

    private: // DON'T move these around, the some variables have construction dependencies

        std::map<curl_socket_t, boost::asio::ip::tcp::socket *> m_sockets;

        boost::asio::io_service       m_ioService;    ///< Asio service which services libcurl's async wait requests
        boost::asio::deadline_timer   m_timer;        ///< Event timer from asio services
        boost::asio::io_service::work m_work;         ///< Keep-alive work for the service 

        boost::thread m_serviceThread;  ///< Service thread for requests
        boost::mutex  m_mutex;

        CURLM * m_handle;     ///< Curl multi handle
        int     m_running;    ///< Active requests
    };

    std::unique_ptr<AsioService> asyncIOService; ///< This will be a service pool at some point 
    
    // ----------------------------------------------------------------------------
    //  Ensures initialization of the asio service pool
    // ----------------------------------------------------------------------------
    void initServicePool() 
    {
        static boost::mutex mutex;

        if (!asyncIOService)
        {
            boost::mutex::scoped_lock lock(mutex);

            if (!asyncIOService)
            {
                asyncIOService.reset(new AsioService());            
            }
        }
    }

} // namespace filescope
} // namespace anonymous

// ----------------------------------------------------------------------------
//  Initializes a libcurl easy handle for use by the asio service
// ----------------------------------------------------------------------------
AsioRequest::AsioRequest(
    ResourceId &     identifier, ///< The resource identifier
    OptionSet const& options     ///< The advanced access options
    )
{
    filescope::initServicePool();

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
    CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_VERBOSE, 1L));
}

// ----------------------------------------------------------------------------
//  Terminates an async request if necessary
// ----------------------------------------------------------------------------
AsioRequest::~AsioRequest()
{
    // Check if request was pending but never issued
    if (m_handle)
    {
        curl_easy_cleanup(reinterpret_cast<CURL*>(m_handle));
    }
}

// ---------------------------------------------------------------------------- 
//  Indicates the asio holder wishes to initiate cancellation of the request
// ----------------------------------------------------------------------------
void AsioRequest::detach()
{
    boost::mutex::scoped_lock lock(m_mutex);

    if (m_handle)
    {
        m_self = shared_from_this();

        filescope::asyncIOService->cancelRequest(m_self);
    }
}

// ---------------------------------------------------------------------------- 
//  Cleans up the libcurl request handle
// ----------------------------------------------------------------------------
void AsioRequest::onComplete(std::exception_ptr ex)
{
    boost::mutex::scoped_lock lock(m_mutex);

    if (m_handle)
    {
        complete(ex);

        curl_easy_cleanup(reinterpret_cast<CURL*>(m_handle));

        m_handle = nullptr;

        lock.unlock();

        m_self.reset(); // OK to free ourselves
    }
}

// ---------------------------------------------------------------------------- 
//  Issues the request to the asio pool to be serviced
// ----------------------------------------------------------------------------
void AsioRequest::issueRequest()
{
    filescope::asyncIOService->addRequest(this);
}

} // namespace vox