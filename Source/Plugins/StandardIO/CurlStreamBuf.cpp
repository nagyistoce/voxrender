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

namespace vox {

namespace {
namespace filescope {

    // Static member initialization
    static CURLM *      multihandle = nullptr;
    static boost::mutex multiMutex; 
    static int          multiRunning = 0;

    // ----------------------------------------------------------------------------
    //  Continues processing of the multi-handle 
    // ----------------------------------------------------------------------------
    void performMultiWork()
    {
        // Acquire a read-lock on the modules for thread safety support
        boost::unique_lock<decltype(filescope::multiMutex)> lock(filescope::multiMutex);

        int running = multiRunning;
        curl_multi_perform(multihandle, &multiRunning);
    }


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

    // Set the request options using the libcurl client API
    curl_easy_setopt(m_easyhandle, CURLOPT_USERAGENT, "voxlib-agent/" SIO_VERSION_STRING);

    curl_easy_setopt(m_easyhandle, CURLOPT_URL, identifier.asString()); // Request URL
    curl_easy_setopt(m_easyhandle, CURLOPT_FOLLOWLOCATION, 1L);         // Follow redirects

    curl_easy_setopt(m_easyhandle, CURLOPT_UPLOAD, 1L); // Specify intent to upload

    curl_easy_setopt(m_easyhandle, CURLOPT_READDATA, this);
    curl_easy_setopt(m_easyhandle, CURLOPT_READFUNCTION, &CurlOStreamBuf::onGetDataCallback);

    // The FileSize option is a generic option across all protocols
    auto sizeIter = options.find("Size");
    if (sizeIter != options.end())
    {
        auto size = boost::lexical_cast<size_t>(sizeIter->second);
        curl_easy_setopt(m_easyhandle, CURLOPT_INFILESIZE_LARGE, size);
    }

    // Acquire a read-lock on the modules for thread safety support
    boost::mutex::scoped_lock lock(filescope::multiMutex);

      // Register the request with the libcurl multihandle to initiate
      if (filescope::multihandle == nullptr) 
      {
          filescope::multihandle = curl_multi_init();
      }

      curl_multi_add_handle(filescope::multihandle, m_easyhandle);

      int running = filescope::multiRunning;
      curl_multi_perform(filescope::multihandle, &filescope::multiRunning);
}

// ----------------------------------------------------------------------------
//  Terminates the session if it is still active
// ----------------------------------------------------------------------------
void CurlOStreamBuf::cleanup()
{
    if (m_easyhandle)
    {
        // Acquire a read-lock on the modules for thread safety support
        boost::mutex::scoped_lock lock(filescope::multiMutex);

        curl_multi_remove_handle(filescope::multihandle, m_easyhandle);

        curl_easy_cleanup(m_easyhandle);

        m_easyhandle = nullptr;
    }
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
    setg(nullptr, nullptr, nullptr);

    // Create the libcurl request handle
    if (!(m_easyhandle = curl_easy_init()))
    {
        throw PluginError(__FILE__, __LINE__, SIO_LOG_CATEGORY,
            "Failed to initialize handle for libCurl session");
    }

    // Set the request options using the libcurl client API
    curl_easy_setopt(m_easyhandle, CURLOPT_USERAGENT, "voxlib-agent/" SIO_VERSION_STRING);

    curl_easy_setopt(m_easyhandle, CURLOPT_URL, identifier.asString().c_str()); // Request URL
    curl_easy_setopt(m_easyhandle, CURLOPT_FOLLOWLOCATION, 1L);         // Follow redirects

    curl_easy_setopt(m_easyhandle, CURLOPT_WRITEDATA, this);
    curl_easy_setopt(m_easyhandle, CURLOPT_WRITEFUNCTION, &CurlIStreamBuf::onDataReadyCallback);

    // Acquire a read-lock on the modules for thread safety support
    boost::mutex::scoped_lock lock(filescope::multiMutex);

      // Register the request with the libcurl multihandle to initiate
      if (filescope::multihandle == nullptr) 
      {
          filescope::multihandle = curl_multi_init();
      }

      curl_multi_add_handle(filescope::multihandle, m_easyhandle);

      int running = filescope::multiRunning; VOX_LOGF(Severity_Info, Error_None, SIO_LOG_CATEGORY, format("%1% %2%", filescope::multihandle, m_easyhandle));
      curl_multi_perform(filescope::multihandle, &filescope::multiRunning);
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
        // :TODO: Poll multi_perform MUST DO

        m_cond.wait(lock);
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
//  Terminates the session if it is still active
// ----------------------------------------------------------------------------
void CurlIStreamBuf::cleanup()
{
    if (m_easyhandle)
    {
        // Acquire a read-lock on the modules for thread safety support
        boost::mutex::scoped_lock lock(filescope::multiMutex);

        curl_multi_remove_handle(filescope::multihandle, m_easyhandle);

        curl_easy_cleanup(m_easyhandle);

        m_easyhandle = nullptr;
    }
}

// ----------------------------------------------------------------------------
//  Cleans up the multihandle if it is still active
// ----------------------------------------------------------------------------
void CurlIStreamBuf::onUnload()
{
    // Acquire a read-lock on the modules for thread safety support
    boost::mutex::scoped_lock lock(filescope::multiMutex);

      if (filescope::multihandle != nullptr) 
      {
          curl_multi_cleanup(filescope::multihandle);

          filescope::multihandle = nullptr;
      }
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