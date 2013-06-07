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

#define CHECK_CURL(x) if ((x) != CURLE_OK) { throw Error(__FILE__, __LINE__, SIO_LOG_CATEGORY, "call to libcurl api has returned error"); }

namespace vox {
    
// ----------------------------------------------------------------------------
//  Impl structure for output stream buffer
// ----------------------------------------------------------------------------
class CurlOStreamBuf::Impl : public AsioRequest
{
public:
    /** Creates a new asio request handle */
    static std::shared_ptr<CurlOStreamBuf::Impl> makeImpl(ResourceId & identifier, OptionSet const& options) 
    {
        return std::shared_ptr<CurlOStreamBuf::Impl>(new CurlOStreamBuf::Impl(identifier, options)); 
    }

    std::exception_ptr m_error;  ///< Internal exception buffer
    std::vector<UInt8> m_buffer; ///< Output data buffer

    boost::condition_variable m_cond;   ///< Data buffer empty condition
    
    /** Provides callback notification of request completion */
    virtual void complete(std::exception_ptr ex)
    {
        VOX_LOG_TRACE(SIO_LOG_CATEGORY, format("Completing request: id=%1%", (void*)this));
    }

    /** Returns the internal mutex */
    boost::mutex & mutex() { return m_mutex; }

private:
    Impl(ResourceId & identifier, OptionSet const& options) : m_error(nullptr), AsioRequest(identifier, options) 
    {
        CURL * easyhandle = reinterpret_cast<CURL*>(m_handle);

        // Set the request options using the libcurl client API (see http://curl.haxx.se/libcurl/c/curl_easy_setopt.html)
        CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_UPLOAD,         1L));
        CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_READDATA,       this));
        CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_READFUNCTION,   &CurlOStreamBuf::Impl::onGetDataCallback));

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
    } 

    /** Provides upload data to libcurl */
    size_t onGetData(void * ptr, size_t maxBytes)
    {
        return 0;
    }

    /** Callback function for the libcurl library which redirects to onData */
    static size_t onGetDataCallback(void * ptr, size_t size, size_t nmemb, void * buf)
    {
        return reinterpret_cast<CurlOStreamBuf::Impl*>(buf)->onGetData(ptr, size*nmemb);
    }
};

// ----------------------------------------------------------------------------
//  Impl structure for input stream buffer
// ----------------------------------------------------------------------------
class CurlIStreamBuf::Impl : public AsioRequest
{
public:
    class DataBuffer
    {
    public:
        DataBuffer(char * inData, size_t inSize) :
          data(new UInt8[inSize]),
          size(inSize)
        {
            memcpy(data, inData, inSize);
        }

        void free() { delete[] data; } 

        UInt8 * data;
        size_t  size;
    };

public:
    /** Creates a new asio request handle */
    static std::shared_ptr<CurlIStreamBuf::Impl> makeImpl(ResourceId & identifier, OptionSet const& options) 
    {
        return std::shared_ptr<CurlIStreamBuf::Impl>(new CurlIStreamBuf::Impl(identifier, options)); 
    }

    std::exception_ptr    m_error;     ///< Internal exception buffer
    std::list<DataBuffer> m_data;      ///< Internal data buffers

    boost::condition_variable m_cond;   ///< Data buffer empty condition
    
    /** Returns the internal mutex */
    boost::mutex & mutex() { return m_mutex; }

    /** Provides callback notification of request completion */
    virtual void complete(std::exception_ptr ex)
    {
        VOX_LOG_TRACE(SIO_LOG_CATEGORY, format("Completing request: id=%1%", (void*)this));

        m_error = ex; // Copy potential exception

        m_data.push_back( DataBuffer(nullptr, 0) );

        m_cond.notify_all();
    }

private:
    Impl(ResourceId & identifier, OptionSet const& options) : m_error(nullptr), AsioRequest(identifier, options) 
    {
        CURL * easyhandle = reinterpret_cast<CURL*>(m_handle);

        // Set the request options using the libcurl client API (see http://curl.haxx.se/libcurl/c/curl_easy_setopt.html)
        CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_WRITEDATA,     this));
        CHECK_CURL(curl_easy_setopt(easyhandle, CURLOPT_WRITEFUNCTION, &CurlIStreamBuf::Impl::onDataReadyCallback));
    } 

    /** Provides upload data to libcurl */
    size_t onDataReady(char * ptr, size_t bytes)
    {
        VOX_LOG_TRACE(SIO_LOG_CATEGORY, format("Receiving request content: id=%1%", (void*)this));

        boost::mutex::scoped_lock lock(m_mutex);

        m_data.push_back( DataBuffer(ptr, bytes) );

        m_cond.notify_all();

        return bytes;
    }

    /** Callback function for the libcurl library which redirects to onData */
    static size_t onDataReadyCallback(char * ptr, size_t size, size_t nmemb, void * buf)
    {
        return reinterpret_cast<CurlIStreamBuf::Impl*>(buf)->onDataReady(ptr, size*nmemb);
    }
};

// ----------------------------------------------------------------------------
//  Initializes a request, creating and registering the curl handle
// ----------------------------------------------------------------------------
CurlOStreamBuf::CurlOStreamBuf(
    ResourceId &     identifier,
    OptionSet const& options
    )
{
    // Initialize the buf ptrs to ensure immediate underflow/overflow 
    setg(nullptr, nullptr, nullptr); 
    setp(nullptr, nullptr, nullptr);

    m_pImpl = Impl::makeImpl(identifier, options);

    m_pImpl->issueRequest(); // Begin the request
}

// ----------------------------------------------------------------------------
//  Detaches the request for asynchronous completion
// ----------------------------------------------------------------------------
CurlOStreamBuf::~CurlOStreamBuf()
{
    m_pImpl->detach(); 
}

// ----------------------------------------------------------------------------
//  Overflow method which flushes the data buffer 
// ----------------------------------------------------------------------------
 int CurlOStreamBuf::overflow()
 {
     return EOF;
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

    m_pImpl = Impl::makeImpl(identifier, options);

    m_pImpl->issueRequest(); // Begin the request
}

// ----------------------------------------------------------------------------
//  Detaches the request for asynchronous completion
// ----------------------------------------------------------------------------
CurlIStreamBuf::~CurlIStreamBuf()
{
    m_pImpl->detach(); 
}

// ----------------------------------------------------------------------------
//  Underflow method which awaits a chunk and resets the read head
// ----------------------------------------------------------------------------
int CurlIStreamBuf::underflow()
{
    boost::mutex::scoped_lock lock(m_pImpl->mutex());

    // Pop the old buffer from the list ...
    if (gptr() != nullptr) 
    {
        // ... but leave terminal chunk for EOF detection
        if (m_pImpl->m_data.front().size != 0)
        {
            m_pImpl->m_data.front().free();
            m_pImpl->m_data.pop_front();
        }
    }

    while (m_pImpl->m_data.empty()) m_pImpl->m_cond.wait(lock);

    if (!(m_pImpl->m_error == nullptr)) std::rethrow_exception(m_pImpl->m_error); 

    if (m_pImpl->m_data.front().size == 0) return EOF; /// nullbuf indicates eof

    // Reset the get pointers to the next buffer chunk
    char* data = reinterpret_cast<char*>(m_pImpl->m_data.front().data);
    size_t len = m_pImpl->m_data.front().size;
    setg(data, data, data+len);

    return *data;
}

} // namespace vox