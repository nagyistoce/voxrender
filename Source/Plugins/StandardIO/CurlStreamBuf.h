/* ===========================================================================

    Project: StandardIO - Standard IO protocols for VoxIO

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

// Begin definition
#ifndef SIO_CURL_STREAMBUF_H
#define SIO_CURL_STREAMBUF_H

// Include Dependencies
#include "StandardIO/Common.h"
#include "VoxLib/Core/Functors.h"
#include "VoxLib/IO/Resource.h"
#include "VoxLib/IO/ResourceModule.h"

// LibCurl Library
#include <curl/curl.h>

// 3rd Party Dependencies
#include <boost/thread.hpp>

namespace vox {

/**
 * Output oriented StreamBuf which wraps libcurl requests
 *
 * An implementation of std::streambuf which provides access to the libcurl request for data upload.
 */
class CurlOStreamBuf : public std::streambuf
{
public:
    /** Initializes a new libcurl session */
    CurlOStreamBuf(
        ResourceId &     identifier, ///< The resource identifier
        OptionSet const& options     ///< The advanced access options
        );
    
    /** Terminates the libcurl session */
    ~CurlOStreamBuf() { cleanup(); }

    /** */

private:
    CURL * m_easyhandle; ///< Session easy handle

    std::vector<UInt8> m_buffer; ///< Output data buffer

    /** Provides upload data to libcurl */
    size_t onGetData(void * ptr, size_t maxBytes);

    /** Terminates the session and closes active handles */
    void cleanup();

    /** Callback function for the libcurl library which redirects to onData */
    static size_t onGetDataCallback(void * ptr, size_t size, size_t nmemb, void * buf);
};

/**
 * Input oriented StreamBuf which wraps libcurl requests
 *
 * An implementation of std::streambuf which provides access to the libcurl request for data download.
 */
class CurlIStreamBuf : public std::streambuf
{
public:
    /** Initializes a new libcurl session */
    CurlIStreamBuf(
        ResourceId &     identifier, ///< The resource identifier
        OptionSet const& options     ///< The advanced access options
        );
    
    /** Terminates the libcurl session */
    ~CurlIStreamBuf() { cleanup(); }

    /** */

    /** Performs unload cleanup for the DLL */
    static void onUnload();

protected:

    virtual int underflow();

private:
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

private:
    CURL * m_easyhandle; ///< Session easy handle

    std::exception_ptr      m_exception; ///< Pointer to internal exception
    std::list<DataBuffer>   m_data;      ///< Internal data buffers
    boost::mutex            m_mutex;     ///< Data buffer mutex

    boost::condition_variable m_cond;   ///< Data buffer empty condition

    /** Recieves downloaded data from the Resource */
    size_t onDataReady(char * ptr, size_t bytes);

    /** Terminates the session and closes active handles */
    void cleanup();

    /** Callback function for the libcurl library which redirects to onData */
    static size_t onDataReadyCallback(char * ptr, size_t size, size_t nmemb, void * buf);
};

} // namespace vox

// End definition
#endif // SIO_CURL_STREAMBUF_H
