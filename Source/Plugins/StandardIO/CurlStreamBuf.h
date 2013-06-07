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
#include "StandardIO/AsioService.h"
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
    /** Dispatches the modify request to the io service */
    CurlOStreamBuf(
        ResourceId &     identifier, ///< The resource identifier
        OptionSet const& options     ///< The advanced access options
        );

    /** Detaches the request for asynchronous completion/cancellation */
    ~CurlOStreamBuf();

protected: 
    virtual int overflow();

private:
    class Impl; std::shared_ptr<Impl> m_pImpl;
};

/**
 * Input oriented StreamBuf which wraps libcurl requests
 *
 * An implementation of std::streambuf which provides access to the contents of a data request
 */
class CurlIStreamBuf : public std::streambuf
{
public:
    /** Dispatches the access request to the io service */
    CurlIStreamBuf(
        ResourceId &     identifier, ///< The resource identifier
        OptionSet const& options     ///< The advanced access options
        );
    
    /** Detaches the request for asynchronous completion/cancellation */
    ~CurlIStreamBuf();

protected:
    virtual int underflow();

private:
    class Impl; std::shared_ptr<Impl> m_pImpl;
};

} // namespace vox

// End definition
#endif // SIO_CURL_STREAMBUF_H
