/* ===========================================================================

    Project: StandardIO - Standard IO protocols for VoxIO

    Description: A libcurl wrapper compatible with the VoxIO library

    Copyright (C) 2012 Lucas Sherman

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
#include "StandardIO.h"

// Include Dependencies
#include "StandardIO/CurlStreamBuf.h"
#include "VoxLib/Core/Logging.h"

// 3rd Party Dependencies
#include "boost/property_tree/ptree.hpp"

// LibCurl Library
#include <curl/curl.h>

#define CHECK(x) x

// API namespace
namespace vox {

// --------------------------------------------------------------------
//  Constructs a curlStreamBuf for the specified request
// --------------------------------------------------------------------
std::shared_ptr<std::streambuf> StandardIO::access(
    ResourceId &     identifier, 
    OptionSet const& options,
    unsigned int     openMode)
{
    // Input and Output mode are not supported simultaneously for libcurl protocols
    if ( (openMode&Resource::Mode_Input) && (openMode&Resource::Mode_Output) )
    {
        throw Error(__FILE__, __LINE__, SIO_LOG_CATEGORY,
            "Invalid openMode: Input and Output not supported",
            Error_NotImplemented);
    }

    // Return the proper (input / output) streambuffer wrapper
    if (openMode&Resource::Mode_Input)
    {
        return std::make_shared<vox::CurlIStreamBuf>(identifier, options);
    }
    else if (openMode&Resource::Mode_Output)
    {
        return std::make_shared<vox::CurlOStreamBuf>(identifier, options);
    }
    
    throw Error(__FILE__, __LINE__, SIO_LOG_CATEGORY, 
        "Invalid access flags (must specify read or write)", Error_Range);

    return nullptr; // Satisfy less intelligent compilers
}

// --------------------------------------------------------------------
//  Issues a synchronous delete request through the easy interface
// --------------------------------------------------------------------
void StandardIO::remove(
    ResourceId const& identifier, 
    OptionSet  const& options )
{
    // Compose the URL string for libcurl
    std::string url = identifier.asString();

    // Create an easy handle for the request
    auto curl = curl_easy_init();
    if (!curl)
    {
        throw Error(__FILE__, __LINE__, SIO_LOG_CATEGORY, "curl_easy_init has failed");
    }

    // Configure the request for a protocol specific deletion
    if (identifier.scheme == "http" || identifier.scheme == "https")
    {
        CHECK(curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "delete")); 
    }
    else if (identifier.scheme == "ftp" || identifier.scheme == "sftp")
    {
        // :TODO:
    }
    else throw Error(__FILE__, __LINE__, SIO_LOG_CATEGORY,
        format("Unsupported scheme: %1%", url), Error_NotAllowed);

    // Perform the request and ensure successful execution
    CHECK(curl_easy_setopt(curl, CURLOPT_URL, url.c_str()));
    CHECK(curl_easy_perform(curl));

    // Determine the effective URL 
    // :TODO:

    // Perform cleanup through CURL
    CHECK(curl_easy_cleanup(curl));
}

// --------------------------------------------------------------------
//  Queries for information depending on the specified scheme
// --------------------------------------------------------------------
std::shared_ptr<QueryResult> StandardIO::query(
    ResourceId const& identifier, 
    OptionSet  const& options)
{
    return std::make_shared<QueryResult>();
}

} // namespace vox