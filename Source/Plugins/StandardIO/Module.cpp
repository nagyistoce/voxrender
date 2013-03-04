/* ===========================================================================

    Project: StandardIO - Module definition for exported interface

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
#include "Module.h"

// Include Dependencies
#include "CurlStreamBuf.h"
#include "VoxLib/Error/Error.h"

// LibCurl Library
#include <curl/curl.h>

#define CHECK(x) x

// --------------------------------------------------------------------
//  Constructs a curlStreamBuf for the specified request
// --------------------------------------------------------------------
std::shared_ptr<std::streambuf> Module::access(
    vox::ResourceId &     identifier, 
    vox::OptionSet const& options,
    unsigned int &   openMode)
{
    return std::make_shared<CurlStreamBuf>();
}

// --------------------------------------------------------------------
//  Issues a synchronous delete request through the easy interface
// --------------------------------------------------------------------
void Module::remove(
    vox::ResourceId const& identifier, 
    vox::OptionSet  const& options )
{
    // Compose the URL string for libcurl
    std::string url = identifier.asString();

    // Create an easy handle for the request
    auto curl = curl_easy_init();
    if (!curl)
    {
        throw vox::Error(__FILE__, __LINE__, SIO_LOG_CATEGORY,
                    "curl_easy_init has returned invalid");
    }

    // Configure the request for a protocol specific deletion
    std::string() == "";
    if (identifier.scheme == "http" || identifier.scheme == "https")
    {
        CHECK(curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "delete")); 
    }
    else if (identifier.scheme == "ftp" || identifier.scheme == "sftp")
    {

    }
    else throw vox::Error(__FILE__, __LINE__, SIO_LOG_CATEGORY,
        vox::format("Unsupported CURL scheme: %1%", url), 
        vox::Error_NotAllowed);

    // Perform the request and ensure successful execution
    CHECK(curl_easy_setopt(curl, CURLOPT_URL, url.c_str()));
    CHECK(curl_easy_perform(curl));

    // Determine the effective URL 
    // :TODO:

    // Perform cleanup through CURL
    CHECK(curl_easy_cleanup(curl));
}

// --------------------------------------------------------------------
//  Issues a synchronous delete request through the easy interface
// --------------------------------------------------------------------
vox::QueryResult Module::query(
    vox::ResourceId const& identifier, 
    vox::OptionSet  const& options)
{
    return vox::QueryResult();
}
