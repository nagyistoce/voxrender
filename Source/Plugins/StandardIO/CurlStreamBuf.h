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

// Begin definition
#ifndef SIO_CURL_STREAMBUF_H
#define SIO_CURL_STREAMBUF_H

// Include Dependencies
#include "Common.h"
#include "VoxLib/IO/Resource.h"
#include "VoxLib/IO/ResourceModule.h"

// LibCurl Library
#include <curl/curl.h>

/**
 * StreamBuf which wraps libcurl request
 *
 * An implementation of std::streambuf which provides access to the
 * libcurl request for data upload and download.
 */
class CurlStreamBuf : public std::streambuf
{
public:
    CurlStreamBuf(CURL * curl = nullptr) : m_curl(curl) {}

private:
    CURL * m_curl;
};

// End definition
#endif // SIO_CURL_STREAMBUF_H
