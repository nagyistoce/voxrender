/* ===========================================================================

	Project: Uniform Resource IO 
    
	Description: Defines a ResourceId structure for encapsulating URIs

    Copyright (C) 2012-2013 Lucas Sherman

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
#include "ResourceId.h"

// Include Dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Core/Format.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/Error/ErrorCodes.h"

#include <iomanip>

// API namespace
namespace vox
{

// Filescope
namespace {
namespace filescope {
    
    // --------------------------------------------------------------------
    //  Removes the last segment from the input path 
    // --------------------------------------------------------------------
    void removeLastSegment(String & path)
    {
        auto pathDelim = path.find_last_of('/');
        if (pathDelim != path.npos)
        {
            path = path.substr(0, pathDelim);
        }
        else
        {
            path.clear();
        }
    }

    // --------------------------------------------------------------------
    //  Merges the base and relative URI paths to form a fully qualified path
    //  http://tools.ietf.org/html/rfc3986#section-5.2.3
    // --------------------------------------------------------------------
    String mergePaths(ResourceId const &B, ResourceId const& R)
    {
        if (!B.authority.empty() && B.path.empty())
        {
            return '/' + R.path;
        }

        auto rootDelim = B.path.find_last_of('/');
        if (rootDelim == B.path.npos)
        {
            return R.path;
        }

        return B.path.substr(0, rootDelim+1) + R.path;
    }

} // namespace filescope
} // namespace anonymous

//
// http://cplusplus.bordoon.com/speeding_up_string_conversions.html //
//

// --------------------------------------------------------------------
//  Performs URI encoding of input strings
// --------------------------------------------------------------------
String ResourceId::uriEncode(String const& string, String const& reserved)
{
    // :TODO: consider threadlocal static stream for encode

    OStringStream result;

    BOOST_FOREACH (auto & c, string)
    {
        if (reserved.find(c) == reserved.npos)
        {
            result << c;
        }
        else
        {
            result << '%' << std::hex << std::setfill('0') << std::setw(2) 
                   << static_cast<unsigned int>(c)  << std::dec;
        }
    }

    return result.str();
}

// --------------------------------------------------------------------
//  Performs URI decoding of input strings
// --------------------------------------------------------------------
String ResourceId::uriDecode(String const& string, String const& reserved)
{
    String result;

    auto size = string.size();
    result.reserve(size);

    for (String::size_type p = 0; p < string.size(); ++p)
    {
        if (string[p] == '%')
        {
            // :TODO: consider threadlocal static stream for decode

            if (string.length() < p+3) // Check for early termination
            {
                throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                            "Unfinished pct-encoded octet in URI",
                            Error_BadToken);
            }

            IStringStream stream( string.substr(p+1, 2) );
            int c; stream >> std::hex >> c;
            result += static_cast<Char>(c);
            
            if (!stream.eof()) // Check for bad hex characters
            {
                throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                            "Invalid pct-encoded octet in URI",
                            Error_BadToken);
            }

            p += 2; // Advance read head
        }
        else if (string[p] == '+') result += ' '; // Permit the common non-standard usage of '+' for space characters
        else                       result += string[p];
    }

    return result;
}

// --------------------------------------------------------------------
//  Constructs a ResourceId structure from base URI components
// --------------------------------------------------------------------
ResourceId::ResourceId(
    String const& ischeme,
    String const& iauthority,
    String const& ipath,
    String const& iquery,
    String const& ifragment,
    bool decode
    )
{
    if (decode)
    {
        scheme    = uriDecode(ischeme,    ""   );
        authority = uriDecode(iauthority, "@:%");
        path      = uriDecode(ipath,      "/:%");
        query     = uriDecode(iquery,     "&=%");
        fragment  = uriDecode(ifragment,  ""   );
    }
    else
    {
        scheme    = ischeme;
        authority = iauthority;
        path      = ipath;
        query     = iquery;
        fragment  = ifragment;
    }
}

// --------------------------------------------------------------------
//  Decomposes the URI string into its 5 basic components
// --------------------------------------------------------------------
void ResourceId::delegatedConstructor(String const& uri)
{
    static const boost::regex uriRegex("^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\\?([^#]*))?(#(.*))?");

    boost::match_results<String::const_iterator> what;
    
    if (!boost::regex_search(uri, what, uriRegex))
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                    "Invalid URI syntax", Error_Syntax);
    }

    scheme    = String(what[2].first, what[2].second);
    authority = String(what[4].first, what[4].second);
    path      = String(what[5].first, what[5].second);
    query     = String(what[7].first, what[7].second);
    fragment  = String(what[9].first, what[9].second);

    scheme    = uriDecode(scheme,    ""   );
    authority = uriDecode(authority, "%:@");
    path      = uriDecode(path,      "/%" );
    query     = uriDecode(query,     "%&=");
    fragment  = uriDecode(fragment,  ""   );

    // Perform case normalization
    boost::to_lower(scheme);
}

// --------------------------------------------------------------------
//  Merges this URI base and a relative URI into a target URI
//  http://tools.ietf.org/html/rfc3986#section-5.2.1
// --------------------------------------------------------------------
ResourceId ResourceId::applyRelativeReference(ResourceId const& R, bool strict) const
{
    ResourceId const& B = *this; ResourceId T;

    // Compose the target URI 
    if (!R.scheme.empty() && (strict || R.scheme != B.scheme))
    {
         T.scheme    = R.scheme;
         T.authority = R.authority;
         T.path      = R.normalizedPath();
         T.query     = R.query;
    }
    else
    {
        if (!R.authority.empty())
        {
            T.authority = R.authority;
            T.path      = R.normalizedPath();
            T.query     = R.query;
        }
        else
        {
            if (R.path.empty())
            {
                T.path = B.path;
                if (!R.query.empty())
                {
                    T.query = R.query;
                }
                else
                {
                    T.query = B.query;
                }
            }
            else
            {
                if (R.path.front() == '/')
                {
                    T.path = R.normalizedPath();
                }
                else
                {
                    T.path = filescope::mergePaths(B, R);
                    T.normalizePath();
                }
                T.query = R.query;
            }
            T.authority = B.authority;
        }
        T.scheme = B.scheme;
    }

    T.fragment = R.fragment;

    return T;
}

// --------------------------------------------------------------------
//  Returns the path component after dot segment normalization 
//  http://tools.ietf.org/html/rfc3986#section-5.2.4
//  :TODO: Case normalize encoded chars
// --------------------------------------------------------------------
String ResourceId::normalizedPath() const
{
    String::size_type readHead = 0;
    String const&     input    = path;

    String result;
    while (readHead < input.size())
    {
        // (A) Remove leading dot segments
        if (input.find("../", readHead) == readHead) readHead += 3;
        else if (input.find("./", readHead) == readHead) readHead += 2;

        // (B) Remove current directory dot segments
        else if (input.find("/./", readHead) == readHead) readHead += 2; 
        else if (input.substr(readHead) == "/.") { result+='/'; break; }

        // (C) Execute previous directory segments
        else if (input.find("/../", readHead) == readHead) 
        {
            readHead+=3;
            filescope::removeLastSegment(result);
        }
        else if (input.substr(readHead) == "/..") 
        {
            readHead+=3;
            filescope::removeLastSegment(result);
            break;
        }

        // (D) Remove null relative references
        else if (input.substr(readHead) == ".." || input.substr(readHead) == ".") readHead = input.npos; 
            
        // (E) Transfer the next path segment
        else
        {
            auto pathDelim = input.find('/', readHead+1);
            result += input.substr(readHead, pathDelim-readHead);
            readHead = pathDelim;
        }
    }

    return result;
}

// --------------------------------------------------------------------
//  Extracts the query parameters as an option set
// --------------------------------------------------------------------
OptionSet ResourceId::extractQueryParams() const
{
    OptionSet queryParams;

    String::size_type readhead = 0;

    while (true)
    {
        // Determine substring containing next parameters
        auto keyEnd = query.find('=', readhead);
        if (keyEnd == query.npos) // Missing '=' seperator
        {
            throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                        "Invalid Query Syntax", Error_Syntax);
        }

        auto valEnd = query.find('&', keyEnd);

        // Extract and decode the next key/value pair
        auto key = query.substr(readhead, keyEnd-readhead);
        auto val = query.substr(keyEnd+1, valEnd-keyEnd-1);

        queryParams.insert(std::make_pair(
            uriDecode(key, ""),
            uriDecode(val, "")
            ));

        // Detect end of query string
        if (valEnd == query.npos) break; 

        readhead = valEnd+1; // Advance read head
    }

    return queryParams;
}

// --------------------------------------------------------------------
//  Extracts the userinfo from the authority component of the URI
// --------------------------------------------------------------------
String ResourceId::extractUserinfo() const
{
    auto end = authority.find('@');
    if (end == authority.npos) 
        return String();

    return authority.substr(0, end);
}

// --------------------------------------------------------------------
//  Extracts the hostname from the authority component of the URI
// --------------------------------------------------------------------
String ResourceId::extractHostname() const
{
    auto beg = authority.find('@');
    if (beg == authority.npos) beg = 0;
    else beg += 1; // Skip '@"

    auto end = authority.find(':', beg);

    return authority.substr(beg, end-beg);
}

// --------------------------------------------------------------------
//  Extracts the port number from the authority component of the URI 
// --------------------------------------------------------------------
int ResourceId::extractPortNumber(bool evaluateToDefault) const
{
    auto beg = authority.find_last_of(':');
    if (beg == authority.npos) 
    {
        return -1;
    }

    auto endInfo = authority.find_first_of('@');
    if (endInfo != authority.npos && endInfo > beg) 
    {
        return -1;
    }

    auto portStr = authority.substr(beg+1);

    try
    {
        int port = boost::lexical_cast<int>(portStr);
        if (port > 65535)
        {
            throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                        "Invalid Port Number", Error_Range);
        }
        
        // :TODO: Evaluate default port

        return port;
    }
    catch(boost::bad_lexical_cast &)
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
                    "Invalid Port Syntax", Error_BadToken);
    }
}

// --------------------------------------------------------------------
//  Extracts the file extension from the path component of the URI 
// --------------------------------------------------------------------
String ResourceId::extractFileExtension() const
{
    String filename = extractFileName();
    auto extensionDelim  = filename.find_last_of('.');
    if (extensionDelim == filename.npos)
    {
        return String();
    }

    return filename.substr(extensionDelim);
}

// --------------------------------------------------------------------
//  Appends a key/value pair to the query component of the URI
// --------------------------------------------------------------------
void ResourceId::appendQueryParam(String const& key, String const& val)
{
    if (!query.empty()) query += '&';

    query += uriEncode(key,"%&=");
    query += '=';
    query += uriEncode(val,"%&=");
}

// --------------------------------------------------------------------
//  Composes the URI components into a fully qualified string format
// --------------------------------------------------------------------
String ResourceId::asString() const
{ 
    String uri;

    if (!scheme.empty())
    {
        uri += scheme; // RFC prohibits pct-encoded chars
        uri += ':';
    }

    if (!authority.empty()) 
    {
        uri += "//";
        uri += uriEncode(authority, "<>!*'();&=+$,/?#[]");
    }

    uri += uriEncode(path, " <>?#!*'();&=+$,[]");

    if (!query.empty())
    {
        uri += '?';
        uri += uriEncode(query, " <>?#!*'();<>#+$,[]");
    }

    if (!fragment.empty()) 
    {
        uri += '#';
        uri += uriEncode(fragment, " <>?#!*'();+$,[]");
    }

    return uri;
}

// --------------------------------------------------------------------
//  Comparison operators for ResourceIds
// --------------------------------------------------------------------
bool ResourceId::operator == (ResourceId const& rhs) const
{
    return (scheme    == rhs.scheme   ) &&
           (authority == rhs.authority) &&
           (path      == rhs.path     ) &&
           (query     == rhs.query    ) &&
           (fragment  == rhs.fragment );
}
bool ResourceId::operator != (ResourceId const& rhs) const
{
    return (scheme    != rhs.scheme   ) ||
           (authority != rhs.authority) ||
           (path      != rhs.path     ) ||
           (query     != rhs.query    ) ||
           (fragment  != rhs.fragment );
}
bool ResourceId::operator <  (ResourceId const& rhs) const
{
    return (scheme    < rhs.scheme   ) || ( (scheme    == rhs.scheme   ) &&
           (authority < rhs.authority) || ( (authority == rhs.authority) &&
           (path      < rhs.path     ) || ( (path      == rhs.path     ) &&
           (query     < rhs.query    ) || ( (query     == rhs.query    ) &&
           (fragment  < rhs.fragment ) ))));
}
bool ResourceId::operator <= (ResourceId const& rhs) const
{
    return (scheme    < rhs.scheme   ) || ( (scheme    == rhs.scheme   ) &&
           (authority < rhs.authority) || ( (authority == rhs.authority) &&
           (path      < rhs.path     ) || ( (path      == rhs.path     ) &&
           (query     < rhs.query    ) || ( (query     == rhs.query    ) &&
           (fragment  <= rhs.fragment ) ))));
}
bool ResourceId::operator >  (ResourceId const& rhs) const
{
    return (scheme    > rhs.scheme   ) || ( (scheme    == rhs.scheme   ) &&
           (authority > rhs.authority) || ( (authority == rhs.authority) &&
           (path      > rhs.path     ) || ( (path      == rhs.path     ) &&
           (query     > rhs.query    ) || ( (query     == rhs.query    ) &&
           (fragment  > rhs.fragment ) ))));
}
bool ResourceId::operator >= (ResourceId const& rhs) const
{
    return (scheme    > rhs.scheme   ) || ( (scheme    == rhs.scheme   ) &&
           (authority > rhs.authority) || ( (authority == rhs.authority) &&
           (path      > rhs.path     ) || ( (path      == rhs.path     ) &&
           (query     > rhs.query    ) || ( (query     == rhs.query    ) &&
           (fragment  >= rhs.fragment ) ))));
}

} // namespace vox