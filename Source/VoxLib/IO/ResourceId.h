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

// Begin definition
#ifndef VOX_RESOURCE_ID_H
#define VOX_RESOURCE_ID_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Types.h"
#include "VoxLib/IO/MimeTypes.h"
#include "VoxLib/IO/OptionSet.h"

// API Namespace
namespace vox
{

/**
 * Uniform Resource Identifier
 *
 * The generic syntax for a uniform resource identifier as it pertains to
 * this interface follows the standards outlined in RFC-3986 and RFC-3987
 * (http://tools.ietf.org/html/rfc3986) and http://tools.ietf.org/html/rfc3987)  
 * which provides this basic outline:
 *
 * URI = <scheme name> :// <authority> <path> [ ? <query> ] [ # <fragment> ]
 *
 * Where the [ <userinfo> @ ] <host> [ : <port> ] portion of the <authority> of the
 * URI is handled as a single hierarchical component internally to facilitate
 * variations in schemes. As such, retrieval modules are responsible for the
 * parsing and verification of the URI <authority> component.
 *
 * \note The scheme component of an IRI (Internationalized RI) is still ASCII based
 *
 * The resource identifier will from hereforth be referred to as a URI, though 
 * support of unencoded Unicode characters is dependent on the resource
 * retrieval modules implementation.
 *
 * Detections of violations of this format during management of the URI
 * by any part of the abstract IO interface will and should result in the
 * rejection of the URI for the current operation (Typically indicated
 * by throwing an exception) rather than a silent failure or ignorance of
 * the violating characters.
 *
 * <b>Reserved Characters</b>
 *
 * Even within the URL subcomponents, certain characters are reserved. These 
 * characters are specified in the subcomponent descriptions below. These 
 * additional reserved characters are characters which will not be percent '
 * decoded when decrypting the URI. In addition, these characters will not
 * be percent-encoded when recomposing the URI if they are added by the user
 * directly to the component in an unencoded form.
 *
 * <b>Normalization</b>
 * 
 * Normalization of the scheme component of the resource URI is performed automatically
 * by the IO frontend. The normalization of other components of the URI however, if desired,
 * is delegated to the retrieval module. This class provides several helper functions for
 * performing common normalization operations.
 *
 * Wikipedia also provides some information on more advanced normalization techniques:
 * http://en.wikipedia.org/wiki/URL_normalization
 *
 * <b>Scheme</b> 
 * 
 * http://tools.ietf.org/html/rfc3986#section-3.1
 *
 * Reserves: "" (RFC standards do not permit pct-encoded URL components)
 *
 * The scheme component of the URI is used for the selection of a registered
 * resource retrieval module. The scheme component of the URI is optional and
 * if it is not detected will be assumed to refer to the application default.
 * Unless otherwise specified by the user, this will typically be 'file'. 
 *
 * For reference, the officially registered URI schemes are listed by the IANA
 * at the following page:
 *
 * http://www.iana.org/assignments/uri-schemes.html
 * 
 * <b>Authority</b>
 *
 * Reserves: "@:%"
 *
 * http://tools.ietf.org/html/rfc3986#section-3.2
 *
 * As far as the abstract IO interface is concerned, the authority component of the URI
 * is completely transparent. Validation and parsing of the authority is delegated to the
 * retriever. It is however, recommended that retrieval modules enforce the standards
 * of the RFC when parsing the <authority> component of the URI.
 *
 * <b>Path</b>
 *
 * Reserves: "/%:"
 *
 * http://tools.ietf.org/html/rfc3986#section-3.3
 *
 * The path component of the URI is utilized by the library for resolving relative URIs.
 * In accordance with the URI standards, the only accepted segment delimiter character is
 * the forward slash ('/'). Windows system users should convert all backslashes prior to
 * decomposing the URI string into a ResourceId.
 *
 * <b>Query</b>
 *
 * Reserves: "=&%"
 *
 * http://tools.ietf.org/html/rfc3986#section-3.4
 *
 * The specifications for the format of the query component is dependent on the scheme and
 * authority. As such the verification of the query component of the URI will likely be handled
 * by the module which executes the actual dereferencing of the URI. As the RFC mentions however,
 * many query systems utilize simple key value pairs in the format ?key=value&key=value ...
 * In order to simply the process of URI management, the IO library provides support for managing 
 * these key : value pairs through the QueryParams helper class.
 *
 * <b>Fragment</b>
 *
 * http://tools.ietf.org/html/rfc3986#section-3.5
 *
 * The fragment identifier is specified to be dependent on the media type of the retrieved
 * resource. Although some allowances are made within this library for extraction and anticipation 
 * of media types based on the URI file extension the library does not enforce the standards 
 * of the fragment identifier and again delegates responsibility to the retriever module for
 * ensuring correct use of the fragment component of the URI.
 *
 * <b>Escape Encoding</b>
 *
 * http://tools.ietf.org/html/rfc3986#section-2.4
 *
 * Before decomposing a URI into a ResourceID using the direct decomposition constructor, the URI
 * should be completely escaped. This ensures the URI components can be correctly parsed and is
 * in line with the recommendations of the RFC. When using the direct component initialization 
 * constructor, the content will be assumed to be in an unencoded form unless otherwise specified.
 *
 * <b>Motivations</b>
 *
 * The forced conformance of the URI syntax and structure provides a method
 * of specifying resource retrieval modules which define and implement the protocols 
 * for accessing a resource. In this way, it becomes possible to implement 
 * resource retrieval for a given scheme such that the retrieval can be implemented 
 * to cache and retrieve data from sources outside the context of the original 
 * scheme. (Such as a network request caching resources to the local filesystem)
 *
 * It also provides a simple mechanism for programs which may need to resolve simple URNs
 * to a specific URL during runtime. A "urn" scheme module can be configured to resolve
 * the resource URL and call back into the abstract resource IO layer to retrieve
 * a streambuffer for accessing the specified resource without requiring the user
 * to explicitly resolve the URN.
 *
 * Note that resolution of actual URI through redirection handling would then have to
 * properly resolve the URI references (http://tools.ietf.org/html/rfc3986#section-5.1.3) 
 * to maintain adherence to the standards.
 */
struct VOX_EXPORT ResourceId
{
    // URI comparison operators
    bool operator == (ResourceId const& rhs) const;
    bool operator != (ResourceId const& rhs) const;
    bool operator <  (ResourceId const& rhs) const;
    bool operator <= (ResourceId const& rhs) const;
    bool operator >  (ResourceId const& rhs) const;
    bool operator >= (ResourceId const& rhs) const;

    /** Direct component initialization constructor */
    ResourceId(
        String const& ischeme,
        String const& iauthority,
        String const& ipath,
        String const& iquery,
        String const& ifragment,
        bool decode
        );

    /** Decomposes a URI string into a ResourceId structure */   
    ResourceId(String const& uri = "")
    {
        delegatedConstructor(uri);
    }

    /** Decomposes a URI string into a ResourceId structure */
    ResourceId(Char const* uri)
    {
        delegatedConstructor(String(uri));
    }

    /** Converts a relative URI using this ResourceId as a base */
    ResourceId applyRelativeReference(ResourceId const& relativeUri, bool strict = false) const;
    
    /** Operator overloaded version of relative reference application */
    ResourceId operator+(ResourceId const& right) const
    {
        return applyRelativeReference(right);
    }
    
    /** Operator overloaded version of relative reference application */
    ResourceId & operator+=(ResourceId const& right) 
    {
        return *this = applyRelativeReference(right);
    }

    /** Returns a dot segment normalized path component */
    String normalizedPath() const;

    /** 
     * Normalizes the path component of the ResourceId 
     *
     * This function should be used with reservations. This function is not technically
     * gaurenteed to preserve semantics. Filesystem resources for example may not 
     * resolve dot segments in a consistent manner due to the existance of symlinks
     * in the input path. Normalizing the path component of the URI should generally be 
     * left to the URI's dereferencer.
     * 
     * @returns A ResourceId with no unecessary dot segments or capitalized percent encoding chars
     */
    ResourceId const& normalizePath() 
    { 
        path = normalizedPath(); return *this; 
    }

    /** Extracts the query parameters as an OptionSet */
    OptionSet extractQueryParams() const;

    /** Appends query parameters to the URI */
    void appendQueryParams(OptionSet const& params)
    {
        BOOST_FOREACH(auto & entry, params) 
        {
            appendQueryParam(entry.first, entry.second);
        }
    }

    /** Appends a query parameter to the URI */
    template<typename T> void appendQueryParam(String const& key, T const& val)
    {
        try { appendQueryParam(key, boost::lexical_cast<String>(val)); }
        catch (boost::bad_lexical_cast &)
        {
            throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
                        format("Base Cast <key=%1%> <typename=%2%>", 
                               key, typeid(T).name()),
                        Error_BadToken);
        }
    }

    /** Appends a query parameter to the URI */
    void appendQueryParam(String const& key, String const& val);

    /** Sets the query parameters of the URI */
    void setQueryParams(OptionSet const& params)
    {
        query.clear(); appendQueryParams(params);
    }

    /** Extracts the userinfo component of the authority */
    String extractUserinfo() const;

    /** Extracts the hostname component of the authority */
    String extractHostname() const;
    
    /** Extracts the hostname component of the authority */
    String extractNormalizedHostname() const
    {
        return boost::to_lower_copy(extractHostname());
    }

    /** 
     * Extracts the port number from the URI 
     *
     * If a port number is not found, the returned value will be -1
     * If a port number is out of the valid range, an exception will be thrown
     *
     * @param evaluateToDefault If true, missing port numbers will return scheme defaults
     */
    int extractPortNumber(bool evaluateToDefault = true) const;

    /** Attempts to extract a file extension from the URI */
    String extractFileExtension() const;

    /** Attempts to extract a filename from the URI */
    String extractFileName() const
    {
        auto fileDelim = path.find_last_of('/');
        if (fileDelim == path.npos)
        {
            return path;
        }
   
        return path.substr(fileDelim+1); 
    }

    /** Guesses the mime-type of the resource */
    String guessMimeType() const
    {
        String const extension = extractFileExtension();
        return MimeTypes::getType(extension);
    }

    /** Performs encoding of URI characters. Reserved chars are percent encoded */
    static String uriEncode(String const& string, String const& reserved);

    /** Performs decoding of URI characters. Reserved chars are left encoded */
    static String uriDecode(String const& string, String const& reserved);

    /** 
     * Converts the ResourceId to URI string 
     *
     * This conversion does not perform any normalization of the URI string but 
     * does percent encode a range of common characters. The result is a fully
     * qualified URI string.
     *
     * http://tools.ietf.org/html/rfc3986#section-5.3
     */
    String asString() const;

    String scheme;      ///< Scheme
    String authority;   ///< Authority
    String path;        ///< Path 
    String query;       ///< Query Parameters
    String fragment;    ///< Fragment Id

private:
    // Delegated constructor for lacking C++x11 support
    void delegatedConstructor(String const& string);
};

/** 
 * Formatted stream insertion overload for ResourceId 
 *
 * This function outputs the ResourceId formatted as an escaped URL string 
 * to the output stream.
 */
 inline std::ostream & operator<<(std::ostream & os, ResourceId const& id)
 {
     return os << id.asString();
 }

}

// End definition
#endif // VOX_RESOURCE_ID_H