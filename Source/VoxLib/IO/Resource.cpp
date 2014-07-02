/* ===========================================================================

	Project: VoxRender - Resource

	Description: Implements a stream based interface for resource management

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
#include "Resource.h"

// Include Dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Core/Format.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/Error/ErrorCodes.h"

// 3rd Party Dependencies
#include "boost/property_tree/ptree.hpp"

// API namespace
namespace vox
{

// Filescope
namespace {
namespace filescope {
    
    static std::map<String,ResourceModuleH> modules; // Resource modules

    static boost::shared_mutex moduleMutex; // Module access mutex for read-write locks

    // --------------------------------------------------------------------
    //  Verifies the validity of the input scheme according to the RFC
    //  http://tools.ietf.org/html/rfc3986#section-3.1
    // --------------------------------------------------------------------
    void verifyScheme(String const& scheme)
    {
        static const Char * validChars = "";

        if (false)//!boost::is_any_not_of(scheme, validChars))
        {
            throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                        "Invalid scheme", Error_BadToken);
        }
    }

} // namespace filescope
} 

// Static member initialization
ResourceId Resource::m_globalBaseUri("file", "", "", "", "", false);

// ----------------------------------------------------------------------------
//  Initializes a null IOStream
// ----------------------------------------------------------------------------
ResourceStream::ResourceStream() : 
    std::istream(0), std::ostream(0) 
{ 
    m_setMask = Mode_Output | Mode_Input; 
}

// ----------------------------------------------------------------------------
//  Wraps a streambuffer as an IOStream
// ----------------------------------------------------------------------------
ResourceStream::ResourceStream(
    std::shared_ptr<std::streambuf> buffer, 
    unsigned int openMode
    ) :
  std::istream(0), std::ostream(0)
{
    open(buffer, openMode);
}

// ----------------------------------------------------------------------------
//  Registers a new resource retrieval module if the scheme is valid
// ----------------------------------------------------------------------------
void Resource::registerModule(
    String const&   scheme,
    ResourceModuleH module
    )
{ 
    if (!module) throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
        "registerModule requires valid handle", vox::Error_Range);

    // Acquire a read-lock on the modules for thread safe removal support
    boost::unique_lock<decltype(filescope::moduleMutex)> lock(filescope::moduleMutex);

    filescope::verifyScheme(scheme);

    filescope::modules[scheme] = module; 
}

// ----------------------------------------------------------------------------
//  Wraps a streambuffer as a Resource
// ----------------------------------------------------------------------------
void Resource::open(std::shared_ptr<std::streambuf> buffer, unsigned int openMode)
{ 
    if (m_buffer)
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                    "Resource already open", Error_NotAllowed);
    }

    m_buffer = buffer;

    // Apply forced mode settings to the requested mode
    m_openMode = (openMode|m_setMask);
    if ( (openMode & Mode_Append) == Mode_Append )
    {
        m_openMode &= ((~Mode_Truncate)|Mode_Output);
    }

    m_identifier = ""; // Indicates not-a-resource

    // Set the new source buffer
    rdbuf(m_buffer.get());
}

// ----------------------------------------------------------------------------
//  Attempts to open the specified resource for input using the first
//  opener whose regular expression matches the identifier.
// ----------------------------------------------------------------------------
void Resource::open(
    ResourceId const& identifier, 
    OptionSet const&  options,
    unsigned int      openMode 
    )
{ 
    if (m_buffer)
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                    "Resource already open", Error_NotAllowed);
    }

    // Apply forced mode settings to the requested mode
    m_openMode = (openMode|m_setMask);
    if ( (openMode & Mode_Append) == Mode_Append )
    {
        m_openMode &= ((~Mode_Truncate)|Mode_Output);
    }

    // Apply (potentially) relative reference URIs to the application base
    m_identifier = m_globalBaseUri.applyRelativeReference(identifier, true); 

    // Acquire a read-lock on the modules for thread safe removal support
    boost::shared_lock<decltype(filescope::moduleMutex)> lock(filescope::moduleMutex);

      // Verify the resource scheme has a registered handler
      auto & module = filescope::modules.find(m_identifier.scheme);
      if (module == filescope::modules.end())
      {
          throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
                      "No ResourceRetriever found", Error_BadToken);
      }

      // Acquire the resource streambuffer from the retriever
      m_buffer = module->second->access(m_identifier, options, m_openMode);

    lock.unlock(); // Release the read-lock

    // Set the new source buffer
    rdbuf( m_buffer.get() );
}

// ----------------------------------------------------------------------------
//  Performs a query operation on the specified resource
// ----------------------------------------------------------------------------
std::shared_ptr<QueryResult> Resource::query(ResourceId const& identifier, OptionSet const& options)
{
    // Apply (potentially) relative reference URIs to the application base
    ResourceId fullUri = m_globalBaseUri.applyRelativeReference(identifier, true); 
    
    // Acquire a read-lock on the modules for thread safe removal support
    boost::shared_lock<decltype(filescope::moduleMutex)> lock(filescope::moduleMutex);

    // Verify the resource scheme has a registered handler
    auto & module = filescope::modules.find(fullUri.scheme);
    if (module == filescope::modules.end())
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
                    "No ResourceRemover found", Error_BadToken);
    }

    // Acquire the resource streambuffer from the retriever
    return module->second->query(fullUri, options);
}

// --------------------------------------------------------------------
//  Issues a delete request for the specified resource
// --------------------------------------------------------------------
void Resource::remove(ResourceId const& identifier, OptionSet const& options)
{
    // Apply (potentially) relative reference URIs to the application base
    ResourceId fullUri = m_globalBaseUri.applyRelativeReference(identifier, true); 
    
    // Acquire a read-lock on the modules for thread safe removal support
    boost::shared_lock<decltype(filescope::moduleMutex)> lock(filescope::moduleMutex);

    // Verify the resource scheme has a registered handler
    auto & module = filescope::modules.find(fullUri.scheme);
    if (module == filescope::modules.end())
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
                    "No ResourceRemover found", Error_BadToken);
    }

    // Acquire the resource streambuffer from the retriever
    module->second->remove(fullUri, options);
}

// --------------------------------------------------------------------
//  Removes the resource module for a specified scheme
// --------------------------------------------------------------------
void Resource::removeModule(String const& scheme)
{
    // Acquire a write-lock on the modules for thread safe removal support
    boost::unique_lock<decltype(filescope::moduleMutex)> lock(filescope::moduleMutex);

    filescope::modules.erase(scheme);
}

// --------------------------------------------------------------------
//  Removes all registered instances of the specified IO Module
// --------------------------------------------------------------------
void Resource::removeModule(ResourceModuleH module)
{
    // Acquire a write-lock on the modules for thread safe removal support
    boost::unique_lock<decltype(filescope::moduleMutex)> lock(filescope::moduleMutex);

    auto iter = filescope::modules.begin();
    while (iter != filescope::modules.end())
    {
        if (iter->second == module)
        {
            auto old = iter; ++iter;
            filescope::modules.erase(old);
        }
        else
        {
            ++iter;
        }
    }
}

// --------------------------------------------------------------------
//  Returns the number of bytes remaining in the stream (if seekable)
// --------------------------------------------------------------------
std::streamsize ResourceIStream::remaining()
{
    // :TODO: Check seekable

    std::streamsize pos = tellg();
    seekg(std::ios::end);
    std::streamsize end = tellg();
    seekg(pos);

    return end-pos;
}

} // namespace vox