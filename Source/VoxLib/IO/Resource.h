/* ===========================================================================

	Project: Uniform Resource IO 
    
	Description: Implements a uniform resource IO interface 

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

/** 
 * The Resource class and its stream based derivations ResourceIStream,
 * ResourceOStream, and ResourceStream provide a mechanism for identifying
 * and accessing resources abstractly. Resource access is provided by
 * ResourceId objects which adhere to the guidelines of the generic URI
 * standards outline in RFC-3986.
 *
 * @sa ResourceId
 */

// Begin definition
#ifndef VOX_RESOURCE_H
#define VOX_RESOURCE_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Types.h"
#include "VoxLib/IO/ResourceId.h"
#include "VoxLib/IO/ResourceModule.h"

// API Namespace
namespace vox 
{

typedef std::shared_ptr<ResourceModule> ResourceModuleH;

/**
 * Abstract IO ios class
 *
 * This class is the equivalent of std::ios for the ResourceStreams and defines the actual
 * functional component of the IO library, resource retrieval. Resource retrieval itself is
 * managed by registered resource modules, however the Resource interface forms the abstraction
 * between the resource retrieval modules and the resource request. A request ResourceId is 
 * provided to the resource open() member, and the ID is parsed to determine which retrieval 
 * module registered with the library has been registered to the scheme of the ResourceId. The 
 * retrieval module is then executed and the results of the retrieval made accessable through
 * the iostream interface.
 */
class VOX_EXPORT Resource : virtual std::ios
{
public:
    /** 
     * Stream mode options 
     *
     * When opening a resource, additional requests can be made as to the condition of the provided streambuffer.
     * These options are provided as a bitset to the abstracter open interface and can be overridden by the selected 
     * ResourceOpener should it be unable to provide the requested option. The mode settings provided by the 
     * ResourceOpener can be accessed through the member accessor mode().
     *
     * The specification of the output mode flag for a stream which does not allow input will automatically be 
     * interpreted to have the Truncate flag. 
     *
     */
    enum Mode
    {
        Mode_Input           = 1<<0,                                        ///< Allows input operations
        Mode_Output          = 1<<1,                                        ///< Allows output operations
        Mode_Bidir           = 1<<2 | Mode_Input | Mode_Output,             ///< Independent input and output sequences
        Mode_InputSeekable   = 1<<3 | Mode_Input,                           ///< Allow get head positioning
        Mode_OutputSeekable  = 1<<4 | Mode_Output,                          ///< Allow put head positioning
        Mode_Seekable        = Mode_InputSeekable | Mode_OutputSeekable,    ///< Allow get and put head positioning
        Mode_DualSeekable    = 1<<5 | Mode_Seekable,                        ///< Has unique get and put heads for positioning
        Mode_BidirSeekable   = Mode_Bidir | Mode_Seekable,                  ///< Is bidirectional and allow seeking
        Mode_Append          = 1<<6 | Mode_Output,                          ///< Reposition put pointer to EOF on write
        Mode_StartAtEnd      = 1<<7,                                        ///< Start read/write heads at EOF position 
        Mode_Truncate        = 1<<8 | Mode_Output                           ///< Create resource or discard existing resource content
    };

    /** Returns the mode options of the stream */
    unsigned int mode() const throw() { return m_openMode; }

    /** Returns true if the stream is open, false otherwise */
    inline bool isOpen() throw() { return (m_buffer) ? true : false; }

    /** 
     * Open a new resource for reading
     *
     * @param identifier The URI for the resource
     * @param options    An option set for the loader
     * @param openMode   A bitset of OpenMode options
     */
    void open(
        ResourceId const& identifier, 
        OptionSet const&  options     = OptionSet(),
        unsigned int      openMode    = 0
        );

    /** 
     * Open a new resource for reading
     *
     * @param identifier The URI for the resource
     * @param openMode   A bitset of OpenMode options
     */
    inline void open(ResourceId const& identifier, unsigned int openMode)
    {
        open(identifier, OptionSet(), openMode);
    }

    /** Closes a resource stream */
    inline void close() throw()
    {
        if (m_buffer)
        {
            rdbuf(0); // Release me first

            m_buffer.reset();
            m_openMode = 0;
        }
    }
    
    /** Returns The resource identifier associated with this stream */
    inline ResourceId const& identifier() const throw() { return m_identifier; }

    /** Returns The mime-type associated with this stream */
    inline String const& mimeType() const throw() { return m_mimeType; }

    /** Returns a shared pointer to the underlying streambuffer */ 
    inline std::shared_ptr<std::streambuf> buffer() { return m_buffer; }

    /** Performs a query on the specified ResourceId */
    static std::shared_ptr<QueryResult> query(
        ResourceId const& identifier, 
        OptionSet  const& options = OptionSet());

    /** Issues a delete request for the specified ResourceId */
    static void remove(ResourceId const& identifier, OptionSet const& options = OptionSet());

    /** 
     * Registers a new abstract istream open module with the resource interface
     *
     * @param opener  The new opener to be registered for use with the interface
     * @param matcher The regular expression to use for matching with a resource
     */
    static void registerModule(String const& scheme, ResourceModuleH module);

    /** Removes the resource module associated with the specified scheme */
    static void removeModule(String const& scheme)
    {
        m_modules.erase(scheme);
    }

    /** Removes all instances of the specified resource module */
    static void removeModule(ResourceModuleH module);

	/** Returns a reference to the list containing the active resource openers. */
	inline static std::map<String,ResourceModuleH> const& modules() throw() 
    { 
        return m_modules; 
    }

    /** Returns the application level base URI */
    inline static ResourceId const& appBaseIdentifier()
    {
        return m_globalBaseUri;
    }

    /** Sets the application level base URI */
    inline void setAppBaseIdentifier(ResourceId const& identifier)
    {
        m_globalBaseUri = identifier;
    }

protected:
    Resource() : std::ios(0), m_setMask(0), m_openMode(0) { };

    unsigned int m_setMask;   ///< Forced open mode flags

    std::shared_ptr<std::streambuf> m_buffer; ///< Associated streambuffer

private:
    ResourceId   m_identifier;  ///< Resource identifier
    String       m_mimeType;    ///< Returned content-type
    unsigned int m_openMode;    ///< Stream mode settings
    
    static std::map<String,ResourceModuleH> m_modules; ///< Resource modules

    static ResourceId m_globalBaseUri;  ///< Global base URI (http://tools.ietf.org/html/rfc3986#section-5.1.4)
};

/**
 * Resource istream used for abstract IO 
 *
 * A ResourceIStream inherits from std::istream and provides access to the streambuffer returned
 * the internal resource loader which matches the resource. The resource stream wraps the streambuf
 * returned by the loader function to allow internal usage of the resource identifier in issuing
 * warning to the logger and identifying base URIs for relative references within the document. 
 */
class VOX_EXPORT ResourceIStream : virtual public std::istream, virtual public Resource
{
public:
    /** Default constructor - Initialize null streambuf */
    ResourceIStream() : std::istream(0) { m_setMask |= Mode_Input; }

    /** Initialization constructor */
    ResourceIStream(ResourceId const& identifier, 
                    OptionSet const&  options     = OptionSet(),
                    unsigned int      openMode    = Mode_Input
                    ) 
        : std::istream(0)
    { 
        m_setMask |= Mode_Input; 
        
        open(identifier, options, openMode); 
    }
    
    /** Initialization constructor */
    ResourceIStream(ResourceId const& identifier, unsigned int openMode) 
        : std::istream(0)
    { 
        m_setMask |= Mode_Input; 
        
        open(identifier, openMode); 
    }

    /**
     * Returns the expected number of input bytes available from the source stream
     *
     * @return The result of the owned streambuf's in_avail() method
     */
    inline std::streamsize available() const { return m_buffer->in_avail(); }

    /** Returns the remaining length of the stream if seekable */
/*
    inline std::streamsize remaining() const
    {
        std::streamsize pos = tellg();
        seekg(std::ios::end);
        std::streamsize end = tellg();
        seekg(pos);

        return end-pos;
    }
*/
};

/**
 * Resource ostream used for abstract IO 
 *
 * A ResourceOStream inherits from std::ostream and provides access to the streambuffer returned
 * the internal resource loader which matches the resource. The resource stream wraps the streambuf
 * returned by the loader function to allow internal usage of the resource identifier in issuing
 * warning to the logger and identifying base URIs for relative references within the document.
 *
 * By default, a ResourceOStream will send a Mode_Truncate request on open attempts. If this
 * behavior is undesirable, a Mode_Append should be specified for the request.
 */
class VOX_EXPORT ResourceOStream : virtual public std::ostream, virtual public Resource
{
public:
    /** Default constructor - Initialize null streambuf */
    ResourceOStream() : std::ostream(0) { m_setMask |= Mode_Truncate; }

    /** Initialization constructor */
    ResourceOStream(ResourceId const& identifier, 
                    OptionSet const&  options     = OptionSet(),
                    unsigned int      openMode    = Mode_Truncate
                    ) 
        : std::ostream(0)
    { 
        m_setMask |= Mode_Truncate; 
        
        open(identifier, options, openMode); 
    }

    /** Initialization constructor */
    ResourceOStream(ResourceId const& identifier, unsigned int openMode) 
        : std::ostream(0)
    { 
        m_setMask |= Mode_Truncate; 
        
        open(identifier, openMode); 
    }
};

/**
 * Resource iostream used for abstract IO 
 *
 * A ResourceStream inherits from std::iostream and provides access to the streambuffer returned
 * the internal resource loader which matches the resource. The resource stream wraps the streambuf
 * returned by the loader function to allow internal usage of the resource identifier in issuing
 * warning to the logger and identifying base URIs for relative references within the document. 
 */
class VOX_EXPORT ResourceStream : public std::iostream, public ResourceIStream, public ResourceOStream
{
public:
    /** Default constructor - Initialize null streambuf */
    ResourceStream() : std::iostream(0), std::istream(0), std::ostream(0) { m_setMask = Mode_Output | Mode_Input; }

    /** Initialization constructor */
    ResourceStream(ResourceId const& identifier, 
                   OptionSet const&  options     = OptionSet(),
                   unsigned int      openMode    = Mode_Output | Mode_Input
                   ) 
        : std::iostream(0), std::istream(0), std::ostream(0)
    {
        m_setMask = Mode_Output | Mode_Input;

        open(identifier, options, openMode); 
    }

    /** Initialization constructor */
    ResourceStream(ResourceId const& identifier, unsigned int openMode) 
        : std::iostream(0), std::istream(0), std::ostream(0)
    { 
        m_setMask = Mode_Output | Mode_Input;

        open(identifier, openMode); 
    }
};

}

// End definition
#endif // VOX_RESOURCE_H