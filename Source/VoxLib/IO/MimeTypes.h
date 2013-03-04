/* ===========================================================================

	Project: Uniform Resource IO 

	Description: Mime-type / file extension conversion interface 

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
#ifndef VOX_MIME_TYPES_H
#define VOX_MIME_TYPES_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Types.h"
#include "VoxLib/IO/OptionSet.h"

// API namespace
namespace vox
{

struct ResourceId;
    
/**
 * Mime-Types Conversion Interface
 *
 * This class provides an interface for managing and converting Mime-Type information
 * as outline in RFC 2046 (http://tools.ietf.org/html/rfc2046) and documented by the
 * IANA at http://www.iana.org/assignments/media-types. In order to avoid internal
 * confusion regarding the conversion between mime-types and file or application
 * assignments, all mime-type associations must be registered by the user.
 */
class VOX_EXPORT MimeTypes
{
public:
    /** 
     * Reads in a mime.types format list of mime-types 
     *
     * Attempts to parse a mime.types format file of mime type mappings
     * and adds them to the internal map. The format consists of a 
     * Mime-Type followed by a white space delimited list of file
     * extensions on each line. 
     * 
     * Example:
     *
     * BEGIN FILE
     * application/mime-type1 .ext1 .ext2 .ext3
     * application/mime-type2 .ext4 .ext5 .ext6
     * END OF FILE
     */
    VOX_HOST static void readMimeTypes(IStream & input);

    /** String based overload for readMimeTypes */
    VOX_HOST static void readMimeTypes(String const& data)
    {
        readMimeTypes( IStringStream(data) );
    }

    /** URI reference based overload for readMimeTypes */
    VOX_HOST static void readMimeTypes(ResourceId const& identifier, 
                                       OptionSet const& options = OptionSet());

    /**
     * Registers a new extension to mime-type association
     *
     * @param type       The mime-type (type/sub-type)
     * @param extenstion The associated file extension
     */
    VOX_HOST static void addExtension(String const& extension, String const& type)
    {
        m_types.insert( std::make_pair(extension, type) );
    }

    /** Returns an extension associated with the mime-type */
    VOX_HOST static String const& guessExtension(String const& type, bool resolveAlias);

    /** Returns the list of extensions associated with the mime-type */
    VOX_HOST static std::vector<String const&> getAllExtensions(String const& type);
    
    /** Returns the mime-type associated with this extension */
    VOX_HOST static String const& getType(String const& extension)
    {
        return m_types[extension];
    }

    /** Verifies the mime-type and extension are a matched pair */
    VOX_HOST static bool equal(String const& extension, String const& type);

    /** Returns the extension of which this extension is an alias, or the extension */
    VOX_HOST static String const& getExtensionFromAlias(String const& extension);

    /** Adds an alias name for a given extension */
    VOX_HOST static void addExtensionAlias(String const& alias, String const& extension);

private:
    static std::map<String, String>            m_types;    ///< extension to Mime-Type
    static std::map<String, std::list<String>> m_suffixes; ///< extension to alias
};

}

// End definition
#endif // VOX_MIME_TYPES_H