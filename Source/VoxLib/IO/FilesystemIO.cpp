/* ===========================================================================

	Project: Uniform Resource IO 
    
	Description: Implements a file scheme resource module

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
#include "FilesystemIO.h"

// API dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Core/Format.h"
#include "VoxLib/Error/FileError.h"
#include "VoxLib/Error/PluginError.h"
#include "VoxLib/IO/Resource.h"
#include "VoxLib/IO/ResourceId.h"

// Boost filesystem dependencies
#include <boost/filesystem/fstream.hpp>
#include <boost/algorithm/string/case_conv.hpp>

// API namespace
namespace vox
{

namespace {
namespace filescope {

    // --------------------------------------------------------------------
    //  Returns a standard library compatible UNC or local filesystem path
    // --------------------------------------------------------------------
    String getFilename(ResourceId const& identifier)
    {
        if (identifier.path.empty()) 
        {
            throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, "No file was specified", Error_NotAllowed);
        }

        String authority = boost::algorithm::to_lower_copy(identifier.authority);
        if (authority.empty()) // Note: must access localhost specifications as UNC path
        {
            return (identifier.path[0] == '/') ? identifier.path.substr(1) : identifier.path;
        }
        else
        {
            String hostname = "//" + identifier.authority;
            if (identifier.path[0] == '/')
            {
                return hostname + identifier.path;
            }
            else
            {
                return hostname + '/' + identifier.path;
            }
        }
    }

} // namespace filescope
} // namespace anonymous

// --------------------------------------------------------------------
//  Returns a filebuf with the specified mode settings
// --------------------------------------------------------------------
std::shared_ptr<std::streambuf> FilesystemIO::access(
    ResourceId &     identifier, 
    OptionSet const& options,
    unsigned int     openMode
    )
{
    // Canonize the path to remove sym-links, etc
    boost::filesystem::path filepath = filescope::getFilename(identifier);

    filepath.normalize();

    String const filename = filepath.string();

    // Feed back resolved filename information
    if (boost::starts_with(filename, "//"))
    {
        size_t const hostLength = filename.find("/", 2);

        // :TODO:
        //identifier.authority = filename.substr(2, hostLength-2);
        //identifier.path      = filename.substr(hostLength);
    }
    else
    {
        //identifier.path = "/" + filename;
    }

    // Determine filebuffer mode settings
    std::ios::openmode which = std::ios::binary;
    if ((openMode & Resource::Mode_Input)      == Resource::Mode_Input)      which |= std::ios::in;
    if ((openMode & Resource::Mode_Output)     == Resource::Mode_Output)     which |= std::ios::out;
    if ((openMode & Resource::Mode_StartAtEnd) == Resource::Mode_StartAtEnd) which |= std::ios::ate;
    if ((openMode & Resource::Mode_Truncate)   == Resource::Mode_Truncate)   which |= std::ios::trunc;
    if ((openMode & Resource::Mode_Append)     == Resource::Mode_Append)     which |= std::ios::app;

    // Create a shared pointer to a new filebuffer
    auto buffer = std::make_shared<std::filebuf>();

    // Open the specified file with the determined mode settings
    if (!buffer->open(filename, which))
    {
        throw FileError(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                        format("Failed to open with mode %1%", which),
                        identifier.asString(), Error_NotFound);
    }

    return buffer;
}

// --------------------------------------------------------------------
//  Performs a query of the specified file or directory
// --------------------------------------------------------------------
std::shared_ptr<QueryResult> FilesystemIO::query(
    ResourceId const& identifier, 
    OptionSet const&  options
    )
{
    auto result = std::make_shared<QueryResult>();
    
    boost::filesystem::path filepath = filescope::getFilename(identifier);

    // :TODO:
    filepath.normalize();                         // resolved-uri
    boost::filesystem::file_size(filepath);       // size
    boost::filesystem::last_write_time(filepath); // last-modified
    boost::filesystem::is_symlink(filepath);      // is-symlink
    boost::filesystem::is_directory(filepath);    // is-directory
    boost::filesystem::extension(filepath);       // type (extension)

    return result;
}

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
void FilesystemIO::remove(
    ResourceId const& identifier, 
    OptionSet const&  options
    )
{    
    String filename = filescope::getFilename(identifier);

    if (!boost::filesystem::remove(filename))
    {
        throw FileError(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                        "Attempted to delete non-existent file",
                        filename, Error_NotFound);
    }
}

} // namespace vox