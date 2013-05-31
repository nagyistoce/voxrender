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

// Include Header
#include "MimeTypes.h"

// Include Dependencies
#include "VoxLib/IO/Resource.h"

// 3rd Party Dependencies
#include <boost/thread.hpp>

// API Namespace
namespace vox {

namespace {
namespace filescope {

    std::map<String, String>            types;
    std::map<String, std::list<String>> suffixes;
    boost::shared_mutex                 mutex;

}
}

// --------------------------------------------------------------------
//  Reads in mime-types information from the specified resource
// --------------------------------------------------------------------
void MimeTypes::readMimeTypes(IStream & input)
{
    // Acquire a read-lock on the modules for thread safety support
    boost::unique_lock<decltype(filescope::mutex)> lock(filescope::mutex);
}

// --------------------------------------------------------------------
//  Reads in mime-types information from the specified resource
// --------------------------------------------------------------------
void MimeTypes::readMimeTypes(ResourceId const& identifier, OptionSet const& options)
{
    ResourceIStream istream(identifier, options);

    readMimeTypes(istream);
}

// --------------------------------------------------------------------
//  Adds a new association between an extension and a mime type
// --------------------------------------------------------------------
void MimeTypes::addExtension(String const& extension, String const& type)
{
    // Acquire a read-lock on the modules for thread safety support
    boost::unique_lock<decltype(filescope::mutex)> lock(filescope::mutex);

    filescope::types.insert( std::make_pair(extension, type) );
}

// --------------------------------------------------------------------
//  Returns the type associated with an extension 
// --------------------------------------------------------------------
String const& MimeTypes::getType(String const& extension)
{
    // Acquire a read-lock on the modules for thread safety support
    boost::unique_lock<decltype(filescope::mutex)> lock(filescope::mutex);

    return filescope::types[extension];
}

} // namespace vox