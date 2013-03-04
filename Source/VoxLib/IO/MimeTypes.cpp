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

// API Namespace
namespace vox {

// Static member variable initialization
std::map<String, String>            MimeTypes::m_types;
std::map<String, std::list<String>> MimeTypes::m_suffixes;

// --------------------------------------------------------------------
//  Reads in mime-types information from the specified resource
// --------------------------------------------------------------------
void MimeTypes::readMimeTypes(IStream & input)
{
}

// --------------------------------------------------------------------
//  Reads in mime-types information from the specified resource
// --------------------------------------------------------------------
void MimeTypes::readMimeTypes(ResourceId const& identifier, OptionSet const& options)
{
    ResourceIStream istream(identifier, options);

    readMimeTypes(istream);
}

} // namespace vox