    /* ===========================================================================

	Project: Uniform Resource IO 
    
	Description: Provides extended functionality relating to the Resource class

    Copyright (C) 2013 Lucas Sherman

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
#include "ResourceHelper.h"

// Include Dependencies
#include "VoxLib/IO/Resource.h"

// API Namespace
namespace vox 
{

// ----------------------------------------------------------------------------
//  Utilizes the Resource library to copy URI request data into a buffer
// ----------------------------------------------------------------------------
std::string ResourceHelper::pull(ResourceId const& resource)
{
    ResourceIStream istr(resource);
    
    std::ostringstream outStr(std::ios::in|std::ios::binary);
    outStr << istr.rdbuf();

    return outStr.str();
}

// ----------------------------------------------------------------------------
//  Utilizes the Resource library to upload URI request data from a buffer
// ----------------------------------------------------------------------------
void ResourceHelper::push(ResourceId const& resource, std::string const& data)
{
    ResourceOStream ostr(resource);
    
    ostr << data;
}

// ----------------------------------------------------------------------------
//  Utilizes the Resource library to upload URI request data from a buffer
// ----------------------------------------------------------------------------
void ResourceHelper::move(ResourceId const& source, ResourceId const& destination)
{
    ResourceIStream istr(source);
    ResourceOStream ostr(destination);
    
    ostr << istr.rdbuf();
}

}