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

// :TODO: Add parameter for compression mode option

// Begin definition
#ifndef VOX_RESOURCE_HELPER_H
#define VOX_RESOURCE_HELPER_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/IO/ResourceId.h"

// API Namespace
namespace vox 
{

/**
 * Resource Helper Class
 *
 * This class provides some simple helper functions to perform common tasks without the
 * need for tedious code snippet duplication.
 */
class VOX_EXPORT ResourceHelper
{
public:
    /** 
     * Transfers the raw contents of a resource into a local buffer 
     *
     * Note: This function does not perform any formatting (line ending conversion, etc)
     */
    static std::string pull(ResourceId const& resource);
    
    /** 
     * Transfers the raw contents of a resource into a local buffer 
     *
     * Note: This function does not perform any formatting (line ending conversion, etc)
     */
    static void push(ResourceId const& resource, std::string const& data);

    /** Transfers a resource from one location to another unchanged */ 
    static void move(ResourceId const& source, ResourceId const& destination);
};

} // namespace vox

// End definition
#endif // VOX_RESOURCE_HELPER_H