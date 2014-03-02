/* ===========================================================================

    Project: SocketIO                                                       
                                                                           
    Description: Provides an IO module for low level socket IO              
                                                                           
    Copyright (C) 2014 Lucas Sherman                                        
                                                                           
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
#include "UdpSocketIO.h"

// API dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Core/Format.h"
#include "VoxLib/Error/FileError.h"
#include "VoxLib/IO/Resource.h"
#include "VoxLib/IO/ResourceId.h"

// Boost socket dependencies

// API namespace
namespace vox
{

namespace {
namespace filescope {

} // namespace filescope
} // namespace anonymous

// --------------------------------------------------------------------
//  Returns a filebuf with the specified mode settings
// --------------------------------------------------------------------
std::shared_ptr<std::streambuf> UdpSocketIO::access(
    ResourceId &     identifier, 
    OptionSet const& options,
    unsigned int     openMode
    )
{
    return nullptr;
}

// --------------------------------------------------------------------
//  Performs a query of the specified file or directory
// --------------------------------------------------------------------
std::shared_ptr<QueryResult> UdpSocketIO::query(
    ResourceId const& identifier, 
    OptionSet const&  options
    )
{
    auto result = std::make_shared<QueryResult>();
    
    return result;
}

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
void UdpSocketIO::remove(
    ResourceId const& identifier, 
    OptionSet const&  options
    )
{    
}

} // namespace vox