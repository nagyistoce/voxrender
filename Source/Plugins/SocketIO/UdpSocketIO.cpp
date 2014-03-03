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
#include <boost/asio.hpp>

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
    using namespace boost::asio::ip;

    typedef boost::asio::basic_socket_streambuf<boost::asio::ip::udp> UdpStreambuf;
    
    auto streambuf = std::shared_ptr<UdpStreambuf>(new UdpStreambuf());

    if (identifier.authority.empty() && !(openMode & Resource::Mode_Output))
    {
        udp::endpoint endpoint(udp::v4(), identifier.extractPortNumber());
        streambuf->open(udp::v4());
        streambuf->bind(endpoint);
    }
    else
    {
        auto host = address::from_string(identifier.authority);
        udp::endpoint endpoint(host, identifier.extractPortNumber());
        streambuf->connect(endpoint);
    }

    return streambuf;
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