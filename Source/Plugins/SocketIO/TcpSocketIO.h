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

// Begin definition
#ifndef VOX_TCP_SOCKET_IO_H
#define VOX_TCP_SOCKET_IO_H

// Include Dependencies
#include "Plugins/SocketIO/Common.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/IO/Resource.h"
#include "VoxLib/IO/ResourceModule.h"

// API namespace
namespace vox
{

/**
 * Tcp Socket IO interface
 *
 * Provides a resource retrieval module for udp connections
 */
class SOKIO_EXPORT TcpSocketIO : public ResourceModule
{
public:
    TcpSocketIO() { }

    /**
     * Provides filesystem access capabilities 
     *
     * @param identifier The resource identifier
     * @param options    Access options
     * @param openMode   The open mode options
     *
     * @returns A handle to the streambuffer associated with the file
     */
    virtual std::shared_ptr<std::streambuf> access(
        ResourceId &     identifier, 
        OptionSet const& options,
        unsigned int     openMode
        );

    /** 
     * Removes a filesystem resource
     *
     * @param identifier The resource identifier (can be a directory)
     * @param options    Remove options
     */
    virtual void remove(
        ResourceId const& identifier, 
        OptionSet const&  options
        );

    /**
     * Queries the specified resource 
     *
     * @param identifier The resource identifier
     * @param options    Query options
     *
     * @return The rdf format query response
     */
    virtual std::shared_ptr<QueryResult> query(
        ResourceId const& identifier, 
        OptionSet const&  options
        );
};

}

// End definition
#endif // VOX_TCP_SOCKET_IO_H