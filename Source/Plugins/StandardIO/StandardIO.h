/* ===========================================================================

	Project: VoxLib IO Library Plugin
    
	Description: Standard IO Plugin for VoxLib IO library

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

// Begin definition
#ifndef SIO_STANDARD_IO_H
#define SIO_STANDARD_IO_H

// Include Dependencies
#include "Plugins/StandardIO/Common.h"
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/IO/Resource.h"
#include "VoxLib/IO/ResourceModule.h"

// API namespace
namespace vox
{

/**
 * Standard IO interface
 *
 * Provides a resource retrieval module for accessing files using a variety of common protocols.
 *
 * <b>Supported Protocols<b>
 *
 * HTTP(S):  [access remove query]
 * FTP(S):   [access remove query]
 * DICT:     [access remove query]
 * LDAP(S):  [access remove query]
 * IMAP(S):  [access remove query]
 * POP3(S):  [access remove query]
 * SMTP(S):  [access remove query]
 * FILE:     [access remove query]
 * GOPHER:   [access remove query]
 * TELNET:   [access remove query]
 * TFTP:     [access remove query]
 * SFTP:     [access remove query]
 * RTMP:     [access remove query]
 * RTSP:     [access remove query]
 * SCP:      [access remove query]
 *
 * <b>Supported Options<b>
 *
 * UserAgent:       string                          [default='voxlib-agent/' SIO_VERSION_STRING]
 * ConnectTimeout:  unsigned int (seconds)          [default='300']
 * MinimumTransfer: unsigned int (bytes / second)   [default='300']
 * MaxRedirects:    unsigned int                    [default='5']
 */
class SIO_EXPORT StandardIO : public ResourceModule
{
public:
    /**
     * Provides resource access capabilities 
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
     * Removes a resource
     *
     * @param identifier The resource identifier
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
#endif // SIO_STANDARD_IO_H