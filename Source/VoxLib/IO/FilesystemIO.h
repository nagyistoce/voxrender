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

// Begin definition
#ifndef VOX_FILESYSTEM_IO_H
#define VOX_FILESYSTEM_IO_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/IO/Resource.h"
#include "VoxLib/IO/ResourceModule.h"

// API namespace
namespace vox
{

/**
 * Filesystem IO interface
 *
 * Provides a resource retrieval module for accessing files using the 'file' 
 * protocol as outline in RFC-1738 http://www.rfc-editor.org/rfc/rfc1738.txt
 */
class VOX_EXPORT FilesystemIO : public ResourceModule
{
public:
    FilesystemIO() { }

    /** FilesystemIO factory function */
    static std::shared_ptr<FilesystemIO> create() 
    {
        return std::make_shared<FilesystemIO>();
    }

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
#endif // VOX_FILESYSTEM_IO_H