/* ===========================================================================

    Project: StandardIO - Module definition for exported interface

    Description: A libcurl wrapper compatible with the VoxIO library

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
#ifndef SIO_MODULE_H
#define SIO_MODULE_H

// Include Dependencies
#include "Common.h"
#include "VoxLib/IO/Resource.h"
#include "VoxLib/IO/ResourceModule.h"

/**
 * LibCurl IO interface
 *
 * Provides a resource retrieval module for accessing files using curl
 */
class Module : public vox::ResourceModule
{
public:
    Module() { }

    virtual std::shared_ptr<std::streambuf> access(
        vox::ResourceId &     identifier, 
        vox::OptionSet const& options,
        unsigned int &   openMode);

    virtual void remove(
        vox::ResourceId const& identifier, 
        vox::OptionSet  const& options );

    virtual vox::QueryResult query(
        vox::ResourceId const& identifier, 
        vox::OptionSet  const& options);
};

// End definition
#endif // SIO_MODULE_H
