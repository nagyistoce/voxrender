/* ===========================================================================

	Project: Uniform Resource IO 
    
	Description: Implements a module for a specific type of IO 

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
#ifndef VOX_RESOURCE_MODULE_H
#define VOX_RESOURCE_MODULE_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/IO/ResourceId.h"

// Property tree declaration
#include "boost/property_tree/ptree_fwd.hpp"

// API Namespace
namespace vox
{

/** Resource Query Result Structure : PTree embedded RDF?:TODO: tree */
typedef boost::property_tree::basic_ptree<String,String,std::less<String>> QueryResult;

/** Convenience typdef for QueryResult handle */
typedef std::shared_ptr<QueryResult> QueryResultH;

/**
 * Resource module concept used for abstract IO
 *
 * The purpose of a ResourceModule is to implement the IO functionality for a specific
 * subset of URI requests. A ResourceModule implementation is registered with the Resource
 * class function registerModule.
 *
 * During registration with the Resource system, an the module is specified along with a 
 * corresponding scheme which denotes the subset of URI dereferencing requests the module 
 * will be called upon to handle. 
 *
 * Several aspects of the URI parsing are delegated to the resource module for a 
 * given request and the modules are recommended to follow the RFC guidelines referenced 
 * in 'ResourceId.h'. These include but may not be limited to:
 *
 *  o Decomposition, validation, and parsing of the hierarchical authority component
 *  o Validation of the query component 
 *  o Validation of the fragment identifier
 *  o Specification of the resolved URI
 */
class VOX_EXPORT ResourceModule
{
public:
    virtual ~ResourceModule() {}

    /**
     * Resource access
     *
     * The accessor method provides a streambuf associated with the resource identifer. 
     * If the resource cannot be opened in a manner compliant with the specifications of the
     * openMode, then this method should fail. This method should also modify the input
     * resource identifier to correspond with the resolved URI of the request.
     *
     * @sa Resource::Mode
     */
    virtual std::shared_ptr<std::streambuf> access(
        ResourceId &     identifier, ///< The resource identifier
        OptionSet const& options,    ///< The advanced access options
        unsigned int     openMode    ///< The desired open mode
    ) = 0;

    /**
     * Resource query 
     *
     * Queries and returns information concerning the specified URI. The return tree should conform
     * to the rdf+xml encoding standard. (Available at http://www.w3.org/TR/REC-rdf-syntax/)
     */
    virtual QueryResultH query(
        ResourceId const& identifier, ///< The resource identifier
        OptionSet const&  options     ///< The advanced query options
    ) = 0;

    /**
     * Resource removal
     *
     * A resource remover performs a delete or remove operation on a ResourceId. 
     * If the removal request could not be completed successfully, this function
     * should fail.
     */
    virtual void remove(
        ResourceId const& identifier, ///< The resource identifier
        OptionSet const&  options     ///< The advanced removal options
    ) = 0;
};

}

// End Definition
#endif // VOX_RESOURCE_MODULE_H