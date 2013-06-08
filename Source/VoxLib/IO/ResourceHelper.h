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

// :TODO: Add parameter for compression/decompression mode option

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
     * Provides a configuration method for VoxLib
     *
     * This function will load an XML format (presumably .config extension file) for an
     * application and parse specific attributes to configure elements of the system.
     * The only elements used by the loader are those located within the "Settings"
     * attribute, leaving the potential for users to define additional, application 
     * specific elements.
     *
     * Elements in this version:
     * 
     * Logging: This element contains child elements which modify the VoxLib logging system
     *
     *   ExcludeModules: This elements child elements will be parsed and all child elements
     *                   "Module" have their text attributes added to the list of modules to
     *                   ignore when issuing log entries to the application's logging backend.
     *
     *   Filter: This element will be parsed into a vox::Severity enum type and the filtering
     *           level for the Logger will be set to that severity.
     *
     *   LogFile: Will setup a logger backend which dispatches requests to the ResourceId parsed
     *            in the text attribute of this element. If the backend cannot be established, the
     *            logger will dispatch an error with the special code Error_:TODO:.
     *
     * Plugins: This element contains child elements which modify the VoxLib plugin management system 
     *
     *   SearchDirectories: The directories here will be added to the list of directories searched
     *                      for plugins when attempting to load or find available plugins
     *
     *   LoadWithoutInfo: If set to true, plugins without an associated .pin file will be loaded
     *                    to acquire their vendor, name, etc, information. Otherwise only plugins
     *                    which have a pin file denoting these attributes will be loaded. This
     *                    affects the plugins returned by the PluginManager when searching for
     *                    available plugins.
     *
     *   Authorized: Plugins which are deemed authorized to be immediately loaded and enabled.
     *               This will happen before this function returns. Authorized plugins take the
     *               form of <Plugin> child elements with text attributes of the plugin name dot 
     *               appended with a vendor. (Ex vox.standard_io for the standard_io plugin which
     *               returns a name of 'standard_io' and a vendor of 'vox')
     *
     * @sa vox::Logger vox::PluginManager
     */
    static void loadConfigFile(String const& configFile);

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