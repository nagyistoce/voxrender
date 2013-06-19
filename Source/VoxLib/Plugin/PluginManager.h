/* ===========================================================================

	Project: VoxRender - Plugin management class

	Description: Provides management for loading runtime plugins

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

/** 
 * @file PluginManager.h
 *
 * <b>Overview</b>
 *
 * The PluginManager class, in association with the Plugin class, provides 
 * basic functionality for loading and running C/C++ plugins at runtime in
 * a managed fashion.
 *
 * Although the plugin class allows loading and interaction with any shared
 * library, the PluginManager has stricter requirements concerning the
 * interface of the plugin library. Specifically, the following functions
 * must be provided for the PluginManager to consider a plugin valid:
 *
 * void initPlugin()             <<< Called immediately after plugin loading
 * void freePlugin()             <<< Called before the plugin is unloaded
 *  
 * void enable()                 <<< Called on user permission to activate
 * void disable()                <<< Called on user shutdown request 
 *
 * char const* name()            <<< Returns the name of the plugin
 * char const* vendor()          <<< Returns the plugins creating organization
 * char const* apiVersionMin()   <<< Returns the minimum supported API version
 * char const* apiVersionMax()   <<< Returns the maximum supported API version
 * char const* version()         <<< Returns the version of the plugin
 *
 * bool canFree()                <<< Return true if the plugin is safe to free
 *
 * <b>Reference URL</b>
 *
 * A method of the format
 *
 * char const* referenceUrl() <<< Called if available to get a reference URL for users
 *
 * may optionally be provided to allow an application to provide users with
 * additional information on a plugin.
 *
 * <b>Version Info</b>
 *
 * Plugin and API version info is specified to the PluginManager by a plugin
 * using dot delimited ANSI string. (Ex "12.123.123") A malformed version string
 * will cause the PluginManager to immediately unload the plugin. Tags following
 * the plugin version and seperated by a space character will be ignored.
 * (Ex "12.123.123 (dev)")
 *
 * The API version flags are of the format "major.minor.patch". Minor changes
 * are guaranteed to be only changes which extend the API interface and do not
 * break ABI compatibility with older API versions. Major changes will typically
 * break compatibility with previous API versions. Patches are generally small
 * changes designed to fix errors detected in a specific "major.minor" version.
 *
 * <b>Plugin Selection</b>
 *
 * The return value of the function name() uniquely identifies a single plugin.
 * (The return values should be null terminated ANSI strings) To ensure there 
 * are no conflicts with other vendor's plugins, it may be appropriate to append
 * the plugin name with the vendor tag. If multiple plugins are found which provide 
 * identical name strings, then the PluginManager will load the one with the highest 
 * version number which returns a minApiVersion below or equal to the current API
 * version.
 */

// Begin definition
#ifndef VOX_PLUGIN_MANAGER_H
#define VOX_PLUGIN_MANAGER_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Types.h"
#include "VoxLib/Plugin/PluginInfo.h"

// API namespace
namespace vox
{
	/** 
	 * Plugin Management class
     *
     * This class manages the loading of runtime plugins which conform to the plugin interface
     * as described above. In order to properly support usage of the Plugin Manager by plugins
     * themselves for purposes of automatic dependency loading, all member functions are thread
     * safe and reentrant. In addition, access to the class is restricted to a singleton object
     * which manages all plugins and ensures they are properly unloaded on application closure.
	 */
    class VOX_EXPORT PluginManager
	{
    public:
        typedef std::function<void(std::shared_ptr<PluginInfo>)> DiscoveryCallback;

        /** Returns the global PluginManager */
        static PluginManager & instance();

        /**
         * Disables and unloads active plugins
         *
         * The user should consider explicitly unloading plugins before initiating shutdown of their 
         * application. This will avoid any issues with plugins having to detect shutdown of application 
         * components while they are still enabled.
         */
        ~PluginManager();

        /** 
         * Adds a path to the set of plugin search directories
         *
         * This function may provide either a file or directory. if the parameter is a file, only 
         * the specific file will potentially be added to the list of available plugins. Otherwise
         * the directory will be searched on calls to load for any files fitting the operating 
         * systems shared library convention (.DLL on windows .so on Linux). UNC/network paths
         * are acceptable as long as the fully qualified path is specified.
         *
         * Any relative paths will be evaluated with the current filesystem directory as a base. 
         * As a result, calls to removePath() with the same path are not guaranteed to remove the 
         * path added by a previous call to this function.
         *
         * @param path The new filesystem directory or file to be added to the search paths
         */
        void addPath(String const& path);

        /**
         * Removes a path from the set of plugin search directories
         *
         * This function may provide either a file or directory. Relative paths will be 
         * evaluated using the applications current directory as a base path. Paths need
         * not be canonicalized.
         *
         * @param path The filesystem directory or file to be removed from the search paths
         */
        void removePath(String const& path);

        /** 
         * Loads a specified plugin from a file
         *
         * This function will attempt to load a specified plugin from a filesystem path. If the plugin
         * is already loaded, a warning will be logged and the PluginInfo structure will be returned.
         *
         * @param name   A plugin name or filesystem path identifying the plugin
         * @param enable If true, the plugin is enabled after load
         *
         * @returns A handle to the plugin information
         */
        std::shared_ptr<PluginInfo> loadFromFile(String const& file, bool enable = true);

        /** Attempts to load a plugin */
        void load(std::shared_ptr<PluginInfo> const& info);

        /** Unloads a plugin if it has been loaded */
        void unload(std::shared_ptr<PluginInfo> const& info);

        /** Returns true if the given plugin is currently active */
        bool isLoaded(std::shared_ptr<PluginInfo> const& info);

        /** 
         * Unloads and then enables a plugin 
         *
         * If a handle to the plugin dll is held elsewhere, then the
         * operation will silently default to a softReload().
         */
        void reload(String const& pluginName);

        /** Disables and then enables a plugin */
        void softReload(String const& pluginName);

        /**
         * Loads all available plugins
         *
         * This function causes the plugin manager to find all plugins in the available 
         * search directories. This does not include multiple versions of the same plugin.
         *
         * @param load      If true, any detected plugins will automatically be loaded 
         * @param checkBins If true, binaries with plugin extensions (dll, so, etc) and no associated .pin will be loaded, 
         *                  for a short time, to extract plugin info. Otherwise only .pin associated files are loaded.
         */
        void findAll(DiscoveryCallback callback, bool load = false, bool checkBins = false);

        /** Explicitly unloads any loaded plugins */
        void unloadAll();

        /** Returns an info structure for a given plugin */
        std::shared_ptr<PluginInfo> getPluginInfo(String const& name);

        /** 
         * Enables automatic runtime detection and loading of plugins
         *
         * This function will initiate a continuous management thread
         * which will occasionally poll the plugin directories for new 
         * plugins and automatically load and optionally enable them.
         * If the set of plugin search directories is large, this process
         * may be expensive.
         *
         * @param pluginOptions A bitset of PluginOption flags for detected plugins
         */
        void enableRuntimeDetection(unsigned int pluginOptions);

        /**
         * If runtime detection of plugins was previously enabled, it
         * will be disabled.
         */
        void disableRuntimeDetection();

    private:
        PluginManager();

        PluginManager(PluginManager const&) { }

        struct Impl; Impl * m_pImpl;
	};
}

// End definition
#endif // VOX_PLUGIN_MANAGER_H