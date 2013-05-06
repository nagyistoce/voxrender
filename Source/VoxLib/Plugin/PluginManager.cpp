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

// Include Header
#include "PluginManager.h"

// Include Dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Core/Format.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/Plugin/Plugin.h"
#include "VoxLib/Plugin/PluginContainer.h"

// Boost filesystem library
#include <boost/filesystem.hpp>

// API namespace
namespace vox {

    // Implementation of private members for PluginManager
    struct PluginManager::Impl 
    {
        std::list<boost::filesystem::path> searchPaths; // Directories + Files to search

        std::list<PluginContainer> loadedPlugins;

        boost::mutex mutex;

        std::unique_ptr<boost::thread> runtimeDetectionThread;
    };
    
    // --------------------------------------------------------------------
    //  Instantiates the pImpl member structure for the plugin manager
    // --------------------------------------------------------------------
    PluginManager::PluginManager() { m_pImpl = new Impl(); }
    
    // --------------------------------------------------------------------
    //  Deletes the pImple member structure and unloads all plugins
    // --------------------------------------------------------------------
    PluginManager::~PluginManager() { unloadAll(); delete m_pImpl; }

    // --------------------------------------------------------------------
    //  Adds a path to the search paths for the program
    // --------------------------------------------------------------------
    void PluginManager::addPath(String const& string)
    {
        boost::mutex::scoped_lock lock(m_pImpl->mutex);

        boost::filesystem::path path(string);

        path.normalize();

        m_pImpl->searchPaths.push_back(path);
    }
    
    // --------------------------------------------------------------------
    //  Removes a path from the search paths for the program
    // --------------------------------------------------------------------
    void PluginManager::removePath(String const& string)
    {
        boost::mutex::scoped_lock lock(m_pImpl->mutex);

        boost::filesystem::path path(string);

        path.normalize();

        m_pImpl->searchPaths.remove(path);
    }   
    
    // --------------------------------------------------------------------
    //  Returns an info structure describing a loaded plugin
    // --------------------------------------------------------------------
    PluginInfo PluginManager::getPluginInfo(String const& pluginName)
    {
        PluginInfo result; 
        
        boost::mutex::scoped_lock lock(m_pImpl->mutex);
        
        BOOST_FOREACH(auto & plugin, m_pImpl->loadedPlugins)
        {
            if (plugin.name() == pluginName)
            {
                result.name          = plugin.name();
                result.vendor        = plugin.vendor();
                result.version       = plugin.version();
                result.apiVersionMax = plugin.apiVersionMax();
                result.apiVersionMin = plugin.apiVersionMin();
                result.url           = plugin.referenceUrl();
                result.path          = plugin.path();
            }

            return result;
        }

        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                    format("Requested info for missing plugin: %1%", pluginName));
    }
    
    // --------------------------------------------------------------------
    //  Searches for any plugins in the set of search directories and
    //  attempts to load and optionally enable them.
    // --------------------------------------------------------------------
    void PluginManager::loadAll(bool enable)
    {
        BOOST_FOREACH(auto & searchPath, m_pImpl->searchPaths)
        {
        }
    }

    // --------------------------------------------------------------------
    //  Unloads any plugins loaded by the PluginManager (nothrow)
    // --------------------------------------------------------------------
    void PluginManager::unloadAll()
    {
        boost::mutex::scoped_lock lock(m_pImpl->mutex);

        m_pImpl->loadedPlugins.clear();
    }

    // --------------------------------------------------------------------
    //  Loads a plugin by name or path. If an existing plugin is loaded
    //  with a less appropriate version it will automatically be replaced.
    // --------------------------------------------------------------------
    void PluginManager::load(String const& name, bool enable)
    {
    }

    // --------------------------------------------------------------------
    //  Unloads a loaded plugin if it is currently loaded
    // --------------------------------------------------------------------
    void PluginManager::unload(String const& pluginName)
    {
        boost::mutex::scoped_lock lock(m_pImpl->mutex);

        for (auto iter = m_pImpl->loadedPlugins.begin(); 
                  iter != m_pImpl->loadedPlugins.end(); ++iter)
        {
            if (pluginName == iter->name())
            {
                m_pImpl->loadedPlugins.erase(iter); return;
            }
        }
    }

    // --------------------------------------------------------------------
    //  Performs a soft reset of an enabled plugin
    // --------------------------------------------------------------------
    void PluginManager::softReload(String const& pluginName)
    {
        boost::mutex::scoped_lock lock(m_pImpl->mutex);

        BOOST_FOREACH(auto & plugin, m_pImpl->loadedPlugins)
        {
            if (plugin.name() == pluginName)
            {
                plugin.softReload(); return;
            }
        }

        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                    format("Requested soft reload for unloaded plugin: %1%", pluginName));
    }

    // --------------------------------------------------------------------
    //  Performs a hard reset of an enabled plugin - load and unload dll
    // --------------------------------------------------------------------
    void PluginManager::reload(String const& pluginName)
    {
        boost::mutex::scoped_lock lock(m_pImpl->mutex);

        BOOST_FOREACH(auto & plugin, m_pImpl->loadedPlugins)
        {
            if (plugin.name() == pluginName)
            {
                plugin.reload(); return;
            }
        }

        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                    format("Requested reload for unloaded plugin: %1%", pluginName));
    }

} // namespace vox