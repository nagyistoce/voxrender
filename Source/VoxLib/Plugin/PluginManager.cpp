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
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/Plugin/Plugin.h"
#include "VoxLib/Plugin/PluginContainer.h"

// Boost filesystem library
#include <boost/filesystem.hpp>
#include <boost/property_tree/xml_parser.hpp>

// API namespace
namespace vox {

// Implementation of private members for PluginManager
struct PluginManager::Impl 
{
    std::list<boost::filesystem::path> searchPaths; // Directories or files to search

    std::list<std::shared_ptr<PluginContainer>> plugins; // List of detected/loaded plugins

    boost::mutex mutex;

    std::unique_ptr<boost::thread> runtimeDetectionThread;
};
    

// --------------------------------------------------------------------
//  Instantiates the pImpl member structure for the plugin manager
// --------------------------------------------------------------------
PluginManager & PluginManager::instance()
{ 
    static PluginManager pmanager;
    return pmanager;
}

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
std::shared_ptr<PluginInfo> PluginManager::getPluginInfo(String const& name)
{
    boost::mutex::scoped_lock lock(m_pImpl->mutex);
        
    BOOST_FOREACH(auto & plugin, m_pImpl->plugins)
    {
    }

    throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                format("Requested info for unknown plugin: %1%", name));
}
    
// --------------------------------------------------------------------
//  Searches for any plugins in the set of search directories and
//  optionally enables them.
// --------------------------------------------------------------------
void PluginManager::findAll(DiscoveryCallback callback, bool load, bool checkBins)
{
    BOOST_FOREACH(auto & searchPath, m_pImpl->searchPaths)
    {
        BOOST_FOREACH (auto & entry, boost::make_iterator_range(boost::filesystem::directory_iterator(searchPath), 
                                                                boost::filesystem::directory_iterator()))
        {
            // Check file extension for .pin or system's library extension
            if (entry.path().extension() == ".pin")
            {
                VOX_LOG_ERROR(Error_None, VOX_LOG_CATEGORY, ".pin file");
            }
            else if (entry.path().extension() == ".dll" && checkBins)
            {
                try
                {
                    auto info = loadFromFile(entry.path().string(), false);

                    callback(info);
                }
                catch (Error & error)
                {
                    VOX_LOG_WARNING(error.code, VOX_LOG_CATEGORY, format("Attempt to load <%1%> as plugin failed: %2%", entry.path().string(), error.message));
                }

                if (load) ;
            }
        }
    }
}

// --------------------------------------------------------------------
//  Unloads any plugins loaded by the PluginManager (nothrow)
// --------------------------------------------------------------------
void PluginManager::unloadAll()
{
    boost::mutex::scoped_lock lock(m_pImpl->mutex);

    BOOST_FOREACH(auto & plugin, m_pImpl->plugins) plugin->unload();
}

// --------------------------------------------------------------------
//  Loads a specified plugin. If the plugin info file cannot be found
//  and the permissions are not available, an exception will be thrown.
// --------------------------------------------------------------------
std::shared_ptr<PluginInfo> PluginManager::loadFromFile(String const& file, bool enable)
{
    boost::filesystem::path filePath(file);
    filePath.replace_extension("dll"); // :TODO: Platform specific system extension name

    filePath = boost::filesystem::absolute(filePath, boost::filesystem::current_path());

    // Verify the file can be found successfully
    if (boost::filesystem::exists(filePath))
    {
        auto plugin  = std::make_shared<PluginContainer>(filePath.string());
        auto newInfo = plugin->info();
        auto newId   = newInfo->vendor + "." + newInfo->name;

        // Verify that the same plugin is not already enabled
        BOOST_FOREACH(auto & plugin, m_pImpl->plugins)
        {
            auto info = plugin->info();
            
            if (newId == (info->vendor + "." + info->name))
            {
                // :TODO: m_pImpl->compare(newInfo, plugin->info())
            }
        }

        if (enable) plugin->enable();
        else plugin->unload();

        m_pImpl->plugins.push_back(plugin);

        return newInfo;
    }
    else
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
            format("Plugin file not found: %1%", file));
    }
}

// --------------------------------------------------------------------
//  Loads a specified plugin (ie enable or load-enable)
// --------------------------------------------------------------------
void PluginManager::load(std::shared_ptr<PluginInfo> const& info)
{
    boost::mutex::scoped_lock lock(m_pImpl->mutex);
    
    // Unload the plugin but keep the info structure for reference
    BOOST_FOREACH(auto & plugin, m_pImpl->plugins)
    {
        if (info == plugin->info()) { plugin->load(); plugin->enable(); return; }
    }
}
        
// --------------------------------------------------------------------
//  Returns true if the specified plugin is 'loaded' (ie enabled)
// --------------------------------------------------------------------
bool PluginManager::isLoaded(std::shared_ptr<PluginInfo> const& info)
{
    boost::mutex::scoped_lock lock(m_pImpl->mutex);
    
    // Unload the plugin but keep the info structure for reference
    BOOST_FOREACH(auto & plugin, m_pImpl->plugins)
    {
        if (info == plugin->info()) { return plugin->isEnabled(); }
    }

    return false;
}

// --------------------------------------------------------------------
//  Unloads a loaded plugin if it is currently loaded
// --------------------------------------------------------------------
void PluginManager::unload(std::shared_ptr<PluginInfo> const& info)
{
    boost::mutex::scoped_lock lock(m_pImpl->mutex);
    
    // Unload the plugin but keep the info structure for reference
    BOOST_FOREACH(auto & plugin, m_pImpl->plugins)
    {
        if (info == plugin->info()) { plugin->unload(); return; }
    }
}

// --------------------------------------------------------------------
//  Performs a soft reset of an enabled plugin
// --------------------------------------------------------------------
void PluginManager::softReload(String const& pluginName)
{
    boost::mutex::scoped_lock lock(m_pImpl->mutex);

    BOOST_FOREACH(auto & plugin, m_pImpl->plugins)
    {
        if (plugin->info()->name == pluginName)
        {
            plugin->softReload(); return;
        }
    }

    throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                format("Requested soft reload for unknown plugin: %1%", pluginName));
}

// --------------------------------------------------------------------
//  Performs a hard reset of an enabled plugin - load and unload dll
// --------------------------------------------------------------------
void PluginManager::reload(String const& name)
{
    boost::mutex::scoped_lock lock(m_pImpl->mutex);

    BOOST_FOREACH(auto & plugin, m_pImpl->plugins)
    {
        if (plugin->info()->name == name)
        {
            plugin->reload(); return;
        }
    }

    throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                format("Requested reload for unknown plugin: %1%", name));
}

} // namespace vox