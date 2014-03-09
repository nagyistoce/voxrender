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
public:
    Impl() : m_complete(false), m_pluginThread(std::bind(&Impl::threadEntryPoint, this))
    {
    }

    ~Impl()
    {
        m_complete = true;

        m_noWorkCond.notify_one();

        m_pluginThread.join();
    }

    std::list<boost::filesystem::path> searchPaths; // Directories or files to search

    std::list<std::shared_ptr<PluginContainer>> plugins; // List of detected/loaded plugins

    boost::recursive_mutex mutex;
    
    std::shared_ptr<void> pluginHandle;

    // ----------------------------------------------------------------------------
    //  Constructs a handle for a plugin to manage its lifetime
    // ----------------------------------------------------------------------------
    void buildPluginHandle(std::shared_ptr<PluginContainer> container)
    {
        pluginHandle = std::shared_ptr<void>(container.get(), boost::bind(&Impl::unload, this, _1));
    }
    
    // ----------------------------------------------------------------------------
    //  Locates the most recent version of a plugin identified by its vendor, name pairing
    // ----------------------------------------------------------------------------
    std::shared_ptr<PluginInfo> getByVendorName(String const& vendor, String const& name)
    {
        boost::recursive_mutex::scoped_lock lock(mutex);
    
        std::shared_ptr<PluginInfo> result = nullptr;
        BOOST_FOREACH (auto & plugin, plugins)
        {
            auto info = plugin->info();
            if (info->name == name && info->vendor == vendor)
            if (!result || result->version < info->version)
            {
                result = info;
            }
        }

        return result;
    }

private: 

    std::list<std::shared_ptr<PluginContainer>> m_loadQueue;
    std::list<std::shared_ptr<PluginContainer>> m_unloadQueue;
    
    boost::condition_variable_any m_noWorkCond;
    bool                          m_complete;
    boost::thread                 m_pluginThread;

    // ----------------------------------------------------------------------------
    //  Instantiates the pImpl member structure for the plugin manager
    // ----------------------------------------------------------------------------
    void unload(void* plugin)
    {
        boost::recursive_mutex::scoped_lock lock(mutex);

        BOOST_FOREACH (auto & container, plugins)
        {
            if (container.get() == plugin) { m_unloadQueue.push_back(container); break; }
        }

        lock.unlock();

        m_noWorkCond.notify_one();
    }

    // ----------------------------------------------------------------------------
    //  Unload service for plugins as well as the search service for automated
    //  plugin detection
    //  :TODO: Boost lockfree_queue
    // ----------------------------------------------------------------------------
    void threadEntryPoint()
    {
        boost::recursive_mutex::scoped_lock lock(mutex);

        while (!m_complete)
        {
            while (m_unloadQueue.empty()) m_noWorkCond.wait(lock);

            while (!m_unloadQueue.empty())
            {
                m_unloadQueue.front()->unload();

                m_unloadQueue.pop_front();
            }
        }
    }
};
    
// ----------------------------------------------------------------------------
//  Instantiates the pImpl member structure for the plugin manager
// ----------------------------------------------------------------------------
PluginManager & PluginManager::instance()
{ 
    static PluginManager pmanager;
    return pmanager;
}

// ----------------------------------------------------------------------------
//  Instantiates the pImpl member structure for the plugin manager
// ----------------------------------------------------------------------------
PluginManager::PluginManager() 
{ 
    m_pImpl = new Impl(); 
}
    
// ----------------------------------------------------------------------------
//  Deletes the pImple member structure and unloads all plugins
// ----------------------------------------------------------------------------
PluginManager::~PluginManager() 
{ 
    unloadAll(); 
    
    delete m_pImpl; 
}

// ----------------------------------------------------------------------------
//  Returns the plugin handle which keeps the plugin from being unloaded
// ----------------------------------------------------------------------------
std::shared_ptr<void> PluginManager::acquirePluginHandle()
{
    auto handle = m_pImpl->pluginHandle;

    m_pImpl->pluginHandle.reset();

    return handle;
}

// ----------------------------------------------------------------------------
//  Adds a path to the search paths for the program
// ----------------------------------------------------------------------------
void PluginManager::addPath(String const& string)
{
    boost::recursive_mutex::scoped_lock lock(m_pImpl->mutex);

    boost::filesystem::path path(string);

    path.normalize();

    m_pImpl->searchPaths.push_back(path);
}
    
// ----------------------------------------------------------------------------
//  Removes a path from the search paths for the program
// ----------------------------------------------------------------------------
void PluginManager::removePath(String const& string)
{
    boost::recursive_mutex::scoped_lock lock(m_pImpl->mutex);

    boost::filesystem::path path(string);

    path.normalize();

    m_pImpl->searchPaths.remove(path);
}   
    
// ----------------------------------------------------------------------------
//  Searches for any plugins in the set of search directories and
//  optionally enables them.
// ----------------------------------------------------------------------------
void PluginManager::search(bool load, bool checkBins)
{
    BOOST_FOREACH(auto & searchPath, m_pImpl->searchPaths)
    {
        // Verify the search path is valid before attempting to iterate
        if (!boost::filesystem::exists(searchPath)) 
        {
            VOX_LOG_WARNING(Error_NotFound, VOX_LOG_CATEGORY, format("Plugin search directory <%1%> not found.", searchPath)); 
            continue;
        }

        // Locate and assess all files in the plugin directory
        BOOST_FOREACH (auto & entry, boost::make_iterator_range(boost::filesystem::directory_iterator(searchPath), 
                                                                boost::filesystem::directory_iterator()))
        {
            // Check file extension for .pin or system's library extension
            if (entry.path().extension() == ".pin")
            {
                VOX_LOG_WARNING(Error_NotImplemented, VOX_LOG_CATEGORY, 
                    format("Unable to read <%1%>: PIN file support not implemented", entry.path().filename()));
            }
            else if (entry.path().extension() == ".dll" && checkBins)
            {
                try
                {
                    auto info = loadFromFile(entry.path().string(), load);
                }
                catch (Error & error)
                {
                    VOX_LOG_WARNING(error.code, VOX_LOG_CATEGORY, format("Attempt to load <%1%> as plugin failed: %2%", entry.path().string(), error.message));
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
//  Issues the specified callback for each available plugin
// ----------------------------------------------------------------------------
void PluginManager::forEach(PluginCallback callback)
{
    BOOST_FOREACH (auto & plugin, m_pImpl->plugins) callback(plugin->info());
}

// ----------------------------------------------------------------------------
//  Unloads any plugins loaded by the PluginManager (nothrow)
// ----------------------------------------------------------------------------
void PluginManager::unloadAll()
{
    boost::recursive_mutex::scoped_lock lock(m_pImpl->mutex);

    BOOST_FOREACH(auto & plugin, m_pImpl->plugins) plugin->unload();
}

// ----------------------------------------------------------------------------
//  Loads a specified plugin. If the plugin info file cannot be found
//  and the permissions are not available, an exception will be thrown.
// ----------------------------------------------------------------------------
std::shared_ptr<PluginInfo> PluginManager::loadFromFile(String const& file, bool enable)
{
    boost::recursive_mutex::scoped_lock lock(m_pImpl->mutex);

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

        if (enable) 
        {
            plugin->load(); 
            plugin->enable();
        }

        m_pImpl->plugins.push_back(plugin);

        return newInfo;
    }
    else
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
            format("Plugin file not found: %1%", file));
    }
}

// ----------------------------------------------------------------------------
//  Loads a specified plugin (ie enable or load-enable)
// ----------------------------------------------------------------------------
void PluginManager::load(std::shared_ptr<PluginInfo> const& info)
{
    if (!info) return;

    boost::recursive_mutex::scoped_lock lock(m_pImpl->mutex);
    
    // Unload the plugin but keep the info structure for reference
    BOOST_FOREACH(auto & plugin, m_pImpl->plugins)
    {
        if (info == plugin->info()) 
        {
            m_pImpl->buildPluginHandle(plugin);
            plugin->load(); 
            if (!m_pImpl->pluginHandle) plugin->enable();
            break;
        }
    }
    
    lock.unlock();

    m_pImpl->pluginHandle.reset();
}
        
// ----------------------------------------------------------------------------
//  Returns true if the specified plugin is 'loaded' (ie enabled)
// ----------------------------------------------------------------------------
bool PluginManager::isLoaded(std::shared_ptr<PluginInfo> const& info)
{
    boost::recursive_mutex::scoped_lock lock(m_pImpl->mutex);
    
    // Unload the plugin but keep the info structure for reference
    BOOST_FOREACH(auto & plugin, m_pImpl->plugins)
    {
        if (info == plugin->info()) { return plugin->isEnabled(); }
    }

    return false;
}

// ----------------------------------------------------------------------------
//  Unloads a loaded plugin if it is currently loaded
// ----------------------------------------------------------------------------
void PluginManager::unload(std::shared_ptr<PluginInfo> const& info)
{
    boost::recursive_mutex::scoped_lock lock(m_pImpl->mutex);
    
    // Unload the plugin but keep the info structure for reference
    BOOST_FOREACH(auto & plugin, m_pImpl->plugins)
    {
        if (info == plugin->info()) { plugin->disable(); return; }
    }
}

// ----------------------------------------------------------------------------
//  Performs a soft reset of an enabled plugin
// ----------------------------------------------------------------------------
void PluginManager::softReload(String const& pluginName)
{
    boost::recursive_mutex::scoped_lock lock(m_pImpl->mutex);

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

// ----------------------------------------------------------------------------
//  Loads a plugin based on a vendor and name. (Most recent compatible version)
// ----------------------------------------------------------------------------
std::shared_ptr<PluginInfo> PluginManager::findByNameVendor(String const& vendor, String const& name)
{
    return m_pImpl->getByVendorName(vendor, name);
}

// ----------------------------------------------------------------------------
//  Performs a hard reset of an enabled plugin - load and unload dll
// ----------------------------------------------------------------------------
void PluginManager::reload(String const& name)
{
    boost::recursive_mutex::scoped_lock lock(m_pImpl->mutex);

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