/* ===========================================================================

	Project: VoxRender - PluginContainer

	Description: Provides an extended plugin interface for the PluginManager

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

// TODO: Should expose some of this to the user, remove the pluginManager helper stuff // 

// Begin definition
#ifndef VOX_PLUGIN_CONTAINER_H
#define VOX_PLUGIN_CONTAINER_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Types.h"
#include "VoxLib/Plugin/PluginInfo.h"

// API namespace
namespace vox {
    
    // Plugin interface for current API version
    typedef char const* (*NameFunc)();
    typedef char const* (*VersionFunc)();
    typedef char const* (*VendorFunc)();
    typedef char const* (*ApiVersionMaxFunc)();
    typedef char const* (*ApiVersionMinFunc)();
    typedef char const* (*ReferenceUrlFunc)();
    typedef char const* (*DescriptionFunc)();
    typedef void        (*InitPluginFunc)();
    typedef void        (*FreePluginFunc)();
    typedef void        (*EnableFunc)();
    typedef void        (*DisableFunc)();
    
    /** Structure for loaded plugins in the manager */
    struct PluginContainer 
    {
        /** Initialize from file specification */
        PluginContainer(String const& filename) :
            m_isEnabled(false),
            m_initPluginFunc(nullptr),
            m_freePluginFunc(nullptr),
            m_enableFunc(nullptr),
            m_disableFunc(nullptr)
        {
            getInfo(filename);
        }

        /** Initialize from info structure */
        PluginContainer(std::shared_ptr<PluginInfo> info) :
            m_isEnabled(false),
            m_initPluginFunc(nullptr),
            m_freePluginFunc(nullptr),
            m_enableFunc(nullptr),
            m_disableFunc(nullptr)
        {
            m_info = info;
        }

        /** Ensure plugin disabled+unloaded */
        ~PluginContainer()
        {
            unload();
        }

        /** Reloads the internal plugin */
        void reload()
        {
            if (m_plugin.isOpen())
            {
                disable(); 

                m_freePluginFunc();

                m_plugin.reload();

                getPluginHandles();

                m_initPluginFunc();
            }
        }

        /** Loads the specified plugin */
        void load()
        {
            if (m_plugin.isOpen()) return;

            m_plugin.open(m_info->file);

            getPluginHandles();

            m_initPluginFunc();
        }

        /** Unloads the internal plugin */
        void unload()
        {
            if (m_plugin.isOpen())
            {
                disable();

                m_freePluginFunc();

                m_plugin.close();

                m_initPluginFunc = nullptr;
                m_freePluginFunc = nullptr;
                m_enableFunc     = nullptr;
                m_disableFunc    = nullptr;
            }
        }

        /** Performs a soft reload of the plugin */
        void softReload()
        {
            disable(); enable();
        }

        /** Enables the internal plugin */
        void enable()
        {
            if (!m_isEnabled)
            {
                m_isEnabled = true;
                m_enableFunc();
            }
        }

        /** Disables the internal plugin */
        void disable()
        {
            if (m_isEnabled)
            {
                m_isEnabled = false;
                m_disableFunc();
            }
        }
        
        /** Returns true if the plugin is enabled */
        bool isEnabled()
        {
            return m_isEnabled;
        }

        /** Returns the filesystem path of the plugin */
        String const& path()
        {
            return m_plugin.library();
        }

        /** Returns the info structure for this plugin */
        std::shared_ptr<PluginInfo> info()
        {
            return m_info;
        }

    private:
        Plugin m_plugin;
        bool   m_isEnabled;

        std::shared_ptr<PluginInfo> m_info;

        InitPluginFunc    m_initPluginFunc;
        FreePluginFunc    m_freePluginFunc;
        EnableFunc        m_enableFunc;
        DisableFunc       m_disableFunc;

        /** Extracts the plugin control function pointers */
        void getPluginHandles()
        {
            m_initPluginFunc = m_plugin.findSymbolAs<InitPluginFunc>("initPlugin");
            m_freePluginFunc = m_plugin.findSymbolAs<FreePluginFunc>("freePlugin");
            m_enableFunc     = m_plugin.findSymbolAs<EnableFunc>("enable");
            m_disableFunc    = m_plugin.findSymbolAs<DisableFunc>("disable");
        }

        /** Extracts the plugin info from the plugin dll without initiating it */
        void getInfo(String const& filename)
        {
            m_plugin.open(filename);

                // Extract the plugin info 
                m_info = std::make_shared<PluginInfo>();

                m_info->file = filename;

                m_info->name   = m_plugin.findSymbolAs<NameFunc>("name")();
                m_info->vendor = m_plugin.findSymbolAs<VendorFunc>("vendor")();
                m_info->url    = m_plugin.findSymbolAs<ReferenceUrlFunc>("referenceUrl")();
                
                m_info->description = m_plugin.findSymbolAs<DescriptionFunc>("description")();

                m_info->version       = parseVersionStr(m_plugin.findSymbolAs<VersionFunc>("version")());
                m_info->apiVersionMax = parseVersionStr(m_plugin.findSymbolAs<ApiVersionMaxFunc>("apiVersionMax")());
                m_info->apiVersionMin = parseVersionStr(m_plugin.findSymbolAs<ApiVersionMinFunc>("apiVersionMin")());
            
            m_plugin.close();
        }

        /** Parses a version string into a version structure */
        PluginInfo::Version parseVersionStr(String const& versionStr)
        {
            PluginInfo::Version version;

            std::vector<String> fields;
            boost::algorithm::split(fields, versionStr, boost::is_any_of("."));
            if (fields.size() > 2) version.patch = boost::lexical_cast<unsigned int>(fields[2]);
            if (fields.size() > 1) version.minor = boost::lexical_cast<unsigned int>(fields[1]);
            if (fields.size() > 0) version.major = boost::lexical_cast<unsigned int>(fields[0]);

            return version;
        }
    };

} // namespace vox

// End definition
#endif // VOX_PLUGIN_CONTAINER_H