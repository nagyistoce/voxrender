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

// Note: This is an internal implementation header // 

// Begin definition
#ifndef VOX_PLUGIN_CONTAINER_H
#define VOX_PLUGIN_CONTAINER_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"

// API namespace
namespace vox {

    // Plugin interface for current API version
    typedef char const* (*NameFunc)();
    typedef char const* (*VersionFunc)();
    typedef char const* (*VendorFunc)();
    typedef char const* (*ApiVersionMaxFunc)();
    typedef char const* (*ApiVersionMinFunc)();
    typedef char const* (*ReferenceUrlFunc)();
    typedef void        (*InitPluginFunc)();
    typedef void        (*FreePluginFunc)();
    typedef void        (*EnableFunc)();
    typedef void        (*DisableFunc)();
    
    /** Structure for loaded plugins in the manager */
    struct PluginContainer 
    {
        /** Initialize pointers to null */
        PluginContainer() :
            m_isEnabled(false),
            name(nullptr),
            version(nullptr),
            vendor(nullptr),
            apiVersionMax(nullptr),
            apiVersionMin(nullptr),
            referenceUrl(nullptr),
            m_initPluginFunc(nullptr),
            m_freePluginFunc(nullptr),
            m_enableFunc(nullptr),
            m_disableFunc(nullptr)
        {
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

                m_initPluginFunc();

                getSymbols();
            }
        }

        /** Loads the specified plugin */
        void load(String const& path)
        {
            unload(); // Ensure plugin is unloaded 

            m_plugin.open(path);

            getSymbols();

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

                name          = nullptr;
                version       = nullptr;
                vendor        = nullptr;
                apiVersionMax = nullptr;
                apiVersionMin = nullptr;
                referenceUrl  = nullptr;

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
        
        /** Returns the filesystem path of the plugin */
        String const& path()
        {
            return m_plugin.library();
        }

        NameFunc          name;
        VersionFunc       version;
        VendorFunc        vendor;
        ApiVersionMaxFunc apiVersionMax;
        ApiVersionMinFunc apiVersionMin;
        ReferenceUrlFunc  referenceUrl;

    private:
        Plugin m_plugin;
        bool   m_isEnabled;
        
        InitPluginFunc    m_initPluginFunc;
        FreePluginFunc    m_freePluginFunc;
        EnableFunc        m_enableFunc;
        DisableFunc       m_disableFunc;

        /** Acquires the required plugin API symbols */
        void getSymbols()
        {
            name          = m_plugin.findSymbolAs<NameFunc>("name");
            version       = m_plugin.findSymbolAs<VersionFunc>("version");
            vendor        = m_plugin.findSymbolAs<VendorFunc>("vendor");
            apiVersionMax = m_plugin.findSymbolAs<ApiVersionMaxFunc>("apiVersionMax");
            apiVersionMin = m_plugin.findSymbolAs<ApiVersionMinFunc>("apiVersionMin");
            referenceUrl  = m_plugin.findSymbolAs<ReferenceUrlFunc>("referenceUrl");
            
            m_initPluginFunc = m_plugin.findSymbolAs<InitPluginFunc>("initPlugin");
            m_freePluginFunc = m_plugin.findSymbolAs<FreePluginFunc>("freePlugin");
            m_enableFunc     = m_plugin.findSymbolAs<EnableFunc>("enable");
            m_disableFunc    = m_plugin.findSymbolAs<DisableFunc>("disable");
        }
    };

} // namespace vox

// End definition
#endif // VOX_PLUGIN_CONTAINER_H