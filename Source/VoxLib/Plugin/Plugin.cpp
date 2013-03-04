/* ===========================================================================

	Project: VoxRender - Plugin

	Description: Provides management for handling runtime linked libraries

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
#include "Plugin.h"

// Include Dependencies
#include "VoxLib/Core/Format.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/Error/ErrorCodes.h"
#include "VoxLib/Error/SystemError.h"

// Additional dependencies
#include <boost/filesystem.hpp>

// Include OS specific required headers
#if defined _WIN32
#   define NOMINMAX
#   include <windows.h>
#   include <LMCons.h>
#elif defined __linux__
#   include <unistd.h>
#   include <sys/types.h>
#   include <dlfcn.h>
#endif

namespace vox {

namespace {
namespace filescope {

    // --------------------------------------------------------------------
    //  Opens and returns a handle to the specified library
    // --------------------------------------------------------------------
    void* openLibrary(String const& library)
    {
    #ifdef _WIN32
        return LoadLibrary(boost::filesystem::path(library).string().c_str());
    #elif defined __linux__
        return dlopen(boost::filesystem::path(library).string().c_str(), RTLD_LAZY);
    #else
    #   pragma message "WARNING openLibrary not implemented for this system"
    #endif

        return nullptr;
    }

    // --------------------------------------------------------------------
    //  Closes a handle to a library loaded with openLibrary
    // --------------------------------------------------------------------
    int closeLibrary(void* handle)
    {
    #ifdef _WIN32
        return FreeLibrary(reinterpret_cast<HMODULE>(handle)) ? 0 : 1;
    #elif defined __linux__
        return dlclose(handle);
    #else
    #   pragma message "WARNING closeLibrary not implemented for this system"
    #endif

        return 0;
    }

    // --------------------------------------------------------------------
    //  Performs symbol lookup for the specified library
    // --------------------------------------------------------------------
    void* findSymbol(void* handle, String const& symbol)
    {
    #ifdef _WIN32
        return GetProcAddress(reinterpret_cast<HMODULE>(handle), symbol.c_str());
    #elif defined __linux__
        return dlsym(handle, symbol.c_str()); 
    #else
    #   pragma message "WARNING closeLibrary not implemented for this system"
    #endif

        return nullptr;
    }

} // namespace filescope
} // namespace anonymous

// --------------------------------------------------------------------
//  Attempts to open the specified shared library 
// --------------------------------------------------------------------
void Plugin::open(String const& library)
{
    if (m_handle)
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                    format("Attempt to open <%1%> failed.", library),
                    Error_NotAllowed);
    }

    if (!(m_handle = filescope::openLibrary(library)))
    {
        throw SystemError(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                          format("Attempt to open <%1%> failed.", library));
    }
    else m_library = library;
}

// --------------------------------------------------------------------
//  Attempts to close the previously opened library
// --------------------------------------------------------------------
void Plugin::close()
{
    if (m_handle)
    {
        if (filescope::closeLibrary(m_handle) != 0)
        {
            // Can't unload library but the error is likely benign and we can't throw
            Logger::addEntry(Severity_Warning, Error_System, VOX_LOG_CATEGORY,
                System::formatError(System::getLastError()), __FILE__, __LINE__);
        }

        m_handle = nullptr;
        m_library.clear();
    }
}

// --------------------------------------------------------------------
//  Reloads the shared library if one is currently loaded
// --------------------------------------------------------------------
void Plugin::reload()
{
    if (m_handle)
    {
        if (filescope::closeLibrary(m_handle) != 0)
        {
            Logger::addEntry(Severity_Warning, Error_System, VOX_LOG_CATEGORY,
                System::formatError(System::getLastError()), __FILE__, __LINE__);
        }

        if (!(m_handle = filescope::openLibrary(m_library)))
        {
            m_library.clear();

            throw SystemError(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                              format("Attempt to reload <%1%> failed.", m_library));
        }
    }
}

// --------------------------------------------------------------------
//  Performs symbol lookup for the currently open library
// --------------------------------------------------------------------
void* Plugin::findSymbol(String const& symbol)
{
    if (!m_handle)
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                    "Plugin is not current open", Error_NotAllowed);
    }

    void * ptr = filescope::findSymbol(m_handle, symbol);

    if (!ptr)
    {
        throw SystemError(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                          "Symbol lookup failed");
    }

    return ptr;
}

} // namespace vox