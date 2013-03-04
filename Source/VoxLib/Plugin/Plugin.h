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

// Begin definition
#ifndef VOX_PLUGIN_H
#define VOX_PLUGIN_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/System.h"
#include "VoxLib/Core/Types.h"

// API namespace
namespace vox
{
	/** 
	 * @brief Plugin class
     *
     * This class manages a loaded library and provides simplified functionality
     * for symbol lookup.
	 */
    class VOX_EXPORT Plugin
	{
    public:
        /** Initializes a plugin not associated with a library */
        VOX_HOST Plugin() : m_handle(nullptr) {}

        /** Ensures library closure on destruction */
        VOX_HOST ~Plugin() { close(); }

        /** Loads the library at path */
        VOX_HOST Plugin(String const& library) : m_handle(nullptr)
        {
            open(library);
        }

        /** Accessor for the internal library path/name */
        VOX_HOST String const& library() const 
        { 
            return m_library; 
        }

        /** Returns true if the library is open */
        VOX_HOST bool isOpen() 
        { 
            return m_handle ? true : false; 
        }

        /** Explicitly opens a library */
        VOX_HOST void open(String const& library);

        /** Explicitly closes a plugin */
        VOX_HOST void close();

        /** Reloads the current library */
        VOX_HOST void reload();

        /** Reopen the handle to the library */
        Plugin& operator= (Plugin& rhs)
        {
            open(rhs.library());

            return *this;
        }

        /** Transfer management of the library handle */
        Plugin& operator= (Plugin&& rhs)
        {
            m_handle  = rhs.m_handle;
            m_library = rhs.m_library;

            return *this;
        }

        /** Finds the raw address of a symbol */
        void* findSymbol(String const& symbol);

        /** Performs a symbol lookup + type-cast and for the plugin */
        template<typename T>
        T findSymbolAs(String const& symbol)
        {
            void * handle = findSymbol(symbol);
            return (T)(handle);
        }

    private:
        String m_library; ///< Library name/path
        void * m_handle;  ///< Library handle
	};
}

// End definition
#endif // VOX_PLUGIN_H