/* ===========================================================================

    Project: VoxRender - VoxIO Plugin Test Module

    Description: Performs unit testing for plugin management

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

#define BOOST_TEST_MODULE "Vox Plugin Unit Test Module"

// Include Boost UnitTest Framework
#include <boost/test/unit_test.hpp>

// Include Dependencies
#include "VoxLib/Core/Types.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/Plugin/Plugin.h"

using namespace vox;

// --------------------------------------------------------------------
//  Performs tests of the Plugin load and unload functionality
// --------------------------------------------------------------------
BOOST_AUTO_TEST_SUITE( PluginManagementSuite )

    // Tests the plugin load/unload functions
    BOOST_AUTO_TEST_CASE( Load_Unload_Test )
    {
        {
        Plugin plugin;
        BOOST_CHECK_THROW(plugin.open("MissingName"), Error);
        }

        {
        Plugin plugin;
        plugin.open("CudaRenderer.dll");
        plugin.close();
        }
    }

    // Tests the symbol lookup functionality
    BOOST_AUTO_TEST_CASE( Symbol_Lookup_Test )
    {
    }

    // Tests the info query functionality
    BOOST_AUTO_TEST_CASE( Info_Query_Test )
    {
    }
    
BOOST_AUTO_TEST_SUITE_END()