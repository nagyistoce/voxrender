/* ===========================================================================

    Project: VoxRender - StandardIO Plugin Test Module

    Description: Performs unit testing for the Standard IO plugin

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

#define BOOST_TEST_MODULE "Vox StandardIO Unit Test Module"

// Include Boost UnitTest Framework
#include <boost/test/unit_test.hpp>

// Include Dependencies
#include "VoxLib/Core/Types.h"
#include "VoxLib/Core/Functors.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/Plugin/Plugin.h"
#include "VoxLib/IO/Resource.h"
#include "VoxLib/IO/ResourceHelper.h"
#include "VoxLib/Plugin/PluginManager.h"

#include "Plugins/StandardIO/StandardIO.h"

#include <boost/thread.hpp>
#include <boost/date_time.hpp>

using namespace vox;

// --------------------------------------------------------------------
//  Performs tests of the Plugin load and unload functionality
// --------------------------------------------------------------------
BOOST_AUTO_TEST_SUITE( StandardIOSuite )

    // Tests the plugins meta info functions
    BOOST_AUTO_TEST_CASE( MetaInfoTest )
    {
    }
    
    // Tests the plugins HTTP IO functionality
    BOOST_AUTO_TEST_CASE( HttpIOTest )
    {
        std::cout << "**** This test requires an external network connection. ****" << std::endl;

        // Load the vox.standard_io plugin
        auto & pluginManager = PluginManager::instance();
        pluginManager.loadFromFile("StandardIO.dll");

        // ftp://ftp.funet.fi/README
        // http://www.example.com
        // dict://dict.org/m:curl
        // gopher://gopher.quux.org:70/About%20This%20Server.txt
        // https://svn.boost.org/htdocs/site/boost.png

        ResourceId example("ftp://ftp.funet.fi/README");

        //std::ofstream os(example.extractFileName(), std::ios::binary); os << webpageStream.rdbuf();

        std::cout << ResourceHelper::pull(example);
    }

BOOST_AUTO_TEST_SUITE_END()
